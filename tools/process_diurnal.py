#!/usr/bin/env python3
"""
Diurnal Temperature Map Generator

Processes multiple forecast hours from a single HRRR model run to generate
diurnal temperature analysis products.

Usage:
    python tools/process_diurnal.py [DATE] [HOUR] [OPTIONS]

Examples:
    # Process latest available model run with default 24-hour window
    python tools/process_diurnal.py --latest

    # Process specific run with custom hour range
    python tools/process_diurnal.py 20251224 12 --start-fhr 0 --end-fhr 24

    # Process all diurnal products for a 48-hour forecast
    python tools/process_diurnal.py 20251224 12 --end-fhr 48

Author: HRRR Maps Pipeline
Version: 1.0
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Use non-interactive matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import xarray as xr

from smart_hrrr.processor_core import HRRRProcessor
from smart_hrrr.availability import get_latest_cycle
from derived_params.diurnal_temperature import (
    diurnal_temperature_range,
    diurnal_max_temperature,
    diurnal_min_temperature,
    diurnal_mean_temperature,
    day_night_temperature_difference,
    heating_rate,
    cooling_rate,
    hour_of_maximum_temperature,
    hour_of_minimum_temperature,
    diurnal_temperature_amplitude,
    compute_all_diurnal_products
)
from config.colormaps import create_all_colormaps
from field_registry import FieldRegistry
from multiprocessing import Pool
from functools import partial


def _create_gif(image_paths: List[Path], output_path: Path, duration: int = 250):
    """Create animated GIF from list of images"""
    from PIL import Image

    if not image_paths:
        return None

    # Sort by filename to ensure correct order
    image_paths = sorted(image_paths)

    # Load all images
    images = []
    for path in image_paths:
        if Path(path).exists():
            img = Image.open(path)
            images.append(img)

    if not images:
        return None

    # Save as animated GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,  # ms per frame
        loop=0  # infinite loop
    )

    return output_path


def _plot_worker(args):
    """Worker function for parallel plotting - must be at module level for pickling"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data, product_name, config, cycle_str, fhr_range, output_file, lons, lats, colormaps = args

    try:
        # Get colormap
        cmap_name = config.get('cmap', 'viridis')
        if cmap_name in colormaps:
            cmap = colormaps[cmap_name]
        else:
            cmap = plt.get_cmap(cmap_name)

        # Create figure with cartopy projection
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
            central_longitude=-96, central_latitude=39
        ))

        # Set extent to CONUS
        ax.set_extent([-125, -66, 22, 50], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')

        # Plot data - use pcolormesh for speed
        levels = config.get('levels', 10)
        extend = config.get('extend', 'both')

        cf = ax.pcolormesh(
            lons, lats, data,
            cmap=cmap,
            vmin=min(levels) if isinstance(levels, list) else None,
            vmax=max(levels) if isinstance(levels, list) else None,
            transform=ccrs.PlateCarree(),
            shading='auto'
        )

        # Add colorbar
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                          pad=0.05, shrink=0.8, aspect=40)
        cbar.set_label(f"{config['title']} ({config['units']})", fontsize=10)

        # Title
        title = f"{config['title']}\n{cycle_str} | {fhr_range}"
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Save
        plt.savefig(output_file, dpi=120, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        return str(output_file)

    except Exception as e:
        print(f"  Error in worker for {product_name}: {e}")
        return None


# Diurnal product configurations
DIURNAL_PRODUCTS = {
    'dtr': {
        'title': 'Diurnal Temperature Range',
        'units': '°C',
        'cmap': 'DiurnalRange',
        'levels': [0, 5, 8, 10, 12, 15, 18, 20, 25, 30],
        'extend': 'max',
        'category': 'diurnal',
        'function': diurnal_temperature_range,
        'description': 'Daily temperature range (T_max - T_min)'
    },
    't_max_diurnal': {
        'title': 'Maximum Temperature (Forecast Period)',
        'units': '°C',
        'cmap': 'RdYlBu_r',
        'levels': [-20, -10, 0, 10, 20, 25, 30, 35, 40, 45],
        'extend': 'both',
        'category': 'diurnal',
        'function': diurnal_max_temperature,
        'description': 'Maximum 2m temperature over forecast period'
    },
    't_min_diurnal': {
        'title': 'Minimum Temperature (Forecast Period)',
        'units': '°C',
        'cmap': 'RdYlBu_r',
        'levels': [-30, -20, -10, 0, 5, 10, 15, 20, 25, 30],
        'extend': 'both',
        'category': 'diurnal',
        'function': diurnal_min_temperature,
        'description': 'Minimum 2m temperature over forecast period'
    },
    't_mean_diurnal': {
        'title': 'Mean Temperature (Forecast Period)',
        'units': '°C',
        'cmap': 'RdYlBu_r',
        'levels': [-20, -10, 0, 10, 15, 20, 25, 30, 35, 40],
        'extend': 'both',
        'category': 'diurnal',
        'function': diurnal_mean_temperature,
        'description': 'Mean 2m temperature over forecast period'
    },
    'day_night_diff': {
        'title': 'Day-Night Temperature Difference',
        'units': '°C',
        'cmap': 'DiurnalRange',
        'levels': [0, 5, 8, 10, 12, 15, 18, 20, 25, 30],
        'extend': 'max',
        'category': 'diurnal',
        'function': day_night_temperature_difference,
        'description': 'Afternoon minus overnight temperature'
    },
    'heating_rate': {
        'title': 'Morning Heating Rate',
        'units': '°C/hr',
        'cmap': 'HeatingRate',
        'levels': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        'extend': 'max',
        'category': 'diurnal',
        'function': heating_rate,
        'description': 'Rate of temperature increase during morning'
    },
    'cooling_rate': {
        'title': 'Evening Cooling Rate',
        'units': '°C/hr',
        'cmap': 'CoolingRate',
        'levels': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        'extend': 'max',
        'category': 'diurnal',
        'function': cooling_rate,
        'description': 'Rate of temperature decrease during evening'
    },
    'diurnal_amplitude': {
        'title': 'Diurnal Temperature Amplitude',
        'units': '°C',
        'cmap': 'DiurnalRange',
        'levels': [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20],
        'extend': 'max',
        'category': 'diurnal',
        'function': diurnal_temperature_amplitude,
        'description': 'Half the diurnal range'
    },
    'hour_of_max_temp': {
        'title': 'Hour of Maximum Temperature',
        'units': 'forecast hour',
        'cmap': 'viridis',
        'levels': [6, 9, 12, 15, 18, 21, 24],
        'extend': 'both',
        'category': 'diurnal',
        'function': hour_of_maximum_temperature,
        'description': 'Forecast hour when T_max occurs'
    },
    'hour_of_min_temp': {
        'title': 'Hour of Minimum Temperature',
        'units': 'forecast hour',
        'cmap': 'viridis',
        'levels': [0, 3, 6, 9, 12, 15, 18],
        'extend': 'both',
        'category': 'diurnal',
        'function': hour_of_minimum_temperature,
        'description': 'Forecast hour when T_min occurs'
    }
}


class DiurnalProcessor:
    """Process multiple forecast hours to generate diurnal temperature products."""

    def __init__(self, model: str = 'hrrr'):
        """Initialize the diurnal processor."""
        self.processor = HRRRProcessor(model=model)
        self.colormaps = create_all_colormaps()
        self.registry = FieldRegistry()
        self.regions = {
            'conus': {
                'name': 'CONUS',
                'extent': [-130, -65, 20, 50],
                'barb_thinning': 60
            }
        }

    def load_temperature_data(
        self,
        cycle: datetime,
        forecast_hours: List[int],
        output_dir: Path
    ) -> Dict[int, np.ndarray]:
        """
        Load 2m temperature data for multiple forecast hours.

        Args:
            cycle: Model initialization time (datetime object)
            forecast_hours: List of forecast hours to load
            output_dir: Directory for downloaded GRIB files

        Returns:
            Dictionary mapping forecast hour -> temperature array (°C)
        """
        temp_data = {}
        reference_coords = None

        # Convert cycle to string format expected by downloader
        cycle_str = cycle.strftime('%Y%m%d%H')

        print(f"Loading temperature data for {len(forecast_hours)} forecast hours...")

        for fhr in forecast_hours:
            try:
                # Download GRIB file for this forecast hour
                grib_file = self.processor.download_model_file(
                    cycle_str, fhr, output_dir, file_type='pressure'
                )

                if not grib_file or not grib_file.exists():
                    print(f"  Skipping f{fhr:02d} - GRIB file not available")
                    continue

                # Load 2m temperature field using registry config
                t2m_config = self.registry.get_field('t2m')
                if t2m_config is None:
                    # Fallback to direct cfgrib access config
                    t2m_config = {
                        'var': 't2m',
                        'access': {'typeOfLevel': 'heightAboveGround', 'level': 2, 'paramId': 167},
                        'transform': 'celsius'
                    }

                data = self.processor.load_field_data(grib_file, 't2m', t2m_config)

                if data is None:
                    print(f"  Skipping f{fhr:02d} - could not load temperature")
                    continue

                # Store coordinates from first successful load
                if reference_coords is None:
                    reference_coords = data.coords
                    self._reference_data = data

                # Convert to numpy and store
                temp_array = data.values

                # Apply Kelvin to Celsius conversion if needed
                if np.nanmean(temp_array) > 200:  # Likely in Kelvin
                    temp_array = temp_array - 273.15

                temp_data[fhr] = temp_array.astype(np.float32)
                print(f"  Loaded f{fhr:02d}: {np.nanmin(temp_array):.1f} to {np.nanmax(temp_array):.1f}°C")

            except Exception as e:
                print(f"  Error loading f{fhr:02d}: {e}")
                continue

        print(f"Successfully loaded {len(temp_data)} forecast hours")
        return temp_data

    def compute_diurnal_products(
        self,
        temp_data: Dict[int, np.ndarray],
        cycle_hour: int,
        products: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute diurnal temperature products.

        Args:
            temp_data: Dictionary mapping forecast hour -> temperature array
            cycle_hour: Model initialization hour (UTC)
            products: List of product names to compute (None = all)

        Returns:
            Dictionary mapping product name -> computed array
        """
        if products is None:
            products = list(DIURNAL_PRODUCTS.keys())

        results = {}

        for product_name in products:
            if product_name not in DIURNAL_PRODUCTS:
                print(f"Unknown product: {product_name}")
                continue

            config = DIURNAL_PRODUCTS[product_name]
            func = config['function']

            try:
                print(f"Computing {product_name}...")

                # Special handling for functions that need extra args
                if product_name == 'day_night_diff':
                    result = func(temp_data, cycle_hour)
                elif product_name == 'heating_rate':
                    result = func(temp_data, start_hour=6, end_hour=12)
                elif product_name == 'cooling_rate':
                    result = func(temp_data, start_hour=18, end_hour=24)
                else:
                    result = func(temp_data)

                results[product_name] = result
                print(f"  {product_name}: {np.nanmin(result):.2f} to {np.nanmax(result):.2f}")

            except Exception as e:
                print(f"  Error computing {product_name}: {e}")
                continue

        return results

    def create_diurnal_plot(
        self,
        data: np.ndarray,
        product_name: str,
        cycle: datetime,
        fhr_range: str,
        output_dir: Path
    ) -> Optional[Path]:
        """
        Create a plot for a diurnal product using direct matplotlib/cartopy.

        Args:
            data: 2D array of computed values
            product_name: Name of the diurnal product
            cycle: Model initialization time
            fhr_range: String describing forecast hour range (e.g., "f00-f24")
            output_dir: Output directory for plots

        Returns:
            Path to created plot, or None if failed
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if product_name not in DIURNAL_PRODUCTS:
            return None

        config = DIURNAL_PRODUCTS[product_name].copy()

        # Create output subdirectory
        diurnal_dir = output_dir / 'diurnal'
        diurnal_dir.mkdir(parents=True, exist_ok=True)

        # Generate plot filename
        date_str = cycle.strftime('%Y%m%d')
        hour_str = f"{cycle.hour:02d}z"
        output_file = diurnal_dir / f"{product_name}_{date_str}_{hour_str}_{fhr_range}.png"

        try:
            # Get coordinates from reference data
            if hasattr(self, '_reference_data') and self._reference_data is not None:
                lons = self._reference_data.longitude.values
                lats = self._reference_data.latitude.values
            else:
                # Fallback - create approximate CONUS grid
                lons = np.linspace(-130, -65, data.shape[1])
                lats = np.linspace(20, 50, data.shape[0])

            # Get colormap
            cmap_name = config.get('cmap', 'viridis')
            if cmap_name in self.colormaps:
                cmap = self.colormaps[cmap_name]
            else:
                cmap = plt.get_cmap(cmap_name)

            # Create figure with cartopy projection
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
                central_longitude=-96, central_latitude=39
            ))

            # Set extent to CONUS
            ax.set_extent([-125, -66, 22, 50], crs=ccrs.PlateCarree())

            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')

            # Plot data
            levels = config.get('levels', 10)
            extend = config.get('extend', 'both')

            cf = ax.contourf(
                lons, lats, data,
                levels=levels,
                cmap=cmap,
                extend=extend,
                transform=ccrs.PlateCarree()
            )

            # Add colorbar
            cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                              pad=0.05, shrink=0.8, aspect=40)
            cbar.set_label(f"{config['title']} ({config['units']})", fontsize=10)

            # Title
            title = f"{config['title']}\n{cycle.strftime('%Y-%m-%d %HZ')} | {fhr_range}"
            ax.set_title(title, fontsize=12, fontweight='bold')

            # Save
            plt.savefig(output_file, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            print(f"  Created: {output_file.name}")
            return output_file

        except Exception as e:
            print(f"  Error creating plot for {product_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_diurnal(
        self,
        cycle: datetime,
        start_fhr: int = 0,
        end_fhr: int = 24,
        output_dir: Path = None,
        products: List[str] = None,
        rolling: bool = False,
        workers: int = 1,
        make_gif: bool = False
    ) -> Dict[str, Path]:
        """
        Main processing function for diurnal temperature products.

        Args:
            cycle: Model initialization time
            start_fhr: Starting forecast hour
            end_fhr: Ending forecast hour
            output_dir: Output directory
            products: List of products to generate (None = all)
            rolling: Generate rolling 24h windows

        Returns:
            Dictionary mapping product name -> output file path
        """
        if output_dir is None:
            output_dir = Path('outputs') / 'hrrr' / cycle.strftime('%Y%m%d') / f"{cycle.hour:02d}z"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate list of forecast hours
        forecast_hours = list(range(start_fhr, end_fhr + 1))

        print(f"\n{'='*60}")
        print(f"DIURNAL TEMPERATURE PROCESSING")
        print(f"{'='*60}")
        print(f"Cycle: {cycle.strftime('%Y-%m-%d %HZ')}")
        print(f"Forecast hours: f{start_fhr:02d} to f{end_fhr:02d}")
        if rolling:
            print(f"Mode: Rolling 24h windows")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

        # Load temperature data for all forecast hours
        temp_data = self.load_temperature_data(cycle, forecast_hours, output_dir)

        if len(temp_data) < 3:
            print("ERROR: Insufficient data loaded (need at least 3 hours)")
            return {}

        output_files = {}

        # Create output subdirectory
        diurnal_dir = output_dir / 'diurnal'
        diurnal_dir.mkdir(parents=True, exist_ok=True)

        if rolling:
            # Generate rolling 24h windows
            available_hours = sorted(temp_data.keys())
            min_hr, max_hr = min(available_hours), max(available_hours)

            # Each window is 24 hours
            window_size = 24
            num_windows = max_hr - min_hr - window_size + 2  # +2 because inclusive

            if num_windows < 1:
                print(f"ERROR: Need at least {window_size}h of data for rolling windows")
                return {}

            print(f"\nComputing {num_windows} rolling 24h windows...")

            # Get coordinates for plotting
            if hasattr(self, '_reference_data') and self._reference_data is not None:
                lons = self._reference_data.longitude.values
                lats = self._reference_data.latitude.values
            else:
                # Fallback
                sample_shape = list(temp_data.values())[0].shape
                lons = np.linspace(-130, -65, sample_shape[1])
                lats = np.linspace(20, 50, sample_shape[0])

            # Collect all plot jobs
            plot_jobs = []
            cycle_str = cycle.strftime('%Y-%m-%d %HZ')

            for window_start in range(min_hr, max_hr - window_size + 2):
                window_end = window_start + window_size

                # Subset temperature data for this window
                window_data = {h: temp_data[h] for h in range(window_start, window_end + 1)
                              if h in temp_data}

                if len(window_data) < 20:  # Need most of the 24h window
                    print(f"  Skipping f{window_start:02d}-f{window_end:02d} - insufficient data")
                    continue

                fhr_range = f"f{window_start:02d}-f{window_end:02d}"

                # Compute diurnal products for this window
                computed = self.compute_diurnal_products(
                    window_data,
                    cycle_hour=cycle.hour,
                    products=products
                )

                # Queue plots for this window
                for product_name, data in computed.items():
                    config = DIURNAL_PRODUCTS[product_name].copy()
                    date_str = cycle.strftime('%Y%m%d')
                    hour_str = f"{cycle.hour:02d}z"
                    output_file = diurnal_dir / f"{product_name}_{date_str}_{hour_str}_{fhr_range}.png"

                    plot_jobs.append((
                        data, product_name, config, cycle_str, fhr_range,
                        str(output_file), lons, lats, self.colormaps
                    ))

            print(f"\nGenerating {len(plot_jobs)} plots with {workers} workers...")

            # Execute plots in parallel or serial
            if workers > 1:
                with Pool(workers) as pool:
                    results = pool.map(_plot_worker, plot_jobs)
                for r in results:
                    if r:
                        output_files[Path(r).stem] = Path(r)
                        print(f"  Created: {Path(r).name}")
            else:
                for job in plot_jobs:
                    result = _plot_worker(job)
                    if result:
                        output_files[Path(result).stem] = Path(result)
                        print(f"  Created: {Path(result).name}")

            # Generate GIFs if requested
            if make_gif:
                print(f"\nGenerating animated GIFs...")
                product_list = products or ['dtr']
                for product_name in product_list:
                    if product_name in DIURNAL_PRODUCTS:
                        # Find all PNGs for this product
                        pattern = f"{product_name}_*.png"
                        png_files = sorted(diurnal_dir.glob(pattern))
                        if png_files:
                            gif_path = diurnal_dir / f"{product_name}_{cycle.strftime('%Y%m%d')}_{cycle.hour:02d}z_rolling.gif"
                            result = _create_gif(png_files, gif_path, duration=300)
                            if result:
                                print(f"  Created: {gif_path.name} ({len(png_files)} frames)")
                                output_files[f"{product_name}_gif"] = gif_path

            # Write metadata
            metadata = {
                'cycle': cycle.isoformat(),
                'start_fhr': start_fhr,
                'end_fhr': end_fhr,
                'mode': 'rolling_24h',
                'windows_generated': num_windows,
                'hours_loaded': sorted(list(temp_data.keys())),
                'products_per_window': products or list(DIURNAL_PRODUCTS.keys()),
                'generated_at': datetime.utcnow().isoformat()
            }
        else:
            # Single window mode (original behavior)
            computed = self.compute_diurnal_products(
                temp_data,
                cycle_hour=cycle.hour,
                products=products
            )

            fhr_range = f"f{start_fhr:02d}-f{end_fhr:02d}"

            print(f"\nGenerating plots...")
            for product_name, data in computed.items():
                output_file = self.create_diurnal_plot(
                    data, product_name, cycle, fhr_range, output_dir
                )
                if output_file:
                    output_files[product_name] = output_file

            metadata = {
                'cycle': cycle.isoformat(),
                'start_fhr': start_fhr,
                'end_fhr': end_fhr,
                'mode': 'single_window',
                'hours_loaded': sorted(list(temp_data.keys())),
                'products_generated': list(output_files.keys()),
                'generated_at': datetime.utcnow().isoformat()
            }

        # Write metadata
        metadata_file = output_dir / 'diurnal' / f"diurnal_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*60}")
        print(f"DIURNAL PROCESSING COMPLETE")
        print(f"Generated {len(output_files)} products")
        print(f"Output directory: {output_dir / 'diurnal'}")
        print(f"{'='*60}\n")

        return output_files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate diurnal temperature maps from HRRR data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --latest                        Process latest available cycle
  %(prog)s 20251224 12                     Process Dec 24, 2025 12Z run
  %(prog)s --latest --end-fhr 48           Process 48-hour forecast
  %(prog)s --latest --products dtr heating_rate cooling_rate
        """
    )

    parser.add_argument('date', nargs='?', help='Date in YYYYMMDD format')
    parser.add_argument('hour', nargs='?', type=int, help='Cycle hour (0, 6, 12, 18)')
    parser.add_argument('--latest', action='store_true', help='Use latest available cycle')
    parser.add_argument('--synoptic', action='store_true',
                       help='Only use synoptic cycles (00/06/12/18Z) which have 48h forecasts')
    parser.add_argument('--start-fhr', type=int, default=0, help='Starting forecast hour (default: 0)')
    parser.add_argument('--end-fhr', type=int, default=24, help='Ending forecast hour (default: 24)')
    parser.add_argument('--rolling', action='store_true',
                       help='Generate rolling 24h windows (e.g., f00-f24, f01-f25, ... f24-f48)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers for plotting (default: 1)')
    parser.add_argument('--gif', action='store_true',
                       help='Generate animated GIF from rolling windows')
    parser.add_argument('--output', '-o', type=Path, help='Output directory')
    parser.add_argument('--model', default='hrrr', choices=['hrrr', 'rrfs'], help='Weather model')
    parser.add_argument('--products', nargs='+', help='Specific products to generate')
    parser.add_argument('--list-products', action='store_true', help='List available products')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # List products and exit
    if args.list_products:
        print("\nAvailable Diurnal Products:")
        print("-" * 60)
        for name, config in DIURNAL_PRODUCTS.items():
            print(f"  {name:20s} - {config['title']}")
            print(f"                       {config['description']}")
        print()
        return 0

    # Determine cycle time
    if args.latest:
        cycle_str, cycle = get_latest_cycle(model=args.model)
        if cycle is None:
            print("ERROR: Could not find latest available cycle")
            return 1

        # If --synoptic, find the most recent synoptic cycle (00/06/12/18Z)
        if args.synoptic and cycle.hour not in (0, 6, 12, 18):
            from datetime import timedelta
            # Roll back to previous synoptic hour
            synoptic_hours = [0, 6, 12, 18]
            current_hour = cycle.hour
            for back in range(1, 7):
                check_hour = (current_hour - back) % 24
                if check_hour in synoptic_hours:
                    cycle = cycle.replace(hour=check_hour)
                    if back > current_hour:  # Crossed midnight
                        cycle = cycle - timedelta(days=1)
                    cycle_str = cycle.strftime('%Y%m%d%H')
                    print(f"Using latest synoptic cycle: {cycle_str} (48h forecasts available)")
                    break
        else:
            print(f"Using latest available cycle: {cycle_str} ({cycle})")

        # Warn if not a synoptic cycle and requesting >18h
        if cycle.hour not in (0, 6, 12, 18) and args.end_fhr > 18:
            print(f"WARNING: {cycle.hour:02d}Z only has 18h forecasts. Use --synoptic for 48h.")
            print(f"         Limiting end_fhr to 18")
            args.end_fhr = 18
    elif args.date and args.hour is not None:
        try:
            cycle = datetime.strptime(f"{args.date}{args.hour:02d}", '%Y%m%d%H')
        except ValueError:
            print(f"ERROR: Invalid date/hour format: {args.date} {args.hour}")
            return 1

        # Warn if not a synoptic cycle and requesting >18h
        if args.hour not in (0, 6, 12, 18) and args.end_fhr > 18:
            print(f"WARNING: {args.hour:02d}Z only has 18h forecasts. Limiting to 18h.")
            args.end_fhr = 18
    else:
        print("ERROR: Must specify either --latest or DATE HOUR")
        return 1

    # Initialize processor and run
    processor = DiurnalProcessor(model=args.model)

    try:
        results = processor.process_diurnal(
            cycle=cycle,
            start_fhr=args.start_fhr,
            end_fhr=args.end_fhr,
            output_dir=args.output,
            products=args.products,
            rolling=args.rolling,
            workers=args.workers,
            make_gif=args.gif
        )

        if results:
            print(f"Successfully generated {len(results)} diurnal products")
            return 0
        else:
            print("No products generated")
            return 1

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
