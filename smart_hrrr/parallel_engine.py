# Move the current OptimizedHRRRProcessor, generate_single_map, and process_hrrr_parallel here.
# Keep the public function signature identical to processor_batch.process_hrrr_parallel.

import os
import sys
import time
import pickle
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count
import shutil

from .processor_core import HRRRProcessor
from field_registry import FieldRegistry
from derived_params import compute_derived_parameter


class OptimizedHRRRProcessor(HRRRProcessor):
    """Optimized weather model processor that loads all fields once"""
    
    def __init__(self, model='hrrr'):
        super().__init__(model=model)
        self._all_base_fields = {}
        self._all_derived_fields = {}
        self.num_workers = min(cpu_count() - 1, 8)  # Leave one CPU free, max 8 workers
        
    def load_all_base_fields(self, pressure_grib_file, surface_grib_file=None):
        """Load all base (non-derived) fields from GRIB files into memory"""
        print("\n🚀 OPTIMIZED: Loading all base fields into memory...")
        start_time = time.time()
        
        # Get all field configurations
        all_fields = self.registry.load_all_fields()
        base_fields = {k: v for k, v in all_fields.items() if not v.get('derived')}
        
        print(f"📊 Found {len(base_fields)} base fields to load")
        
        # Group fields by source file
        pressure_fields = {}
        surface_fields = {}
        wrfsfc_fields = {}
        
        for field_name, field_config in base_fields.items():
            if field_config.get('source') == 'wrfsfc':
                wrfsfc_fields[field_name] = field_config
            elif field_config.get('level_type') == 'surface':
                surface_fields[field_name] = field_config
            else:
                pressure_fields[field_name] = field_config
        
        # Load from pressure file
        if pressure_fields and pressure_grib_file and os.path.exists(pressure_grib_file):
            print(f"\n📂 Loading {len(pressure_fields)} fields from pressure file...")
            for field_name, field_config in pressure_fields.items():
                try:
                    print(f"  Loading {field_name}...", end='', flush=True)
                    data = self.load_field_data(pressure_grib_file, field_name, field_config)
                    if data is not None:
                        self._all_base_fields[field_name] = data
                        print(" ✓")
                    else:
                        print(" ✗")
                except Exception as e:
                    print(f" ✗ ({str(e)})")
        
        # Load from surface file
        if surface_fields and surface_grib_file and os.path.exists(surface_grib_file):
            print(f"\n📂 Loading {len(surface_fields)} fields from surface file...")
            for field_name, field_config in surface_fields.items():
                try:
                    print(f"  Loading {field_name}...", end='', flush=True)
                    data = self.load_field_data(surface_grib_file, field_name, field_config)
                    if data is not None:
                        self._all_base_fields[field_name] = data
                        print(" ✓")
                    else:
                        print(" ✗")
                except Exception as e:
                    print(f" ✗ ({str(e)})")
        
        # Load wrfsfc fields (try surface file first, fall back to pressure)
        if wrfsfc_fields:
            print(f"\n📂 Loading {len(wrfsfc_fields)} wrfsfc fields...")
            for field_name, field_config in wrfsfc_fields.items():
                try:
                    print(f"  Loading {field_name}...", end='', flush=True)
                    data = None
                    if surface_grib_file and os.path.exists(surface_grib_file):
                        data = self.load_field_data(surface_grib_file, field_name, field_config)
                    if data is None and pressure_grib_file and os.path.exists(pressure_grib_file):
                        data = self.load_field_data(pressure_grib_file, field_name, field_config)
                    
                    if data is not None:
                        self._all_base_fields[field_name] = data
                        print(" ✓")
                    else:
                        print(" ✗")
                except Exception as e:
                    print(f" ✗ ({str(e)})")
        
        load_time = time.time() - start_time
        print(f"\n✅ Loaded {len(self._all_base_fields)} base fields in {load_time:.1f}s")
        
        # Print memory usage estimate
        total_size = sum(field.nbytes for field in self._all_base_fields.values())
        print(f"💾 Estimated memory usage: {total_size / 1024 / 1024:.1f} MB")
        
        return self._all_base_fields
    
    def get_cached_field(self, field_name):
        """Get a field from cache (base or derived)"""
        if field_name in self._all_base_fields:
            return self._all_base_fields[field_name]
        elif field_name in self._all_derived_fields:
            return self._all_derived_fields[field_name]
        return None
    
    def compute_all_derived_fields(self):
        """Compute all derived fields using cached base fields"""
        print("\n🧮 Computing all derived fields...")
        start_time = time.time()
        
        # Get all derived field configurations
        all_fields = self.registry.load_all_fields()
        derived_fields = {k: v for k, v in all_fields.items() if v.get('derived')}
        
        print(f"📊 Found {len(derived_fields)} derived fields to compute")
        
        # Sort by dependency order (simple approach - may need refinement)
        computed = set()
        iterations = 0
        max_iterations = 10
        
        while len(computed) < len(derived_fields) and iterations < max_iterations:
            iterations += 1
            made_progress = False
            
            for field_name, field_config in derived_fields.items():
                if field_name in computed:
                    continue
                
                # Check if all inputs are available
                inputs = field_config.get('inputs', [])
                if all(self.get_cached_field(inp) is not None for inp in inputs):
                    try:
                        print(f"  Computing {field_name}...", end='', flush=True)
                        
                        # Special handling for composite fields
                        if field_config.get('plot_style') in ['lines', 'composite', 'lines_with_barbs'] and \
                           field_config.get('function') == 'identity':
                            # Load composite data
                            data = self._load_composite_from_cache(field_name, field_config)
                        else:
                            # Regular derived parameter
                            input_data = {}
                            for inp in inputs:
                                cached = self.get_cached_field(inp)
                                if cached is not None:
                                    input_data[inp] = cached.values
                            
                            # Compute the derived parameter
                            result_array = compute_derived_parameter(field_name, input_data, field_config)
                            
                            if result_array is not None:
                                # Create xarray with coordinates from first input
                                ref_field = self.get_cached_field(inputs[0])
                                import xarray as xr
                                data = xr.DataArray(
                                    result_array,
                                    coords=ref_field.coords,
                                    dims=ref_field.dims,
                                    name=field_name
                                )
                            else:
                                data = None
                        
                        if data is not None:
                            self._all_derived_fields[field_name] = data
                            computed.add(field_name)
                            made_progress = True
                            print(" ✓")
                        else:
                            print(" ✗")
                    except Exception as e:
                        print(f" ✗ ({str(e)})")
            
            if not made_progress and len(computed) < len(derived_fields):
                print(f"\n⚠️ Could not compute all derived fields. Missing dependencies?")
                missing = set(derived_fields.keys()) - computed
                print(f"Missing: {missing}")
                break
        
        compute_time = time.time() - start_time
        print(f"\n✅ Computed {len(self._all_derived_fields)} derived fields in {compute_time:.1f}s")
        
        return self._all_derived_fields
    
    def _load_composite_from_cache(self, field_name, field_config):
        """Load composite data using cached fields"""
        try:
            input_fields = field_config.get('inputs', [])
            if not input_fields:
                return None
            
            # Get first input field as reference
            reference_data = None
            input_data = {}
            
            for input_field in input_fields:
                data = self.get_cached_field(input_field)
                if data is not None:
                    input_data[input_field] = data
                    if reference_data is None:
                        reference_data = data
            
            if not input_data or reference_data is None:
                return None
            
            # Create composite data object
            composite_data = reference_data.copy()
            composite_data.name = field_name
            composite_data.attrs.update({
                'composite_inputs': input_data,
                'plot_style': field_config.get('plot_style'),
                'plot_config': field_config.get('plot_config', {}),
                'long_name': field_config.get('title', field_name),
                'units': field_config.get('units', 'composite'),
                'derived': True
            })
            
            return composite_data
            
        except Exception as e:
            print(f"❌ Error creating composite {field_name}: {e}")
            return None
    
    def process_all_products_parallel(self, cycle, forecast_hour=0, output_dir=None, 
                                    categories=None):
        """Process all products with parallel map generation"""
        print(f"\n🚀 PARALLEL MAP GENERATION")
        print(f"🔧 Using {self.num_workers} worker processes")
        start_time = time.time()
        
        if output_dir is None:
            output_dir = Path('./outputs')
        output_dir = Path(output_dir)
        
        # Set region to conus
        self.set_region('conus')
        
        # Get all fields to process
        all_fields = self.registry.get_all_fields()
        
        # Filter by categories if specified
        if categories:
            fields_to_process = []
            for category in categories:
                cat_fields = [(name, config) for name, config in all_fields.items() 
                             if config.get('category') == category]
                fields_to_process.extend(cat_fields)
        else:
            fields_to_process = list(all_fields.items())
        
        # Group by category for organized output
        categories_dict = {}
        for field_name, field_config in fields_to_process:
            category = field_config.get('category', 'uncategorized')
            if category not in categories_dict:
                categories_dict[category] = []
            categories_dict[category].append((field_name, field_config))
        
        # Prepare work items for parallel processing
        work_items = []
        temp_dir = tempfile.mkdtemp()
        
        # Parse cycle for create_spc_plot
        if '_' in cycle:
            # Format: YYYYMMDD_HHZ
            cycle_parts = cycle.split('_')
            cycle_date_str = cycle_parts[0]
            cycle_hour_str = cycle_parts[1].replace('Z', '').replace('z', '')
        else:
            # Format: YYYYMMDDHH
            cycle_date_str = cycle[:8]
            cycle_hour_str = cycle[8:10]
        plot_cycle = f"{cycle_date_str}{cycle_hour_str}"
        
        # Prepare processor configuration to pass to workers
        processor_config = {
            'colormaps': self.colormaps,
            'regions': self.regions,
            'current_region': self.current_region,
            'model_name': self.model_name
        }
        
        print(f"\n📊 Preparing {len(fields_to_process)} products for parallel processing...")
        
        for category, fields in categories_dict.items():
            # Create category output directory
            # If output_dir is already the F## directory, just add category
            if output_dir.name.startswith('F') and output_dir.name[1:].isdigit():
                cat_output_dir = output_dir / category
            else:
                cat_output_dir = output_dir / f'F{forecast_hour:02d}' / category
            cat_output_dir.mkdir(parents=True, exist_ok=True)
            
            for field_name, field_config in fields:
                # Get data from cache
                data = self.get_cached_field(field_name)
                
                if data is None:
                    print(f"  ⚠️ {field_name}: No data in cache, skipping")
                    continue
                
                # Serialize data to temporary file
                data_pickle_path = os.path.join(temp_dir, f"{field_name}.pkl")
                with open(data_pickle_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Prepare output path
                output_filename = f"{field_name}_f{forecast_hour:02d}_REFACTORED.png"
                output_path = cat_output_dir / output_filename
                
                # Add work item
                work_items.append((
                    field_name,
                    field_config,
                    data_pickle_path,
                    plot_cycle,
                    forecast_hour,
                    str(output_path),
                    processor_config
                ))
        
        if not work_items:
            print("❌ No products to process!")
            return []
        
        print(f"\n🎨 Generating {len(work_items)} map products in parallel...")
        
        # Process in parallel
        successful = 0
        failed = 0
        output_files = []
        
        with Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for better progress reporting
            results = pool.imap_unordered(generate_single_map, work_items)
            
            for i, result in enumerate(results, 1):
                if result['success']:
                    successful += 1
                    output_files.append(result['output_file'])
                    status = "✓"
                else:
                    failed += 1
                    status = "✗"
                
                # Progress update every 10 items or at end
                if i % 10 == 0 or i == len(work_items):
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (len(work_items) - i) / rate if rate > 0 else 0
                    print(f"  Progress: {i}/{len(work_items)} ({successful} ✓, {failed} ✗) "
                          f"| {elapsed:.1f}s elapsed, ~{eta:.1f}s remaining")
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        total_time = time.time() - start_time
        print(f"\n✅ PARALLEL PROCESSING COMPLETE")
        print(f"📊 Successful: {successful} products")
        print(f"⚠️  Failed: {failed} products")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📈 Average: {total_time/len(work_items):.1f}s per product")
        print(f"🚀 Speedup: ~{self.num_workers:.1f}x vs sequential")
        
        return output_files


def generate_single_map(args):
    """Worker function to generate a single map product"""
    try:
        field_name, field_config, data_pickle_path, plot_cycle, forecast_hour, output_path, processor_config = args
        
        # Load the data from pickle
        with open(data_pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Recreate a minimal processor instance with required attributes
        processor = OptimizedHRRRProcessor(model=processor_config.get('model_name', 'hrrr'))
        processor.colormaps = processor_config['colormaps']
        processor.regions = processor_config['regions']
        processor.current_region = processor_config['current_region']
        processor.model_name = processor_config.get('model_name', 'hrrr')
        
        # Generate the plot
        start_time = time.time()
        output_file = processor.create_spc_plot(
            data, field_name, field_config,
            plot_cycle, forecast_hour, Path(output_path).parent
        )
        
        duration = time.time() - start_time
        
        if output_file:
            return {
                'field_name': field_name,
                'success': True,
                'duration': duration,
                'output_file': str(output_file)
            }
        else:
            return {
                'field_name': field_name,
                'success': False,
                'duration': duration,
                'error': 'Failed to create plot'
            }
            
    except Exception as e:
        import traceback
        return {
            'field_name': field_name,
            'success': False,
            'duration': 0,
            'error': f"{str(e)}\n{traceback.format_exc()}"
        }


def process_hrrr_parallel(cycle: str, forecast_hour: int = 0, output_dir: Optional[Path] = None,
                         categories: Optional[List[str]] = None, model: str = 'hrrr'):
    """Main entry point for parallel processing"""
    print("="*60)
    print(f"🚀 {model.upper()} PARALLEL MAP PROCESSOR")
    print("="*60)
    
    processor = OptimizedHRRRProcessor(model=model)
    processor.set_region('conus')
    
    # Handle both cycle formats: YYYYMMDDHH and YYYYMMDD_HHZ
    if '_' in cycle:
        # Format: YYYYMMDD_HHZ
        cycle_date = datetime.strptime(cycle.split('_')[0], '%Y%m%d')
        cycle_hour = cycle.split('_')[1].replace('Z', '').replace('z', '')
    else:
        # Format: YYYYMMDDHH
        cycle_date = datetime.strptime(cycle[:8], '%Y%m%d')
        cycle_hour = cycle[8:10]
        cycle = f"{cycle[:8]}_{cycle_hour}Z"  # Convert to expected format
    
    # Set proper output directory if not specified
    if output_dir is None:
        output_dir = Path(f'outputs/{model}/{cycle_date.strftime("%Y%m%d")}/{cycle_hour}z')
    
    print(f"📅 Cycle: {cycle_date.strftime('%Y-%m-%d')} {cycle_hour}Z")
    print(f"📁 Output directory: {output_dir}")
    
    # Look for GRIB files in various locations
    # When called from smart_hrrr_processor, output_dir IS the F## directory with GRIB files
    if output_dir and output_dir.name.startswith('F'):
        # output_dir is already the forecast hour directory
        possible_dirs = [
            output_dir,  # This is the F## directory with GRIB files
        ]
    else:
        # Standard directory structure
        possible_dirs = [
            Path(f'outputs/{model}/{cycle_date.strftime("%Y%m%d")}/{cycle_hour}z/F{forecast_hour:02d}'),
            Path(f'outputs/{model}/{cycle_date.strftime("%Y%m%d")}/{cycle_hour}/F{forecast_hour:02d}'),
            Path(f'outputs/{model}/{cycle_date.strftime("%Y%m%d")}/{cycle_hour.lower()}/F{forecast_hour:02d}'),
            Path(f'outputs/hrrr/{cycle_date.strftime("%Y%m%d")}/{cycle_hour}z/F{forecast_hour:02d}'),  # fallback
            Path(f'grib_cache/{cycle}/F{forecast_hour:02d}'),
            Path('.')
        ]
    
    prs_file = None
    sfc_file = None
    
    print(f"🔍 Looking for GRIB files in:")
    for base_dir in possible_dirs:
        print(f"   - {base_dir} {'[✓ exists]' if base_dir.exists() else '[✗ not found]'}")
        if base_dir.exists():
            # Look for pressure file
            # Add GFS pattern support
            prs_patterns = ['*wrfprsf*.grib2', '*prs*.grib2', 'gfs.*.pgrb2.0p25.f*']
            for pattern in prs_patterns:
                files = list(base_dir.glob(pattern))
                if files:
                    prs_file = files[0]
                    break
            
            # Look for surface file
            # For GFS, surface data is in the same file as pressure
            sfc_patterns = ['*wrfsfcf*.grib2', '*sfc*.grib2', 'gfs.*.pgrb2.0p25.f*']
            for pattern in sfc_patterns:
                files = list(base_dir.glob(pattern))
                if files:
                    sfc_file = files[0]
                    break
            
            if prs_file or sfc_file:
                break
    
    if not prs_file and not sfc_file:
        print("❌ No GRIB files found!")
        return []
    
    print(f"📂 Found GRIB files:")
    if prs_file:
        print(f"  Pressure: {prs_file}")
    if sfc_file:
        print(f"  Surface: {sfc_file}")
    
    # Phase 1: Load all base fields (sequential - I/O bound)
    print("\n" + "="*60)
    print("PHASE 1: DATA LOADING (Sequential)")
    print("="*60)
    processor.load_all_base_fields(str(prs_file) if prs_file else None, 
                                   str(sfc_file) if sfc_file else None)
    
    # Phase 2: Compute all derived fields (sequential - depends on order)
    print("\n" + "="*60)
    print("PHASE 2: DERIVED FIELD COMPUTATION (Sequential)")
    print("="*60)
    processor.compute_all_derived_fields()
    
    # Phase 3: Generate all plots IN PARALLEL!
    print("\n" + "="*60)
    print("PHASE 3: MAP GENERATION (Parallel)")
    print("="*60)
    output_files = processor.process_all_products_parallel(
        cycle=cycle,
        forecast_hour=forecast_hour,
        output_dir=output_dir,
        categories=categories
    )
    
    return output_files