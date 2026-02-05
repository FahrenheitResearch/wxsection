"""Interactive Cross-Section System with Pre-loaded Data.

Pre-loads all required 3D fields into memory for sub-second cross-section generation.
With 128GB RAM, can easily hold 18 forecast hours (~2GB each).

Features:
- Zarr caching: First load converts GRIB→Zarr (~25s), subsequent loads ~2s
- Parallel loading with multiprocessing
- Sub-second cross-section generation once loaded

Usage:
    from core.cross_section_interactive import InteractiveCrossSection

    ixs = InteractiveCrossSection(cache_dir="cache/zarr")  # Enable Zarr caching
    ixs.load_run("outputs/hrrr/20251224/19z", max_hours=18, workers=4)

    # Generate cross-sections instantly (~0.5s)
    img_bytes = ixs.get_cross_section(
        start_point=(39.74, -104.99),
        end_point=(41.88, -87.63),
        style="wind_speed",
        forecast_hour=0,
    )
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import warnings
import time
import io


@dataclass
class ForecastHourData:
    """Holds all pre-loaded data for a single forecast hour."""
    forecast_hour: int
    pressure_levels: np.ndarray  # (n_levels,) hPa
    lats: np.ndarray  # (ny, nx) or (ny,)
    lons: np.ndarray  # (ny, nx) or (nx,)

    # 3D fields: (n_levels, ny, nx)
    temperature: np.ndarray = None  # K
    u_wind: np.ndarray = None  # m/s
    v_wind: np.ndarray = None  # m/s
    rh: np.ndarray = None  # %
    omega: np.ndarray = None  # Pa/s
    specific_humidity: np.ndarray = None  # kg/kg
    geopotential_height: np.ndarray = None  # gpm
    vorticity: np.ndarray = None  # 1/s
    cloud: np.ndarray = None  # kg/kg
    dew_point: np.ndarray = None  # K

    # Smoke on native hybrid levels (NOT isobaric) — preserves boundary layer detail
    smoke_hyb: np.ndarray = None  # (n_hyb, ny, nx) μg/m³ PM2.5 on hybrid levels
    smoke_pres_hyb: np.ndarray = None  # (n_hyb, ny, nx) pressure in hPa at each hybrid level

    # Surface fields: (ny, nx)
    surface_pressure: np.ndarray = None  # hPa

    # Pre-computed derived fields
    theta: np.ndarray = None  # K
    temp_c: np.ndarray = None  # C

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        total = 0
        for name, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                total += val.nbytes
        return total / 1024 / 1024


# Standalone function for multiprocessing (must be at module level for pickle)
def _load_hour_process(grib_file: str, forecast_hour: int) -> Optional[ForecastHourData]:
    """Load a single forecast hour - standalone function for ProcessPoolExecutor."""
    import cfgrib

    try:
        print(f"Loading F{forecast_hour:02d} from {Path(grib_file).name}...")
        start = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Load temperature first to get grid info
            ds_t = cfgrib.open_dataset(
                grib_file,
                filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 't'},
                backend_kwargs={'indexpath': ''},
            )

            var_name = list(ds_t.data_vars)[0]
            t_data = ds_t[var_name]

            if 'isobaricInhPa' in t_data.dims:
                pressure_levels = t_data.isobaricInhPa.values
            else:
                pressure_levels = t_data.level.values

            if 'latitude' in t_data.coords:
                lats = t_data.latitude.values
                lons = t_data.longitude.values
            else:
                lats = t_data.lat.values
                lons = t_data.lon.values

            if lons.max() > 180:
                lons = np.where(lons > 180, lons - 360, lons)

            fhr_data = ForecastHourData(
                forecast_hour=forecast_hour,
                pressure_levels=pressure_levels,
                lats=lats,
                lons=lons,
                temperature=t_data.values,
            )
            ds_t.close()

            # Load other fields
            fields = {
                'u': 'u_wind', 'v': 'v_wind', 'r': 'rh', 'w': 'omega',
                'q': 'specific_humidity', 'gh': 'geopotential_height',
                'absv': 'vorticity', 'clwmr': 'cloud', 'dpt': 'dew_point',
            }

            for grib_key, field_name in fields.items():
                try:
                    ds = cfgrib.open_dataset(
                        grib_file,
                        filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': grib_key},
                        backend_kwargs={'indexpath': ''},
                    )
                    if ds and len(ds.data_vars) > 0:
                        setattr(fhr_data, field_name, ds[list(ds.data_vars)[0]].values)
                    ds.close()
                except Exception:
                    pass

            # Surface pressure
            try:
                sfc_file = Path(grib_file).parent / Path(grib_file).name.replace('wrfprs', 'wrfsfc')
                sp_file = str(sfc_file) if sfc_file.exists() else grib_file
                ds_sp = cfgrib.open_dataset(
                    sp_file,
                    filter_by_keys={'typeOfLevel': 'surface', 'shortName': 'sp'},
                    backend_kwargs={'indexpath': ''},
                )
                if ds_sp and len(ds_sp.data_vars) > 0:
                    sp_data = ds_sp[list(ds_sp.data_vars)[0]].values
                    while sp_data.ndim > 2:
                        sp_data = sp_data[0]
                    if sp_data.max() > 2000:
                        sp_data = sp_data / 100.0
                    fhr_data.surface_pressure = sp_data
                ds_sp.close()
            except Exception:
                pass

            # Load smoke (MASSDEN) from wrfnat if available — keep on native hybrid levels
            try:
                nat_file = Path(grib_file).parent / Path(grib_file).name.replace('wrfprs', 'wrfnat')
                if nat_file.exists():
                    import eccodes
                    smoke_levels = {}
                    pres_levels_hyb = {}
                    fnat = open(str(nat_file), 'rb')
                    while True:
                        msg = eccodes.codes_grib_new_from_file(fnat)
                        if msg is None:
                            break
                        try:
                            disc = eccodes.codes_get(msg, 'discipline')
                            cat = eccodes.codes_get(msg, 'parameterCategory')
                            num = eccodes.codes_get(msg, 'parameterNumber')
                            lt = eccodes.codes_get(msg, 'typeOfLevel')
                            lev = eccodes.codes_get(msg, 'level')
                            if lt == 'hybrid':
                                Ni = eccodes.codes_get(msg, 'Ni')
                                Nj = eccodes.codes_get(msg, 'Nj')
                                if disc == 0 and cat == 20 and num == 0:
                                    smoke_levels[lev] = eccodes.codes_get_values(msg).reshape(Nj, Ni)
                                elif disc == 0 and cat == 3 and num == 0 and lev not in pres_levels_hyb:
                                    pres_levels_hyb[lev] = eccodes.codes_get_values(msg).reshape(Nj, Ni)
                        except Exception:
                            pass
                        eccodes.codes_release(msg)
                    fnat.close()

                    if smoke_levels and pres_levels_hyb:
                        levels_sorted = sorted(smoke_levels.keys())
                        ny, nx = list(smoke_levels.values())[0].shape
                        n_hyb = len(levels_sorted)
                        smoke_hyb = np.zeros((n_hyb, ny, nx), dtype=np.float32)
                        pres_hyb = np.zeros((n_hyb, ny, nx), dtype=np.float32)
                        for idx, lv in enumerate(levels_sorted):
                            smoke_hyb[idx] = smoke_levels[lv] * 1e9
                            if lv in pres_levels_hyb:
                                pres_hyb[idx] = pres_levels_hyb[lv] / 100.0
                        fhr_data.smoke_hyb = smoke_hyb
                        fhr_data.smoke_pres_hyb = pres_hyb
            except Exception:
                pass

            # Pre-compute theta
            if fhr_data.temperature is not None:
                theta = np.zeros_like(fhr_data.temperature)
                for lev_idx, p in enumerate(pressure_levels):
                    theta[lev_idx] = fhr_data.temperature[lev_idx] * (1000.0 / p) ** 0.286
                fhr_data.theta = theta
                fhr_data.temp_c = fhr_data.temperature - 273.15

            duration = time.time() - start
            print(f"  Loaded F{forecast_hour:02d} in {duration:.1f}s ({fhr_data.memory_usage_mb():.0f} MB)")
            return fhr_data

    except Exception as e:
        print(f"Error loading F{forecast_hour:02d}: {e}")
        return None


class InteractiveCrossSection:
    """Pre-loads HRRR data for fast interactive cross-section generation."""

    # Fields to pre-load (covers all 13 styles)
    FIELDS_TO_LOAD = {
        't': 'temperature',
        'u': 'u_wind',
        'v': 'v_wind',
        'r': 'rh',
        'w': 'omega',
        'q': 'specific_humidity',
        'gh': 'geopotential_height',
        'absv': 'vorticity',
        'clwmr': 'cloud',
        'dpt': 'dew_point',
    }

    CACHE_LIMIT_GB = 400  # Max NPZ cache size on disk

    def __init__(self, cache_dir: str = None):
        """Initialize the interactive cross-section system.

        Args:
            cache_dir: Directory for NPZ cache. If provided, enables fast caching.
                      First load from GRIB takes ~25s, subsequent loads ~2s.
        """
        self.forecast_hours: Dict[int, ForecastHourData] = {}
        self._interpolator_cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Metadata for labeling
        self.model = "HRRR"
        self.init_date = None  # YYYYMMDD
        self.init_hour = None  # HH

    def _get_cache_path(self, grib_file: str) -> Optional[Path]:
        """Get cache path for a GRIB file."""
        if not self.cache_dir:
            return None
        # Create unique cache name based on GRIB path
        grib_path = Path(grib_file)
        # e.g., outputs/hrrr/20251224/19z/F00/hrrr.t19z.wrfprsf00.grib2
        # -> 20251224_19z_F00_hrrr.t19z.wrfprsf00.npz
        parts = grib_path.parts
        try:
            date_idx = next(i for i, p in enumerate(parts) if p.isdigit() and len(p) == 8)
            cache_name = f"{parts[date_idx]}_{parts[date_idx+1]}_{parts[date_idx+2]}_{grib_path.stem}.npz"
        except (StopIteration, IndexError):
            cache_name = f"{grib_path.stem}.npz"
        return self.cache_dir / cache_name

    def _cleanup_cache(self):
        """Evict oldest NPZ cache files if cache exceeds CACHE_LIMIT_GB."""
        if not self.cache_dir:
            return
        try:
            files = list(self.cache_dir.glob('*.npz'))
            total_bytes = sum(f.stat().st_size for f in files)
            total_gb = total_bytes / (1024 ** 3)
            if total_gb <= self.CACHE_LIMIT_GB:
                return
            # Sort by access time (oldest first) and delete until under 85% of limit
            target_gb = self.CACHE_LIMIT_GB * 0.85
            files.sort(key=lambda f: f.stat().st_atime)
            for f in files:
                if total_gb <= target_gb:
                    break
                size_gb = f.stat().st_size / (1024 ** 3)
                f.unlink()
                total_gb -= size_gb
                print(f"Cache cleanup: removed {f.name} ({size_gb:.1f}GB), {total_gb:.1f}GB remaining")
        except Exception as e:
            print(f"Cache cleanup error: {e}")

    def _save_to_cache(self, fhr_data: ForecastHourData, cache_path: Path):
        """Save ForecastHourData to numpy format (uncompressed for speed)."""
        data = {'forecast_hour': np.array([fhr_data.forecast_hour])}

        for field in ['pressure_levels', 'lats', 'lons', 'temperature', 'u_wind', 'v_wind',
                      'rh', 'omega', 'specific_humidity', 'geopotential_height', 'vorticity',
                      'cloud', 'dew_point', 'smoke_hyb', 'smoke_pres_hyb',
                      'surface_pressure', 'theta', 'temp_c']:
            arr = getattr(fhr_data, field, None)
            if arr is not None:
                data[field] = arr

        # Use uncompressed for fast save (~3.5GB per file, but saves in ~5s vs 60s)
        np.savez(cache_path, **data)
        self._cleanup_cache()

    def _load_from_cache(self, cache_path: Path) -> Optional[ForecastHourData]:
        """Load ForecastHourData from compressed numpy format."""
        try:
            data = np.load(cache_path)

            fhr_data = ForecastHourData(
                forecast_hour=int(data['forecast_hour'][0]),
                pressure_levels=data['pressure_levels'],
                lats=data['lats'],
                lons=data['lons'],
            )

            # Load optional arrays
            for field in ['temperature', 'u_wind', 'v_wind', 'rh', 'omega',
                          'specific_humidity', 'geopotential_height', 'vorticity',
                          'cloud', 'dew_point', 'smoke_hyb', 'smoke_pres_hyb',
                          'surface_pressure', 'theta', 'temp_c']:
                if field in data:
                    setattr(fhr_data, field, data[field])

            return fhr_data
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return None

    def _load_smoke_from_wrfnat(self, nat_file: str) -> Optional[tuple]:
        """Load MASSDEN (smoke PM2.5) from wrfnat GRIB using eccodes.

        MASSDEN is GRIB2 discipline=0, category=20, number=0 on hybrid levels.
        cfgrib can't identify this field (shows as 'unknown'), so we use eccodes directly.

        Returns (smoke_hyb, pres_hyb) on native hybrid levels — no interpolation
        to isobaric. This preserves the fine vertical resolution in the boundary
        layer where smoke concentrates (~10-15 levels in lowest 2 km).

        smoke_hyb: (n_hyb, ny, nx) in μg/m³
        pres_hyb:  (n_hyb, ny, nx) in hPa (varies per column due to terrain)
        """
        import eccodes

        # Read MASSDEN and pressure on hybrid levels
        smoke_levels = {}  # level -> 2D array (kg/m³)
        pres_levels = {}   # level -> 2D array (Pa)

        f = open(nat_file, 'rb')
        try:
            while True:
                msg = eccodes.codes_grib_new_from_file(f)
                if msg is None:
                    break
                try:
                    disc = eccodes.codes_get(msg, 'discipline')
                    cat = eccodes.codes_get(msg, 'parameterCategory')
                    num = eccodes.codes_get(msg, 'parameterNumber')
                    ltype = eccodes.codes_get(msg, 'typeOfLevel')
                    level = eccodes.codes_get(msg, 'level')

                    if ltype == 'hybrid':
                        Ni = eccodes.codes_get(msg, 'Ni')
                        Nj = eccodes.codes_get(msg, 'Nj')
                        if disc == 0 and cat == 20 and num == 0:
                            # MASSDEN - smoke mass density
                            smoke_levels[level] = eccodes.codes_get_values(msg).reshape(Nj, Ni)
                        elif disc == 0 and cat == 3 and num == 0 and level not in pres_levels:
                            # Pressure on hybrid levels (shortName='pres')
                            pres_levels[level] = eccodes.codes_get_values(msg).reshape(Nj, Ni)
                except Exception:
                    pass
                eccodes.codes_release(msg)
        finally:
            f.close()

        if not smoke_levels or not pres_levels:
            return None

        # Build sorted 3D arrays (sorted by hybrid level number)
        levels_sorted = sorted(smoke_levels.keys())
        ny, nx = list(smoke_levels.values())[0].shape
        n_hyb = len(levels_sorted)

        smoke_hyb = np.zeros((n_hyb, ny, nx), dtype=np.float32)
        pres_hyb = np.zeros((n_hyb, ny, nx), dtype=np.float32)

        for idx, lev in enumerate(levels_sorted):
            smoke_hyb[idx] = smoke_levels[lev] * 1e9  # kg/m³ → μg/m³
            if lev in pres_levels:
                pres_hyb[idx] = pres_levels[lev] / 100.0  # Pa → hPa

        return smoke_hyb, pres_hyb

    def load_forecast_hour(self, grib_file: str, forecast_hour: int, progress_callback=None) -> bool:
        """Load all fields for a forecast hour into memory.

        Args:
            grib_file: Path to wrfprs GRIB2 file
            forecast_hour: Forecast hour number
            progress_callback: Optional callback(step, total, detail) for progress reporting

        Returns:
            True if successful
        """
        cb = progress_callback or (lambda s, t, d: None)

        # Check cache first
        cache_path = self._get_cache_path(grib_file)
        if cache_path and cache_path.exists():
            cb(1, 2, "Loading from cache...")
            print(f"Loading F{forecast_hour:02d} from cache...")
            start = time.time()
            fhr_data = self._load_from_cache(cache_path)
            if fhr_data is not None:
                # Backfill smoke if missing from cache but wrfnat now available
                if fhr_data.smoke_hyb is None:
                    nat_file = Path(grib_file).parent / Path(grib_file).name.replace('wrfprs', 'wrfnat')
                    if nat_file.exists():
                        print(f"  Backfilling smoke from wrfnat...")
                        try:
                            result = self._load_smoke_from_wrfnat(str(nat_file))
                            if result is not None:
                                fhr_data.smoke_hyb, fhr_data.smoke_pres_hyb = result
                                print(f"  Loaded PM2.5 smoke on {fhr_data.smoke_hyb.shape[0]} hybrid levels "
                                      f"(max={np.nanmax(fhr_data.smoke_hyb):.1f} μg/m³)")
                                # Update cache with smoke included
                                try:
                                    self._save_to_cache(fhr_data, cache_path)
                                    print(f"  Updated cache with smoke data")
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"  Warning: Could not backfill smoke: {e}")
                self.forecast_hours[forecast_hour] = fhr_data
                duration = time.time() - start
                print(f"  Loaded F{forecast_hour:02d} from cache in {duration:.1f}s ({fhr_data.memory_usage_mb():.0f} MB)")
                cb(2, 2, "Done")
                return True

        # Field names for progress reporting
        field_labels = {
            't': 'Temperature', 'u': 'U-Wind', 'v': 'V-Wind', 'r': 'RH',
            'w': 'Omega', 'q': 'Sp. Humidity', 'gh': 'Geopotential',
            'absv': 'Vorticity', 'clwmr': 'Cloud Water', 'dpt': 'Dew Point',
        }
        total_steps = 13  # 10 fields + surface pressure + smoke + derived

        try:
            import cfgrib
            from scipy.spatial import cKDTree

            print(f"Loading F{forecast_hour:02d} from {Path(grib_file).name}...")
            start = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # First, get grid info and pressure levels from temperature
                cb(1, total_steps, "Reading Temperature...")
                ds_t = cfgrib.open_dataset(
                    grib_file,
                    filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 't'},
                    backend_kwargs={'indexpath': ''},
                )

                var_name = list(ds_t.data_vars)[0]
                t_data = ds_t[var_name]

                # Get pressure levels
                if 'isobaricInhPa' in t_data.dims:
                    pressure_levels = t_data.isobaricInhPa.values
                else:
                    pressure_levels = t_data.level.values

                # Get lat/lon grid
                if 'latitude' in t_data.coords:
                    lats = t_data.latitude.values
                    lons = t_data.longitude.values
                else:
                    lats = t_data.lat.values
                    lons = t_data.lon.values

                # Convert lons to -180 to 180
                if lons.max() > 180:
                    lons = np.where(lons > 180, lons - 360, lons)

                # Create data holder
                fhr_data = ForecastHourData(
                    forecast_hour=forecast_hour,
                    pressure_levels=pressure_levels,
                    lats=lats,
                    lons=lons,
                    temperature=t_data.values,
                )
                ds_t.close()

                # Load all other fields
                step = 2
                for grib_key, field_name in self.FIELDS_TO_LOAD.items():
                    if grib_key == 't':
                        continue  # Already loaded

                    cb(step, total_steps, f"Reading {field_labels.get(grib_key, field_name)}...")
                    try:
                        ds = cfgrib.open_dataset(
                            grib_file,
                            filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': grib_key},
                            backend_kwargs={'indexpath': ''},
                        )
                        if ds and len(ds.data_vars) > 0:
                            var = list(ds.data_vars)[0]
                            setattr(fhr_data, field_name, ds[var].values)
                        ds.close()
                    except Exception as e:
                        print(f"  Warning: Could not load {grib_key}: {e}")
                    step += 1

                # Load surface pressure
                cb(11, total_steps, "Reading Surface Pressure...")
                try:
                    # Try wrfsfc file first
                    sfc_file = Path(grib_file).parent / Path(grib_file).name.replace('wrfprs', 'wrfsfc')
                    sp_file = str(sfc_file) if sfc_file.exists() else grib_file

                    ds_sp = cfgrib.open_dataset(
                        sp_file,
                        filter_by_keys={'typeOfLevel': 'surface', 'shortName': 'sp'},
                        backend_kwargs={'indexpath': ''},
                    )
                    if ds_sp and len(ds_sp.data_vars) > 0:
                        sp_var = list(ds_sp.data_vars)[0]
                        sp_data = ds_sp[sp_var].values
                        while sp_data.ndim > 2:
                            sp_data = sp_data[0]
                        # Convert Pa to hPa
                        if sp_data.max() > 2000:
                            sp_data = sp_data / 100.0
                        fhr_data.surface_pressure = sp_data
                    ds_sp.close()
                except Exception as e:
                    print(f"  Warning: Could not load surface pressure: {e}")

                # Load smoke (MASSDEN PM2.5) from wrfnat file if available
                cb(12, total_steps, "Reading Smoke (PM2.5)...")
                try:
                    nat_file = Path(grib_file).parent / Path(grib_file).name.replace('wrfprs', 'wrfnat')
                    if nat_file.exists():
                        result = self._load_smoke_from_wrfnat(str(nat_file))
                        if result is not None:
                            fhr_data.smoke_hyb, fhr_data.smoke_pres_hyb = result
                            print(f"  Loaded PM2.5 smoke on {fhr_data.smoke_hyb.shape[0]} hybrid levels "
                                  f"(max={np.nanmax(fhr_data.smoke_hyb):.1f} μg/m³)")
                except Exception as e:
                    print(f"  Warning: Could not load smoke from wrfnat: {e}")

                # Pre-compute theta and temp_c
                cb(13, total_steps, "Computing derived fields...")
                if fhr_data.temperature is not None:
                    P_ref = 1000.0
                    kappa = 0.286
                    theta = np.zeros_like(fhr_data.temperature)
                    for lev_idx, p in enumerate(pressure_levels):
                        theta[lev_idx] = fhr_data.temperature[lev_idx] * (P_ref / p) ** kappa
                    fhr_data.theta = theta
                    fhr_data.temp_c = fhr_data.temperature - 273.15

                # Store
                self.forecast_hours[forecast_hour] = fhr_data

                duration = time.time() - start
                mem_mb = fhr_data.memory_usage_mb()
                print(f"  Loaded F{forecast_hour:02d} in {duration:.1f}s ({mem_mb:.0f} MB)")

                # Save to cache for fast subsequent loads
                if cache_path:
                    try:
                        self._save_to_cache(fhr_data, cache_path)
                        print(f"  Cached to {cache_path.name}")
                    except Exception as e:
                        print(f"  Warning: Could not cache: {e}")

                return True

        except Exception as e:
            print(f"Error loading F{forecast_hour:02d}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_run(self, run_dir: str, max_hours: int = 18, workers: int = 1) -> int:
        """Load all forecast hours from a run directory.

        Args:
            run_dir: Path to run directory (e.g., outputs/hrrr/20251224/19z)
            max_hours: Maximum forecast hours to load
            workers: Number of parallel workers (1 = sequential)

        Returns:
            Number of hours loaded
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            print(f"Run directory not found: {run_dir}")
            return 0

        # Extract init date/hour from path (e.g., outputs/hrrr/20251224/19z)
        import re
        path_str = str(run_path)
        date_match = re.search(r'/(\d{8})/(\d{2})z', path_str)
        if date_match:
            self.init_date = date_match.group(1)
            self.init_hour = date_match.group(2)
        else:
            # Try to get from directory names
            parts = run_path.parts
            for i, part in enumerate(parts):
                if len(part) == 8 and part.isdigit():
                    self.init_date = part
                    if i + 1 < len(parts) and parts[i + 1].endswith('z'):
                        self.init_hour = parts[i + 1].replace('z', '')
                    break

        # Collect files to load
        files_to_load = []
        for fhr in range(max_hours + 1):
            fhr_dir = run_path / f"F{fhr:02d}"
            prs_files = list(fhr_dir.glob("*wrfprs*.grib2"))
            if prs_files:
                files_to_load.append((str(prs_files[0]), fhr))

        if not files_to_load:
            print("No GRIB files found")
            return 0

        print(f"Loading {len(files_to_load)} forecast hours with {workers} workers...")
        start_time = time.time()

        if workers <= 1:
            # Sequential loading
            for grib_file, fhr in files_to_load:
                self.load_forecast_hour(grib_file, fhr)
        else:
            # Parallel loading with multiprocessing (bypasses GIL)
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_load_hour_process, grib_file, fhr): fhr
                    for grib_file, fhr in files_to_load
                }

                for future in as_completed(futures):
                    fhr = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            self.forecast_hours[result.forecast_hour] = result
                    except Exception as e:
                        print(f"Error loading F{fhr:02d}: {e}")

        duration = time.time() - start_time
        loaded = len(self.forecast_hours)
        total_mem = sum(fh.memory_usage_mb() for fh in self.forecast_hours.values())
        print(f"\nLoaded {loaded} forecast hours ({total_mem:.0f} MB total) in {duration:.1f}s")
        print(f"  ({duration/max(1,loaded):.1f}s per hour with {workers} workers)")
        return loaded

    def _load_hour_worker(self, grib_file: str, forecast_hour: int) -> Optional[ForecastHourData]:
        """Worker function for parallel loading. Returns ForecastHourData or None."""
        try:
            import cfgrib

            print(f"Loading F{forecast_hour:02d} from {Path(grib_file).name}...")
            start = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # First, get grid info and pressure levels from temperature
                ds_t = cfgrib.open_dataset(
                    grib_file,
                    filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 't'},
                    backend_kwargs={'indexpath': ''},
                )

                var_name = list(ds_t.data_vars)[0]
                t_data = ds_t[var_name]

                # Get pressure levels
                if 'isobaricInhPa' in t_data.dims:
                    pressure_levels = t_data.isobaricInhPa.values
                else:
                    pressure_levels = t_data.level.values

                # Get lat/lon grid
                if 'latitude' in t_data.coords:
                    lats = t_data.latitude.values
                    lons = t_data.longitude.values
                else:
                    lats = t_data.lat.values
                    lons = t_data.lon.values

                # Convert lons to -180 to 180
                if lons.max() > 180:
                    lons = np.where(lons > 180, lons - 360, lons)

                # Create data holder
                fhr_data = ForecastHourData(
                    forecast_hour=forecast_hour,
                    pressure_levels=pressure_levels,
                    lats=lats,
                    lons=lons,
                    temperature=t_data.values,
                )
                ds_t.close()

                # Load all other fields
                for grib_key, field_name in self.FIELDS_TO_LOAD.items():
                    if grib_key == 't':
                        continue  # Already loaded

                    try:
                        ds = cfgrib.open_dataset(
                            grib_file,
                            filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': grib_key},
                            backend_kwargs={'indexpath': ''},
                        )
                        if ds and len(ds.data_vars) > 0:
                            var = list(ds.data_vars)[0]
                            setattr(fhr_data, field_name, ds[var].values)
                        ds.close()
                    except Exception:
                        pass  # Silently skip unavailable fields in parallel mode

                # Load surface pressure
                try:
                    sfc_file = Path(grib_file).parent / Path(grib_file).name.replace('wrfprs', 'wrfsfc')
                    sp_file = str(sfc_file) if sfc_file.exists() else grib_file

                    ds_sp = cfgrib.open_dataset(
                        sp_file,
                        filter_by_keys={'typeOfLevel': 'surface', 'shortName': 'sp'},
                        backend_kwargs={'indexpath': ''},
                    )
                    if ds_sp and len(ds_sp.data_vars) > 0:
                        sp_var = list(ds_sp.data_vars)[0]
                        sp_data = ds_sp[sp_var].values
                        while sp_data.ndim > 2:
                            sp_data = sp_data[0]
                        if sp_data.max() > 2000:
                            sp_data = sp_data / 100.0
                        fhr_data.surface_pressure = sp_data
                    ds_sp.close()
                except Exception:
                    pass

                # Pre-compute theta and temp_c
                if fhr_data.temperature is not None:
                    P_ref = 1000.0
                    kappa = 0.286
                    theta = np.zeros_like(fhr_data.temperature)
                    for lev_idx, p in enumerate(pressure_levels):
                        theta[lev_idx] = fhr_data.temperature[lev_idx] * (P_ref / p) ** kappa
                    fhr_data.theta = theta
                    fhr_data.temp_c = fhr_data.temperature - 273.15

                duration = time.time() - start
                mem_mb = fhr_data.memory_usage_mb()
                print(f"  Loaded F{forecast_hour:02d} in {duration:.1f}s ({mem_mb:.0f} MB)")

                return fhr_data

        except Exception as e:
            print(f"Error loading F{forecast_hour:02d}: {e}")
            return None

    def get_cross_section(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        style: str = "wind_speed",
        forecast_hour: int = 0,
        n_points: int = 100,
        return_image: bool = True,
        dpi: int = 100,
        y_axis: str = "pressure",
        vscale: float = 1.0,
        y_top: int = 100,
        units: str = "km",
        terrain_data: Dict = None,
        temp_cmap: str = "green_purple",
    ) -> Optional[bytes]:
        """Generate cross-section from pre-loaded data.

        This is the fast path - should complete in <1 second.

        Args:
            start_point: (lat, lon) start
            end_point: (lat, lon) end
            style: Cross-section style
            forecast_hour: Which forecast hour to use
            n_points: Points along path
            return_image: If True, return PNG bytes; if False, return data dict
            dpi: Output resolution
            y_axis: 'pressure' (hPa) or 'height' (km)
            vscale: Vertical exaggeration factor (1.0 = normal, 2.0 = 2x taller)
            y_top: Top of plot in hPa (100=full atmos, 300=mid, 500=low, 700=boundary layer)
            units: 'km' or 'mi' for distance axis
            terrain_data: Optional dict with 'surface_pressure', 'surface_pressure_hires',
                         'distances_hires' keys to override terrain (for consistent GIF frames)
            temp_cmap: Temperature colormap choice ('green_purple', 'white_zero', 'nws_ndfd')

        Returns:
            PNG image bytes, or data dict if return_image=False
        """
        if forecast_hour not in self.forecast_hours:
            print(f"Forecast hour {forecast_hour} not loaded")
            return None

        start = time.time()

        # Get pre-loaded data
        fhr_data = self.forecast_hours[forecast_hour]

        # Create path
        path_lats = np.linspace(start_point[0], end_point[0], n_points)
        path_lons = np.linspace(start_point[1], end_point[1], n_points)

        # Interpolate all needed fields to path
        data = self._interpolate_to_path(fhr_data, path_lats, path_lons, style)

        t_interp = time.time() - start

        if not return_image:
            return data

        # Override terrain for consistent GIF frames
        if terrain_data is not None:
            for key in ('surface_pressure', 'surface_pressure_hires', 'distances_hires'):
                if key in terrain_data:
                    data[key] = terrain_data[key]

        # Build metadata for labels
        # Use _real_forecast_hour if set by dashboard (engine_key != real fhr)
        real_fhr = getattr(self, '_real_forecast_hour', fhr_data.forecast_hour)
        metadata = {
            'model': self.model,
            'init_date': self.init_date,
            'init_hour': self.init_hour,
            'forecast_hour': real_fhr,
        }
        self._real_forecast_hour = None  # Reset after use

        # Render
        img_bytes = self._render_cross_section(data, style, dpi, metadata, y_axis, vscale, y_top, units=units, temp_cmap=temp_cmap)

        t_total = time.time() - start
        print(f"Cross-section generated in {t_total:.3f}s (interp: {t_interp:.3f}s)")

        return img_bytes

    def _interpolate_to_path(
        self,
        fhr_data: ForecastHourData,
        path_lats: np.ndarray,
        path_lons: np.ndarray,
        style: str,
    ) -> Dict[str, Any]:
        """Interpolate 3D fields to cross-section path."""
        from scipy.spatial import cKDTree
        from scipy.interpolate import RegularGridInterpolator

        n_points = len(path_lats)
        n_levels = len(fhr_data.pressure_levels)

        lats_grid = fhr_data.lats
        lons_grid = fhr_data.lons

        # Build interpolator (curvilinear vs regular grid)
        if lats_grid.ndim == 2:
            # Curvilinear grid - use KDTree
            src_pts = np.column_stack([lats_grid.ravel(), lons_grid.ravel()])
            tree = cKDTree(src_pts)
            tgt_pts = np.column_stack([path_lats, path_lons])
            _, indices = tree.query(tgt_pts, k=1)

            def interp_3d(field_3d):
                result = np.full((n_levels, n_points), np.nan)
                for lev in range(min(field_3d.shape[0], n_levels)):
                    result[lev, :] = field_3d[lev].ravel()[indices]
                return result

            def interp_2d(field_2d):
                return field_2d.ravel()[indices]
        else:
            # Regular grid - use bilinear interpolation
            lats_1d = lats_grid if lats_grid.ndim == 1 else lats_grid[:, 0]
            lons_1d = lons_grid if lons_grid.ndim == 1 else lons_grid[0, :]
            pts = np.column_stack([path_lats, path_lons])

            def interp_3d(field_3d):
                result = np.full((n_levels, n_points), np.nan)
                for lev in range(min(field_3d.shape[0], n_levels)):
                    interp = RegularGridInterpolator(
                        (lats_1d, lons_1d), field_3d[lev],
                        method='linear', bounds_error=False, fill_value=np.nan
                    )
                    result[lev, :] = interp(pts)
                return result

            def interp_2d(field_2d):
                interp = RegularGridInterpolator(
                    (lats_1d, lons_1d), field_2d,
                    method='linear', bounds_error=False, fill_value=np.nan
                )
                return interp(pts)

        # Build result dict
        result = {
            'lats': path_lats,
            'lons': path_lons,
            'distances': self._calculate_distances(path_lats, path_lons),
            'pressure_levels': fhr_data.pressure_levels,
        }

        # Always interpolate base fields
        if fhr_data.temperature is not None:
            result['temperature'] = interp_3d(fhr_data.temperature)
            result['temp_c'] = result['temperature'] - 273.15

        if fhr_data.theta is not None:
            result['theta'] = interp_3d(fhr_data.theta)

        if fhr_data.u_wind is not None:
            result['u_wind'] = interp_3d(fhr_data.u_wind)

        if fhr_data.v_wind is not None:
            result['v_wind'] = interp_3d(fhr_data.v_wind)

        if fhr_data.surface_pressure is not None:
            result['surface_pressure'] = interp_2d(fhr_data.surface_pressure)

            # Extract terrain with enough points for smooth visualization
            # Use bilinear interpolation between HRRR's 3km grid points
            total_dist_km = result['distances'][-1]
            # ~1.5km spacing gives smoother terrain while still following HRRR data
            terrain_res = max(100, int(total_dist_km / 1.5))
            path_lats_hires = np.linspace(path_lats[0], path_lats[-1], terrain_res)
            path_lons_hires = np.linspace(path_lons[0], path_lons[-1], terrain_res)

            if lats_grid.ndim == 2:
                # Curvilinear - use same tree
                tgt_pts_hires = np.column_stack([path_lats_hires, path_lons_hires])
                _, indices_hires = tree.query(tgt_pts_hires, k=1)
                sp_hires = fhr_data.surface_pressure.ravel()[indices_hires]
            else:
                # Regular grid - bilinear interpolation
                pts_hires = np.column_stack([path_lats_hires, path_lons_hires])
                interp_sp = RegularGridInterpolator(
                    (lats_1d, lons_1d), fhr_data.surface_pressure,
                    method='linear', bounds_error=False, fill_value=np.nan
                )
                sp_hires = interp_sp(pts_hires)

            result['surface_pressure_hires'] = sp_hires
            result['distances_hires'] = self._calculate_distances(path_lats_hires, path_lons_hires)

        # Style-specific fields
        if style in ['rh', 'q'] and fhr_data.rh is not None:
            result['rh'] = interp_3d(fhr_data.rh)

        if style == 'smoke' and fhr_data.smoke_hyb is not None:
            # Interpolate smoke and its pressure coordinate along path on native hybrid levels
            # interp_3d works on any (n_levels, ny, nx) array — hybrid levels work the same way
            result['smoke_hyb'] = interp_3d(fhr_data.smoke_hyb)  # (n_hyb, n_points)
            result['smoke_pres_hyb'] = interp_3d(fhr_data.smoke_pres_hyb)  # (n_hyb, n_points)

        if style == 'omega' and fhr_data.omega is not None:
            result['omega'] = interp_3d(fhr_data.omega)

        if style == 'vorticity' and fhr_data.vorticity is not None:
            result['vorticity'] = interp_3d(fhr_data.vorticity)

        if style in ['cloud', 'cloud_total', 'icing'] and fhr_data.cloud is not None:
            result['cloud'] = interp_3d(fhr_data.cloud)

        if style == 'theta_e' and fhr_data.specific_humidity is not None:
            q = interp_3d(fhr_data.specific_humidity)
            result['specific_humidity'] = q
            # Compute theta_e
            T = result['temperature']
            theta = result['theta']
            Lv = 2.5e6
            cp = 1004.0
            theta_e = np.zeros_like(T)
            for lev in range(len(fhr_data.pressure_levels)):
                theta_e[lev, :] = theta[lev, :] * np.exp(Lv * q[lev, :] / (cp * T[lev, :]))
            result['theta_e'] = theta_e

        if style == 'q' and fhr_data.specific_humidity is not None:
            result['specific_humidity'] = interp_3d(fhr_data.specific_humidity)

        # Always extract geopotential_height for height-axis display option
        if fhr_data.geopotential_height is not None:
            gh = interp_3d(fhr_data.geopotential_height)
            result['geopotential_height'] = gh

            if style == 'shear':
                # Compute shear
                u = result.get('u_wind')
                v = result.get('v_wind')
                if u is not None and v is not None:
                    shear = np.full((n_levels, n_points), np.nan)
                    for lev in range(n_levels - 1):
                        dz = gh[lev, :] - gh[lev + 1, :]
                        dz = np.where(np.abs(dz) < 10, np.nan, dz)
                        du = u[lev, :] - u[lev + 1, :]
                        dv = v[lev, :] - v[lev + 1, :]
                        dwind = np.sqrt(du**2 + dv**2)
                        shear[lev, :] = (dwind / np.abs(dz)) * 1000
                    shear[-1, :] = shear[-2, :]
                    result['shear'] = shear

            if style == 'lapse_rate':
                T = result['temperature']
                lapse = np.full((n_levels, n_points), np.nan)
                for lev in range(n_levels - 1):
                    dz = (gh[lev, :] - gh[lev + 1, :]) / 1000.0
                    dz = np.where(np.abs(dz) < 0.01, np.nan, dz)
                    dT = T[lev, :] - T[lev + 1, :]
                    lapse[lev, :] = -dT / dz
                lapse[-1, :] = lapse[-2, :]
                result['lapse_rate'] = lapse

        if style == 'wetbulb' and fhr_data.rh is not None:
            T_c = result['temp_c']
            RH = interp_3d(fhr_data.rh)
            result['rh'] = RH
            Tw = (T_c * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
                  + np.arctan(T_c + RH)
                  - np.arctan(RH - 1.676331)
                  + 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH)
                  - 4.686035)
            result['wetbulb'] = Tw

        if style == 'icing' and fhr_data.cloud is not None:
            T_c = result['temp_c']
            cloud = result['cloud'] * 1000  # g/kg
            icing = np.where((T_c >= -20) & (T_c <= 0), cloud, 0)
            result['icing'] = icing

        if style == 'frontogenesis':
            # Petterssen Kinematic Frontogenesis (Winter Bander Mode)
            # Compute frontogenesis on the cross-section using NumPy
            # F = -|∇θ|^(-1) * deformation_term
            from scipy.ndimage import gaussian_filter

            theta = result.get('theta')
            u = result.get('u_wind')
            v = result.get('v_wind')

            if theta is not None and u is not None and v is not None:
                # Apply Gaussian smoothing to reduce noise from high-res (3km) data
                # Sigma=1.5 smooths over ~4-5 grid points, removing small-scale noise
                # while preserving synoptic-scale frontal features
                sigma_val = 1.5
                theta_smooth = gaussian_filter(theta, sigma=sigma_val)
                u_smooth = gaussian_filter(u, sigma=sigma_val)
                v_smooth = gaussian_filter(v, sigma=sigma_val)

                distances_m = result['distances'] * 1000  # km to m

                # Compute gradients along section (ds = along-section distance)
                dtheta_ds = np.gradient(theta_smooth, distances_m, axis=1)

                # Compute section azimuth for wind rotation
                dlat = path_lats[-1] - path_lats[0]
                dlon = path_lons[-1] - path_lons[0]
                azimuth = np.arctan2(dlon * np.cos(np.radians(np.mean(path_lats))), dlat)

                # Project winds to section-parallel component
                u_section = u_smooth * np.cos(azimuth) + v_smooth * np.sin(azimuth)
                du_section_ds = np.gradient(u_section, distances_m, axis=1)

                # Magnitude of theta gradient
                grad_theta_mag = np.abs(dtheta_ds)
                grad_theta_mag = np.where(grad_theta_mag < 1e-10, 1e-10, grad_theta_mag)  # Avoid div by zero

                # Frontogenesis: F = -|∇θ| * d(u_n)/ds where u_n is normal to theta gradient
                # For section: F ≈ -(dθ/ds)^2 / |dθ/ds| * du_section/ds
                # Simplified: F = -sign(dθ/ds) * |dθ/ds| * du_section/ds
                frontogenesis_raw = -dtheta_ds * du_section_ds / grad_theta_mag

                # Convert units: K/m * m/s / m = K/m/s
                # Scale to K/100km/3hr: multiply by 100000 (100km) * 10800 (3hr)
                # = 1.08e9, but values are very small, so use 1e11 scaling
                frontogenesis = frontogenesis_raw * 1.08e9

                # Light smoothing on output for cleaner visualization
                frontogenesis = gaussian_filter(frontogenesis, sigma=0.8)

                # Mask unrealistic values (cap at ±5 K/100km/3hr)
                frontogenesis = np.clip(frontogenesis, -5, 5)

                result['frontogenesis'] = frontogenesis

        return result

    def _calculate_distances(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Calculate cumulative distance along path in km."""
        R = 6371
        distances = [0]
        for i in range(1, len(lats)):
            lat1, lon1 = np.radians(lats[i-1]), np.radians(lons[i-1])
            lat2, lon2 = np.radians(lats[i]), np.radians(lons[i])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances.append(distances[-1] + R * c)
        return np.array(distances)

    @staticmethod
    def _build_temp_colormap(name: str = "green_purple"):
        """Build a temperature colormap by name.

        Options:
            green_purple: Green (0°C) -> yellow -> orange -> red -> purple (hot)
            white_zero:   White (0°C) with purples below and warm colors above
            nws_ndfd:     Classic NWS blue -> cyan -> yellow (0°C) -> orange -> red
        """
        import matplotlib.colors as mcolors

        def from_f_anchors(anchors_f):
            """Convert (°F, (R,G,B)) anchors to a LinearSegmentedColormap."""
            anchors_c = [((f - 32) * 5/9, (r/255, g/255, b/255))
                         for f, (r, g, b) in anchors_f]
            vals = [a[0] for a in anchors_c]
            cols = [a[1] for a in anchors_c]
            norms = [(v - vals[0]) / (vals[-1] - vals[0]) for v in vals]
            return mcolors.LinearSegmentedColormap.from_list(
                f'temp_{name}', list(zip(norms, cols)), N=512)

        if name == "white_zero":
            # White at 0°C/32°F, purples below freezing, warm colors above
            return from_f_anchors([
                (-80, (100,  50, 150)),  # Deep purple (extreme cold)
                (-60, (120,  60, 180)),  # Purple
                (-40, (140,  80, 200)),  # Medium purple
                (-20, (160, 110, 220)),  # Light purple
                (  0, (190, 150, 230)),  # Pale purple
                ( 15, (220, 200, 240)),  # Lavender
                ( 32, (255, 255, 255)),  # White (freezing)
                ( 45, (255, 240, 200)),  # Warm white
                ( 55, (255, 220, 150)),  # Pale yellow
                ( 65, (255, 200, 100)),  # Yellow-orange
                ( 75, (255, 170,  70)),  # Orange
                ( 85, (255, 130,  50)),  # Dark orange
                ( 95, (240,  80,  30)),  # Red-orange
                (105, (210,  40,  30)),  # Red
                (115, (160,  20,  60)),  # Red-purple
                (125, ( 90,  10, 140)),  # Deep purple (hot)
            ])
        elif name == "nws_ndfd":
            # Classic NWS: purple -> blue -> cyan -> yellow (0°C) -> orange -> red
            return from_f_anchors([
                (-80, ( 75,   0, 130)),  # Indigo
                (-60, (106,   0, 205)),  # Purple
                (-40, (  0,   0, 205)),  # Dark blue
                (-20, (  0,   0, 255)),  # Blue
                (  0, (  0, 191, 255)),  # Deep sky blue
                ( 15, (  0, 255, 255)),  # Cyan
                ( 32, (255, 255,   0)),  # Yellow (freezing)
                ( 50, (255, 215,   0)),  # Gold
                ( 65, (255, 165,   0)),  # Orange
                ( 80, (255,  69,   0)),  # Red-orange
                (100, (255,   0,   0)),  # Red
                (115, (180,   0,  60)),  # Red-purple
                (125, ( 90,  10, 140)),  # Deep purple
            ])
        else:
            # green_purple (default): blue-teal below freezing, green at 0°C, warm above
            return from_f_anchors([
                (-80, (220, 220, 255)),  # Pale blue-white (extreme cold)
                (-60, (180, 180, 255)),  # Light blue
                (-40, (140, 160, 240)),  # Blue
                (-20, (100, 140, 220)),  # Medium blue
                (  0, ( 60, 140, 160)),  # Blue-teal
                ( 10, ( 60, 160, 130)),  # Teal-green
                ( 20, ( 70, 180, 110)),  # Cool green
                ( 32, ( 80, 160,  80)),  # Green (freezing 32°F/0°C)
                ( 40, (140, 210, 140)),  # Light green
                ( 50, (255, 225, 140)),  # Yellow
                ( 60, (255, 200, 100)),  # Yellow-orange
                ( 70, (255, 170,  80)),  # Orange
                ( 80, (255, 140,  60)),  # Dark orange
                ( 90, (255, 100,  40)),  # Red-orange
                (100, (230,  60,  40)),  # Red
                (105, (200,  30,  30)),  # Deep red
                (110, (170,  20,  40)),  # Red-purple
                (115, (140,  20,  70)),  # Purple-red
                (120, (110,  20, 110)),  # Purple
                (125, ( 90,  10, 140)),  # Deep purple
            ])

    def _render_cross_section(self, data: Dict, style: str, dpi: int, metadata: Dict = None,
                               y_axis: str = "pressure", vscale: float = 1.0, y_top: int = 100,
                               units: str = "km", temp_cmap: str = "green_purple") -> bytes:
        """Render cross-section to PNG bytes.

        Args:
            data: Interpolated cross-section data
            style: Visualization style
            dpi: Output resolution
            metadata: Model run info for labels
            y_axis: 'pressure' (hPa) or 'height' (km)
            vscale: Vertical exaggeration (1.0 = normal)
            y_top: Top of plot in hPa (100, 300, 500, or 700)
            units: 'km' or 'mi' for distance axis
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.ticker import MultipleLocator
        from datetime import datetime, timedelta

        KM_TO_MI = 0.621371
        use_miles = (units == 'mi')
        dist_scale = KM_TO_MI if use_miles else 1.0
        dist_unit = 'mi' if use_miles else 'km'

        distances = data['distances'] * dist_scale
        pressure_levels = data['pressure_levels']
        theta = data.get('theta')
        temperature = data.get('temperature')
        surface_pressure = data.get('surface_pressure')
        u_wind = data.get('u_wind')
        v_wind = data.get('v_wind')
        lats = data['lats']
        lons = data['lons']
        geopotential_height = data.get('geopotential_height')

        # Parse metadata for labels
        metadata = metadata or {}
        model = metadata.get('model', 'HRRR')
        init_date = metadata.get('init_date')
        init_hour = metadata.get('init_hour')
        forecast_hour = metadata.get('forecast_hour', 0)

        # Calculate valid time
        if init_date and init_hour:
            try:
                init_dt = datetime.strptime(f"{init_date}{init_hour}", "%Y%m%d%H")
                valid_dt = init_dt + timedelta(hours=forecast_hour)
                init_str = init_dt.strftime("%Y-%m-%d %HZ")
                valid_str = valid_dt.strftime("%Y-%m-%d %HZ")
            except:
                init_str = f"{init_date} {init_hour}Z" if init_date else "Unknown"
                valid_str = "Unknown"
        else:
            init_str = "Unknown"
            valid_str = "Unknown"

        n_levels, n_points = theta.shape if theta is not None else (len(pressure_levels), len(distances))

        # Compute wind speed
        if u_wind is not None and v_wind is not None:
            wind_speed = np.sqrt(u_wind**2 + v_wind**2) * 1.944
        else:
            wind_speed = None

        # Build terrain mask for contour lines (prevents theta/freezing level going underground)
        # contourf is left unmasked — terrain fill (zorder=5) covers it visually
        terrain_mask = None
        if surface_pressure is not None:
            terrain_mask = np.zeros((len(pressure_levels), n_points), dtype=bool)
            for i in range(n_points):
                terrain_mask[:, i] = pressure_levels > surface_pressure[i]

        # Create figure - 25% larger with room for inset above and labels below
        base_height = 11.0
        fig_height = base_height * min(max(vscale, 0.5), 3.0)
        fig, ax = plt.subplots(figsize=(17, fig_height), facecolor='white')
        ax.set_position([0.06, 0.12, 0.82, 0.68])  # Room above for inset, below for labels

        # Determine Y coordinate based on y_axis choice
        use_height = (y_axis == 'height' and geopotential_height is not None)

        # Filter to vertical range (y_top sets the top of the plot in hPa)
        # Pressure levels are ordered high-to-low (1000, 975, 950, ... 100)
        # We want levels >= y_top (i.e., from surface up to y_top)
        level_mask = pressure_levels >= y_top
        pressure_levels_filtered = pressure_levels[level_mask]

        # Helper to filter any 3D array to vertical range
        def filter_levels(arr):
            if arr is None:
                return None
            return arr[level_mask, :]

        # Filter all 3D data arrays to the selected vertical range
        theta = filter_levels(theta)
        wind_speed = filter_levels(wind_speed)
        u_wind = filter_levels(u_wind)
        v_wind = filter_levels(v_wind)
        geopotential_height = filter_levels(geopotential_height)

        # Also filter style-specific arrays from data dict
        # Note: smoke_hyb/smoke_pres_hyb are on native hybrid levels, not isobaric — filtered separately in render
        for key in ['rh', 'omega', 'vorticity', 'cloud', 'temperature', 'temp_c',
                    'specific_humidity', 'theta_e', 'shear', 'dew_point', 'frontogenesis',
                    'icing', 'wetbulb', 'lapse_rate']:
            if key in data and data[key] is not None and data[key].ndim == 2:
                data[key] = filter_levels(data[key])

        # Filter terrain mask to same vertical range
        if terrain_mask is not None:
            terrain_mask = terrain_mask[level_mask, :]

        # Re-read temperature after filtering (local var was extracted before filter_levels)
        temperature = data.get('temperature')

        n_levels = len(pressure_levels_filtered)

        if use_height:
            # Convert geopotential height to km (gpm -> km)
            # Use mean height at each pressure level for Y coordinate
            heights_km = np.nanmean(geopotential_height, axis=1) / 1000.0  # gpm to km
            X, Y = np.meshgrid(distances, heights_km)
            y_coord = heights_km
        else:
            X, Y = np.meshgrid(distances, pressure_levels_filtered)
            y_coord = pressure_levels_filtered

        # Style-specific shading
        shading_label = style

        if style == "wind_speed" and wind_speed is not None:
            # Wind speed colormap: PG&E/SJSU-WIRC style (smooth gradient)
            # Blues for light-moderate, magenta for strong, yellow-orange-red for extreme
            wspd_colors = [
                (0.00, '#FFFFFF'),  # 0 kt - calm (white)
                (0.10, '#E3F2FD'),  # 10 kt - light (very light blue)
                (0.20, '#90CAF9'),  # 20 kt - light (light blue)
                (0.30, '#42A5F5'),  # 30 kt - moderate (blue)
                (0.40, '#1E88E5'),  # 40 kt - moderate (medium blue)
                (0.50, '#7B1FA2'),  # 50 kt - fresh (magenta)
                (0.60, '#E91E63'),  # 60 kt - strong (pink-magenta)
                (0.70, '#FFEB3B'),  # 70 kt - very strong (yellow)
                (0.80, '#FFC107'),  # 80 kt - gale (gold)
                (0.90, '#FF9800'),  # 90 kt - strong gale (orange)
                (1.00, '#F44336'),  # 100 kt - extreme (red)
            ]
            # Create smooth colormap from color stops
            colors_only = [c[1] for c in wspd_colors]
            wspd_cmap = mcolors.LinearSegmentedColormap.from_list('wspd', colors_only, N=256)
            # Fine resolution levels for smooth gradient (every 2 kts)
            cf = ax.contourf(X, Y, wind_speed, levels=np.arange(0, 102, 2), cmap=wspd_cmap, extend='max')
            cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
            cbar = plt.colorbar(cf, cax=cbar_ax)
            cbar.set_label('Wind Speed (kts)')
            shading_label = "Wind Speed"
        elif style == "temp":
            temp_c = data.get('temp_c')
            if temp_c is not None:
                # Build colormap from selected option
                cmap_obj = self._build_temp_colormap(temp_cmap)
                cf = ax.contourf(X, Y, temp_c, levels=np.arange(-65, 55, 2), cmap=cmap_obj, extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Temperature (°C)')
                shading_label = "T(°C)"
        elif style == "rh":
            rh = data.get('rh')
            if rh is not None:
                rh_colors = [(0.6, 0.4, 0.2), (0.7, 0.5, 0.3), (0.85, 0.75, 0.5),
                             (0.9, 0.9, 0.7), (0.7, 0.9, 0.7), (0.4, 0.8, 0.4),
                             (0.2, 0.6, 0.3), (0.1, 0.4, 0.2)]
                rh_cmap = mcolors.LinearSegmentedColormap.from_list('rh', rh_colors, N=256)
                cf = ax.contourf(X, Y, rh, levels=np.arange(0, 105, 5), cmap=rh_cmap, extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Relative Humidity (%)')
                shading_label = "RH(%)"
        elif style == "omega":
            omega = data.get('omega')
            if omega is not None:
                omega_display = omega * 36.0
                omega_max = min(np.nanmax(np.abs(omega_display)), 20)
                cf = ax.contourf(X, Y, omega_display, levels=np.linspace(-omega_max, omega_max, 21),
                                cmap='RdBu_r', extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('ω (hPa/hr)')
                shading_label = "ω"
        elif style == "theta_e":
            theta_e = data.get('theta_e')
            if theta_e is not None:
                cf = ax.contourf(X, Y, theta_e, levels=np.arange(280, 365, 4), cmap='Spectral_r', extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('θₑ (K)')
                shading_label = "θₑ"
        elif style == "shear":
            shear = data.get('shear')
            if shear is not None:
                cf = ax.contourf(X, Y, shear, levels=np.linspace(0, 10, 11), cmap='OrRd', extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Shear (10⁻³/s)')
                shading_label = "Shear"
        elif style == "q":
            q = data.get('specific_humidity')
            if q is not None:
                q_gkg = q * 1000  # kg/kg to g/kg
                cf = ax.contourf(X, Y, q_gkg, levels=np.arange(0, 21, 1), cmap='YlGnBu', extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Specific Humidity (g/kg)')
                shading_label = "q"
        elif style == "cloud_total":
            cloud = data.get('cloud')
            if cloud is not None:
                cloud_gkg = cloud * 1000  # kg/kg to g/kg
                # Cloud colormap: clear sky to dense cloud
                cloud_colors = [
                    '#FFFFFF',  # Clear
                    '#F0F0F5',  # Very thin
                    '#D8DCE8',  # Thin cirrus
                    '#B8C4D8',  # Light cloud
                    '#98ACC8',  # Moderate
                    '#7894B8',  # Thick
                    '#5878A8',  # Dense
                    '#385898',  # Very dense
                ]
                cloud_cmap = mcolors.LinearSegmentedColormap.from_list('cloud', cloud_colors, N=256)
                cf = ax.contourf(X, Y, cloud_gkg, levels=np.linspace(0, 1.0, 11), cmap=cloud_cmap, extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Total Condensate (g/kg)')
                shading_label = "Cloud"
        elif style == "cloud":
            cloud = data.get('cloud')
            if cloud is not None:
                cloud_gkg = cloud * 1000
                cf = ax.contourf(X, Y, cloud_gkg, levels=np.linspace(0, 0.5, 11), cmap='Blues', extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Cloud LWC (g/kg)')
                shading_label = "Cloud"
        elif style == "lapse_rate":
            lapse = data.get('lapse_rate')
            if lapse is not None:
                # Diverging colormap: blue (stable) -> white -> red (unstable)
                cf = ax.contourf(X, Y, lapse, levels=np.linspace(0, 12, 13), cmap='RdYlBu_r', extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Lapse Rate (°C/km)')
                # Add dry adiabatic line
                ax.contour(X, Y, lapse, levels=[9.8], colors='black', linewidths=2, linestyles='--')
                shading_label = "Γ"
        elif style == "wetbulb":
            wetbulb = data.get('wetbulb')
            if wetbulb is not None:
                cf = ax.contourf(X, Y, wetbulb, levels=np.arange(-40, 35, 5), cmap='coolwarm', extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Wet-Bulb Temp (°C)')
                # Wet-bulb 0°C line (critical for precip type)
                try:
                    cs_wb0 = ax.contour(X, Y, wetbulb, levels=[0], colors='lime', linewidths=3)
                    ax.clabel(cs_wb0, inline=True, fontsize=9, fmt='Tw=0°C', colors='black')
                except:
                    pass
                shading_label = "Tw"
        elif style == "icing":
            icing = data.get('icing')
            if icing is not None:
                # Icing colormap: clear → light blue → cyan → intense blue
                # Blues convey "cold/ice hazard" intuitively
                icing_colors = [
                    '#FFFFFF',  # None
                    '#E3F2FD',  # Trace
                    '#BBDEFB',  # Light
                    '#64B5F6',  # Light-Moderate
                    '#2196F3',  # Moderate
                    '#1565C0',  # Moderate-Severe
                    '#0D47A1',  # Severe
                ]
                icing_cmap = mcolors.LinearSegmentedColormap.from_list('icing', icing_colors, N=256)
                cf = ax.contourf(X, Y, icing, levels=np.linspace(0, 0.3, 7), cmap=icing_cmap, extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Icing (SLW g/kg)')
                shading_label = "Icing"
        elif style == "vorticity":
            vort = data.get('vorticity')
            if vort is not None:
                vort_scaled = vort * 1e5  # Scale for display
                vort_max = min(np.nanmax(np.abs(vort_scaled)), 30)
                cf = ax.contourf(X, Y, vort_scaled, levels=np.linspace(-vort_max, vort_max, 21),
                                cmap='RdBu_r', extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Vorticity (10⁻⁵/s)')
                shading_label = "ζ"
        elif style == "smoke":
            smoke_hyb = data.get('smoke_hyb')  # (n_hyb, n_points)
            smoke_pres = data.get('smoke_pres_hyb')  # (n_hyb, n_points) — pressure in hPa
            if smoke_hyb is not None and smoke_pres is not None:
                from config.colormaps import create_all_colormaps
                smoke_cmap = create_all_colormaps()['NOAASmoke']

                # Build smoke's own X/Y mesh on native hybrid levels
                # Y = per-column pressure (varies with terrain), X = distance along path
                n_hyb, n_pts = smoke_hyb.shape
                X_smoke = np.broadcast_to(distances[np.newaxis, :], (n_hyb, n_pts))
                Y_smoke = smoke_pres  # (n_hyb, n_pts) — native pressure coordinate

                # Filter to visible pressure range (y_top to 1050 hPa)
                # Keep hybrid levels where at least some points are in range
                y_bot = 1050.0
                mask = np.any((Y_smoke >= y_top) & (Y_smoke <= y_bot), axis=1)
                if np.any(mask):
                    X_smoke = X_smoke[mask]
                    Y_smoke = Y_smoke[mask]
                    smoke_plot = smoke_hyb[mask]
                else:
                    smoke_plot = smoke_hyb

                # Auto-scale with many levels for smooth shading
                smoke_max = np.nanmax(smoke_plot)
                if smoke_max > 100:
                    cap = max(250, min(500, smoke_max * 1.1))
                    levels = np.concatenate([
                        np.linspace(0, 10, 10, endpoint=False),
                        np.linspace(10, 55, 15, endpoint=False),
                        np.linspace(55, cap, 25),
                    ])
                elif smoke_max > 10:
                    cap = max(55, smoke_max * 1.2)
                    levels = np.linspace(0, cap, 50)
                else:
                    levels = np.linspace(0, max(smoke_max * 1.3, 5), 50)

                # Plot on smoke's own native-level grid (NOT the isobaric X, Y)
                cf = ax.contourf(X_smoke, Y_smoke, smoke_plot, levels=levels,
                                 cmap=smoke_cmap, extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                if smoke_max > 100:
                    ticks = [0, 5, 10, 20, 35, 55, 100, 150, 250]
                    ticks = [t for t in ticks if t <= levels[-1]]
                elif smoke_max > 10:
                    ticks = [0, 5, 10, 20, 35, 55]
                    ticks = [t for t in ticks if t <= levels[-1]]
                else:
                    ticks = None
                cbar = plt.colorbar(cf, cax=cbar_ax, ticks=ticks)
                cbar.set_label('PM2.5 (μg/m³)')


                shading_label = "PM2.5 Smoke"
        elif style == "frontogenesis":
            # Winter Bander Mode - Petterssen Frontogenesis
            fronto = data.get('frontogenesis')
            if fronto is not None:
                # Diverging colormap: blue (frontolysis) -> white -> red (frontogenesis)
                # Red = frontogenesis (temperature gradient increasing) = banding
                # Blue = frontolysis (temperature gradient decreasing)
                fronto_colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0',
                                '#F7F7F7',
                                '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
                fronto_cmap = mcolors.LinearSegmentedColormap.from_list('fronto', fronto_colors, N=256)

                # Levels centered on zero, range typically -2 to +2 K/100km/3hr
                levels = np.linspace(-2, 2, 21)
                cf = ax.contourf(X, Y, fronto, levels=levels, cmap=fronto_cmap, extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Frontogenesis (K/100km/3hr)')

                # Highlight strong frontogenesis bands
                try:
                    cs_strong = ax.contour(X, Y, fronto, levels=[0.5, 1.0, 1.5],
                                          colors='darkred', linewidths=[0.8, 1.2, 1.6])
                except:
                    pass

                shading_label = "❄ Frontogenesis"
        else:
            # Default to theta shading
            if theta is not None:
                cf = ax.contourf(X, Y, theta, levels=np.arange(270, 360, 4), cmap='viridis', extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('θ (K)')
                shading_label = "θ"

        # Theta contours (masked below terrain)
        if theta is not None:
            theta_plot = np.ma.masked_where(terrain_mask, theta) if terrain_mask is not None else theta
            cs = ax.contour(X, Y, theta_plot, levels=np.arange(270, 330, 4), colors='black', linewidths=0.8)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

        # Temperature overlays (freezing level, DGZ, additional isotherms) — masked below terrain
        if temperature is not None:
            temp_c_raw = temperature - 273.15
            temp_c_plot = np.ma.masked_where(terrain_mask, temp_c_raw) if terrain_mask is not None else temp_c_raw
            try:
                # Freezing level (0°C) - magenta
                ax.contour(X, Y, temp_c_plot, levels=[0], colors='magenta', linewidths=2)

                # Additional isotherms for winter weather styles
                if style in ['icing', 'wetbulb', 'temp', 'cloud', 'cloud_total']:
                    # -10°C and -20°C isotherms
                    cs_iso = ax.contour(X, Y, temp_c_plot, levels=[-10, -20],
                                       colors=['cyan', 'blue'], linewidths=1.2, linestyles='--')
                    ax.clabel(cs_iso, inline=True, fontsize=8, fmt='%.0f°C')

                # DGZ band (-12 to -18°C) for snow growth - highlight on relevant styles
                if style in ['icing', 'wetbulb', 'cloud', 'cloud_total', 'temp']:
                    dgz_mask = (temp_c_plot >= -18) & (temp_c_plot <= -12)
                    if np.any(dgz_mask):
                        ax.contourf(X, Y, dgz_mask.astype(float), levels=[0.5, 1.5],
                                   colors=['lightblue'], alpha=0.25, zorder=2)
            except:
                pass

        # Wind barbs
        if u_wind is not None and v_wind is not None:
            # Subsample for readability
            x_skip = max(1, n_points // 25)
            y_skip = max(1, n_levels // 12)

            x_idx = np.arange(0, n_points, x_skip)
            y_idx = np.arange(0, n_levels, y_skip)

            X_barb = distances[x_idx]
            Y_barb = y_coord[y_idx]  # Use appropriate Y coordinate (height or pressure)
            XX_barb, YY_barb = np.meshgrid(X_barb, Y_barb)

            # Get wind at subsampled points
            U_barb = u_wind[np.ix_(y_idx, x_idx)].copy()
            V_barb = v_wind[np.ix_(y_idx, x_idx)].copy()

            # Mask below terrain
            if surface_pressure is not None:
                for i, xi in enumerate(x_idx):
                    sp = surface_pressure[xi]
                    for j, yj in enumerate(y_idx):
                        if pressure_levels_filtered[yj] > sp:
                            U_barb[j, i] = np.nan
                            V_barb[j, i] = np.nan

            # Convert m/s to knots
            U_kt = U_barb * 1.944
            V_kt = V_barb * 1.944

            # Show wind direction relative to cross-section orientation
            # Barbs point in direction wind is FROM
            # U_kt = eastward component, V_kt = northward component
            U_rot = U_kt
            V_rot = V_kt

            # Plot wind barbs
            ax.barbs(
                XX_barb, YY_barb, U_rot, V_rot,
                length=5, barbcolor='black', flagcolor='black',
                linewidth=0.6, pivot='middle',
                sizes=dict(emptybarb=0.04, spacing=0.12, height=0.35),
            )

        # Terrain fill - use high-resolution terrain data if available
        if surface_pressure is not None:
            # Prefer native high-res terrain from extraction
            sp_hires = data.get('surface_pressure_hires')
            dist_hires = data.get('distances_hires')

            if sp_hires is None or dist_hires is None:
                # Fallback: use native ~3km HRRR resolution
                from scipy.interpolate import interp1d
                total_dist_km = distances[-1]
                terrain_res = max(50, int(total_dist_km / 3))  # Native 3km spacing
                dist_hires = np.linspace(distances[0], distances[-1], terrain_res)
                try:
                    sp_interp = interp1d(distances, surface_pressure, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
                    sp_hires = sp_interp(dist_hires)
                except Exception:
                    sp_hires = np.interp(dist_hires, distances, surface_pressure)

            if use_height:
                # For height axis: convert surface pressure to approximate terrain height
                # Use hypsometric equation: z ≈ 44330 * (1 - (P/P0)^0.19) meters
                # where P0 = 1013.25 hPa (sea level pressure)
                terrain_height_m = 44330 * (1 - (sp_hires / 1013.25) ** 0.19)
                terrain_height_km = terrain_height_m / 1000.0
                terrain_x = np.concatenate([[dist_hires[0]], dist_hires, [dist_hires[-1]]])
                terrain_y = np.concatenate([[0], terrain_height_km, [0]])
                ax.fill(terrain_x, terrain_y, color='saddlebrown', alpha=0.9, zorder=5)
                ax.plot(dist_hires, terrain_height_km, 'k-', linewidth=1.5, zorder=6)
            else:
                # Pressure axis: fill below surface pressure
                max_p = max(pressure_levels_filtered.max(), np.nanmax(sp_hires)) + 20
                terrain_x = np.concatenate([[dist_hires[0]], dist_hires, [dist_hires[-1]]])
                terrain_y = np.concatenate([[max_p], sp_hires, [max_p]])
                ax.fill(terrain_x, terrain_y, color='saddlebrown', alpha=0.9, zorder=5)
                ax.plot(dist_hires, sp_hires, 'k-', linewidth=1.5, zorder=6)

        # Axes - configure based on y_axis choice
        ax.set_xlim(0, distances[-1])
        ax.set_xlabel(f'Distance ({dist_unit})', fontsize=11)

        if use_height:
            # Height axis: 0 at bottom, increasing upward
            ax.set_ylim(0, max(y_coord))
            ax.set_ylabel('Height (km)', fontsize=11)
            ax.yaxis.set_major_locator(MultipleLocator(2))  # Every 2 km
        else:
            # Pressure axis: high values at bottom (surface), low at top
            ax.set_ylim(max(pressure_levels_filtered), min(pressure_levels_filtered))
            ax.set_ylabel('Pressure (hPa)', fontsize=11)
            ax.yaxis.set_major_locator(MultipleLocator(100))

        ax.grid(True, alpha=0.3)

        # Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_items = []
        if theta is not None:
            legend_items.append(Line2D([0], [0], color='black', linewidth=0.8, label='Potential Temp (K)'))
        if temperature is not None:
            legend_items.append(Line2D([0], [0], color='magenta', linewidth=2, label='0°C Isotherm'))
        if u_wind is not None and v_wind is not None:
            legend_items.append(Line2D([0], [0], marker=r'$\rightarrow$', color='black', linestyle='None',
                                       markersize=8, label='Wind Barbs (kt)'))
        if surface_pressure is not None:
            legend_items.append(Patch(facecolor='saddlebrown', alpha=0.9, label='Terrain'))
        if legend_items:
            ax.legend(handles=legend_items, loc='upper left', fontsize=7, framealpha=0.85,
                     edgecolor='#999', fancybox=False, borderpad=0.4, handlelength=1.5)

        # Title with full metadata
        title_main = f'{model} Cross-Section: {shading_label}'
        ax.set_title(title_main, fontsize=14, fontweight='bold', loc='left')
        ax.set_title(f'Init: {init_str}  |  F{forecast_hour:02d}  |  Valid: {valid_str}',
                    fontsize=10, loc='right', color='#555')

        # A/B endpoint labels below the secondary axis
        total_dist = distances[-1]
        a_label = f"A\n{abs(lats[0]):.2f}°{'N' if lats[0]>=0 else 'S'}, {abs(lons[0]):.2f}°{'W' if lons[0]<0 else 'E'}"
        b_label = f"B\n{abs(lats[-1]):.2f}°{'N' if lats[-1]>=0 else 'S'}, {abs(lons[-1]):.2f}°{'W' if lons[-1]<0 else 'E'}"
        fig.text(0.06, 0.045, a_label, ha='left', va='top', fontsize=8, fontweight='bold',
                color='#333', transform=fig.transFigure)
        fig.text(0.88, 0.045, b_label, ha='right', va='top', fontsize=8, fontweight='bold',
                color='#333', transform=fig.transFigure)

        # Lat/lon + nearby city labels along x-axis
        _cities = [
            # Major metros
            (40.71, -74.01, "New York"), (34.05, -118.24, "Los Angeles"), (41.88, -87.63, "Chicago"),
            (29.76, -95.37, "Houston"), (33.45, -112.07, "Phoenix"), (29.95, -90.07, "New Orleans"),
            (39.74, -104.99, "Denver"), (47.61, -122.33, "Seattle"), (25.76, -80.19, "Miami"),
            (33.75, -84.39, "Atlanta"), (42.36, -71.06, "Boston"), (38.91, -77.04, "Washington DC"),
            (32.72, -96.97, "Dallas"), (37.77, -122.42, "San Francisco"), (36.17, -115.14, "Las Vegas"),
            (39.96, -82.99, "Columbus"), (35.23, -80.84, "Charlotte"), (44.98, -93.27, "Minneapolis"),
            (30.27, -97.74, "Austin"), (32.22, -110.93, "Tucson"), (36.16, -86.78, "Nashville"),
            (45.52, -122.68, "Portland OR"), (38.63, -90.20, "St. Louis"), (39.10, -94.58, "Kansas City"),
            (35.47, -97.52, "Oklahoma City"), (37.34, -121.89, "San Jose"),
            (40.76, -111.89, "Salt Lake City"), (43.62, -116.21, "Boise"), (42.33, -83.05, "Detroit"),
            (39.77, -86.16, "Indianapolis"), (41.26, -95.94, "Omaha"), (28.54, -81.38, "Orlando"),
            (27.95, -82.46, "Tampa"), (30.33, -81.66, "Jacksonville"), (40.44, -79.99, "Pittsburgh"),
            # Mid-size & regional
            (35.96, -83.92, "Knoxville"), (36.85, -75.98, "Virginia Beach"), (43.04, -87.91, "Milwaukee"),
            (34.73, -92.33, "Little Rock"), (32.30, -90.18, "Jackson MS"), (37.54, -77.44, "Richmond"),
            (42.89, -78.88, "Buffalo"), (43.05, -76.15, "Syracuse"), (41.76, -72.68, "Hartford"),
            (46.87, -114.00, "Missoula"), (47.66, -117.43, "Spokane"), (35.08, -106.65, "Albuquerque"),
            (31.77, -106.44, "El Paso"), (37.69, -97.34, "Wichita"), (38.80, -97.61, "Salina"),
            (41.59, -93.62, "Des Moines"), (40.81, -96.70, "Lincoln"), (46.81, -100.78, "Bismarck"),
            (38.04, -84.50, "Lexington"), (34.00, -81.03, "Columbia SC"), (36.07, -79.79, "Greensboro"),
            (26.12, -80.14, "Fort Lauderdale"), (43.66, -70.26, "Portland ME"),
            # Western US fill (sparse areas)
            (38.57, -121.49, "Sacramento"), (36.75, -119.77, "Fresno"), (35.37, -119.02, "Bakersfield"),
            (34.42, -119.70, "Santa Barbara"), (33.43, -117.61, "San Clemente"),
            (33.95, -117.40, "Riverside"), (32.72, -117.16, "San Diego"),
            (36.60, -121.89, "Monterey"), (40.59, -122.39, "Redding"), (42.32, -122.87, "Medford"),
            (39.53, -119.81, "Reno"), (38.80, -112.08, "Richfield UT"), (39.17, -119.77, "Carson City"),
            (40.83, -115.76, "Elko"), (38.54, -109.55, "Moab"), (37.10, -113.58, "St. George"),
            (39.64, -106.37, "Vail"), (38.27, -104.76, "Pueblo"), (40.59, -105.08, "Fort Collins"),
            (35.20, -101.83, "Amarillo"), (33.58, -101.85, "Lubbock"), (31.97, -99.90, "Abilene"),
            (27.80, -97.40, "Corpus Christi"), (32.45, -100.43, "Sweetwater"),
            (41.14, -104.82, "Cheyenne"), (42.87, -106.33, "Casper"), (44.77, -108.73, "Cody"),
            (43.48, -110.76, "Jackson WY"), (46.60, -112.04, "Helena"),
            (48.76, -122.47, "Bellingham"), (44.06, -121.31, "Bend OR"),
            (46.73, -117.00, "Lewiston ID"), (44.26, -72.58, "Montpelier"),
            (43.21, -71.54, "Concord NH"), (41.82, -71.41, "Providence"),
            (39.16, -75.52, "Dover"), (44.37, -100.35, "Pierre SD"), (44.97, -89.63, "Wausau"),
            (38.58, -92.17, "Jefferson City"), (37.22, -93.29, "Springfield MO"),
            (36.40, -105.57, "Taos"), (34.51, -105.38, "Roswell"), (32.34, -106.76, "Las Cruces"),
            (48.23, -101.30, "Minot"), (47.51, -111.28, "Great Falls"),
            (39.83, -98.53, "Smith Center KS"), (41.13, -100.77, "North Platte"),
            (44.08, -103.23, "Rapid City"), (38.73, -116.69, "Tonopah"),
            (37.23, -115.02, "Alamo NV"), (40.84, -111.89, "SLC"),
            (21.31, -157.86, "Honolulu"), (61.22, -149.90, "Anchorage"),
        ]
        n_ticks = min(8, max(3, int(total_dist / 200)))  # ~1 tick per 200km
        tick_indices = np.linspace(0, len(distances) - 1, n_ticks + 2, dtype=int)[1:-1]  # Skip A/B endpoints

        tick_positions = []
        tick_labels = []
        used_cities = set()  # Prevent same city appearing twice
        for idx in tick_indices:
            d = distances[idx]
            lat_i, lon_i = lats[idx], lons[idx]

            # Find nearest city within 120km that hasn't been used
            best_city = None
            best_dist = 120  # km threshold
            for clat, clon, cname in _cities:
                if cname in used_cities:
                    continue
                cdist = np.sqrt(((lat_i - clat) * 111) ** 2 + (((lon_i - clon) * 111 * np.cos(np.radians(lat_i)))) ** 2)
                if cdist < best_dist:
                    best_dist = cdist
                    best_city = cname

            label = f"{abs(lat_i):.1f}°{'N' if lat_i >= 0 else 'S'}, {abs(lon_i):.1f}°{'W' if lon_i < 0 else 'E'}"
            if best_city:
                label += f"\n{best_city}"
                used_cities.add(best_city)
            label += f"\n{d:.0f} {dist_unit}"
            tick_positions.append(d)
            tick_labels.append(label)

        # Secondary x-axis with lat/lon + city labels below the main axis
        ax2 = ax.secondary_xaxis(-0.08)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, fontsize=7, color='#555', ha='center')
        ax2.tick_params(axis='x', length=4, color='#999')

        # Path coords incorporated into A/B labels - no separate text needed

        # Add professional inset map showing cross-section path
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            # Calculate extent with padding
            lon_min, lon_max = min(lons) - 2, max(lons) + 2
            lat_min, lat_max = min(lats) - 1.5, max(lats) + 1.5

            # Ensure minimum extent so map doesn't get too skinny
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min

            # Enforce minimum aspect ratio (lon:lat around 1.5:1 for reasonable look)
            min_lat_range = lon_range / 2.5  # At least this much latitude span
            if lat_range < min_lat_range:
                lat_center = (lat_min + lat_max) / 2
                lat_min = lat_center - min_lat_range / 2
                lat_max = lat_center + min_lat_range / 2

            # Calculate inset size based on aspect ratio
            aspect = lon_range / max(lat_range, 0.1)
            inset_width = min(0.25, max(0.16, 0.20 * (aspect / 2)))
            inset_height = 0.12

            # Place inset map above plot, right-aligned (in the top margin)
            axins = fig.add_axes([0.88 - inset_width, 0.83, inset_width, inset_height],
                                projection=ccrs.PlateCarree())
            axins.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

            # Add terrain/imagery background
            try:
                axins.stock_img()  # Natural Earth terrain
            except:
                # Fallback to simple features
                axins.add_feature(cfeature.LAND, facecolor='#C4B896', zorder=0)
                axins.add_feature(cfeature.OCEAN, facecolor='#97B6C8', zorder=0)

            # Add map features
            axins.add_feature(cfeature.LAKES, facecolor='#97B6C8', edgecolor='#6090A0', linewidth=0.3, zorder=1)
            axins.add_feature(cfeature.STATES, edgecolor='#666666', linewidth=0.4, zorder=2)
            axins.add_feature(cfeature.BORDERS, edgecolor='#333333', linewidth=0.6, zorder=2)
            axins.add_feature(cfeature.COASTLINE, edgecolor='#444444', linewidth=0.5, zorder=2)

            # Draw cross-section path
            axins.plot(lons, lats, 'r-', linewidth=2.5, transform=ccrs.PlateCarree(), zorder=10)
            # Start point - A label
            axins.text(lons[0], lats[0], 'A', transform=ccrs.PlateCarree(), zorder=11,
                      fontsize=10, fontweight='bold', ha='center', va='center',
                      color='white', bbox=dict(boxstyle='round,pad=0.15', facecolor='#38bdf8',
                                               edgecolor='white', linewidth=1.5))
            # End point - B label
            axins.text(lons[-1], lats[-1], 'B', transform=ccrs.PlateCarree(), zorder=11,
                      fontsize=10, fontweight='bold', ha='center', va='center',
                      color='white', bbox=dict(boxstyle='round,pad=0.15', facecolor='#f87171',
                                               edgecolor='white', linewidth=1.5))

            # Style the border
            for spine in axins.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

        except Exception as e:
            pass  # Skip inset if cartopy fails

        # Add credit
        fig.text(0.5, 0.005, 'Contributors: @jasonbweather, justincat66, Sequoiagrove & others',
                 ha='center', va='bottom', fontsize=8, color='#888888',
                 transform=fig.transFigure, style='italic', fontweight='bold')

        # Save to bytes (don't use tight_layout or bbox_inches - conflicts with inset positioning)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, facecolor='white')
        plt.close(fig)
        buf.seek(0)

        return buf.read()

    def get_loaded_hours(self) -> List[int]:
        """Get list of loaded forecast hours."""
        return sorted(self.forecast_hours.keys())

    def get_memory_usage(self) -> float:
        """Get total memory usage in MB."""
        try:
            return sum(fh.memory_usage_mb() for fh in list(self.forecast_hours.values()))
        except RuntimeError:
            # Dict changed size during iteration (concurrent load/unload)
            return 0.0

    def unload_hour(self, forecast_hour: int):
        """Unload a forecast hour to free memory."""
        if forecast_hour in self.forecast_hours:
            del self.forecast_hours[forecast_hour]


# Convenience function for testing
def test_interactive():
    """Test interactive cross-section performance."""
    ixs = InteractiveCrossSection()

    # Load a single hour
    run_dir = Path("outputs/hrrr/20251224/19z")
    if not run_dir.exists():
        print("No test data available")
        return

    prs_file = list((run_dir / "F00").glob("*wrfprs*.grib2"))
    if not prs_file:
        print("No GRIB file found")
        return

    print("\n=== Loading data ===")
    ixs.load_forecast_hour(str(prs_file[0]), 0)

    print(f"\nMemory usage: {ixs.get_memory_usage():.0f} MB")

    # Test multiple cross-sections
    print("\n=== Testing cross-section generation ===")

    paths = [
        ((39.74, -104.99), (41.88, -87.63), "Denver → Chicago"),
        ((34.05, -118.24), (33.45, -112.07), "LA → Phoenix"),
        ((40.71, -74.01), (42.36, -71.06), "NYC → Boston"),
    ]

    styles = ['wind_speed', 'temp', 'theta_e', 'rh', 'omega']

    for start, end, name in paths:
        print(f"\n{name}:")
        for style in styles:
            t0 = time.time()
            img = ixs.get_cross_section(start, end, style=style, forecast_hour=0)
            duration = time.time() - t0
            print(f"  {style:12}: {duration:.3f}s ({len(img)/1024:.0f} KB)")


if __name__ == "__main__":
    test_interactive()
