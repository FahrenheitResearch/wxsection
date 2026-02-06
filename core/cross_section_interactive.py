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

    # Source GRIB path (for lazy smoke backfill)
    grib_file: str = None

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB.

        Memory-mapped arrays (np.memmap) report 0 MB since the OS page cache
        manages their physical memory, not the Python heap.
        """
        total = 0
        for name, val in self.__dict__.items():
            if isinstance(val, np.ndarray) and not isinstance(val, np.memmap):
                total += val.nbytes
        return total / 1024 / 1024


@dataclass
class ClimatologyData:
    """Holds coarsened climatology grid for anomaly computation."""
    month: int
    init_hour: int
    fhr: int
    pressure_levels: np.ndarray      # (n_levels,)
    lats: np.ndarray                 # (ny_coarse,) or (ny_coarse, nx_coarse)
    lons: np.ndarray                 # (nx_coarse,) or (ny_coarse, nx_coarse)

    # Coarsened mean fields: (n_levels, ny_coarse, nx_coarse)
    temperature: np.ndarray = None
    u_wind: np.ndarray = None
    v_wind: np.ndarray = None
    rh: np.ndarray = None
    omega: np.ndarray = None
    specific_humidity: np.ndarray = None
    geopotential_height: np.ndarray = None
    vorticity: np.ndarray = None

    n_samples: int = 0
    years: List[int] = field(default_factory=list)


# Styles that support anomaly mode
ANOMALY_STYLES = {
    'temp', 'wind_speed', 'rh', 'omega', 'theta_e',
    'q', 'vorticity', 'shear', 'lapse_rate', 'wetbulb',
    'vpd', 'dewpoint_dep', 'moisture_transport',
}


# Standalone function for multiprocessing (must be at module level for pickle)
def _load_hour_process(
    grib_file: str,
    forecast_hour: int,
    grib_backend: str = 'cfgrib',
    sfc_file: Optional[str] = None,
    nat_file: Optional[str] = None,
) -> Optional[ForecastHourData]:
    """Load a single forecast hour for ProcessPoolExecutor workers.

    Supports the same backend selection strategy as InteractiveCrossSection:
    - 'cfgrib': use cfgrib for core fields
    - 'eccodes': use one-pass eccodes for core fields
    - 'auto': try eccodes, then fallback to cfgrib
    """

    fields = {
        'u': 'u_wind', 'v': 'v_wind', 'r': 'rh', 'w': 'omega',
        'q': 'specific_humidity', 'gh': 'geopotential_height',
        'absv': 'vorticity', 'clwmr': 'cloud', 'dpt': 'dew_point',
    }

    backend = (grib_backend or 'cfgrib').strip().lower()
    if backend not in {'cfgrib', 'eccodes', 'auto'}:
        print(f"  Warning: invalid grib backend '{grib_backend}', using cfgrib")
        backend = 'cfgrib'
    backend_order = ['eccodes', 'cfgrib'] if backend == 'auto' else [backend]

    resolved_sfc = sfc_file
    if not resolved_sfc:
        sfc_guess = Path(grib_file).parent / Path(grib_file).name.replace('wrfprs', 'wrfsfc')
        resolved_sfc = str(sfc_guess) if sfc_guess.exists() else grib_file

    resolved_nat = nat_file
    if not resolved_nat:
        nat_guess = Path(grib_file).parent / Path(grib_file).name.replace('wrfprs', 'wrfnat')
        resolved_nat = str(nat_guess) if nat_guess.exists() else None

    def _decode_msg_to_2d(msg):
        import eccodes
        try:
            ni = int(eccodes.codes_get(msg, 'Ni'))
            nj = int(eccodes.codes_get(msg, 'Nj'))
        except Exception:
            ni = int(eccodes.codes_get(msg, 'Nx'))
            nj = int(eccodes.codes_get(msg, 'Ny'))
        return eccodes.codes_get_values(msg).reshape(nj, ni)

    def _load_core_cfgrib() -> ForecastHourData:
        import cfgrib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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

            data = ForecastHourData(
                forecast_hour=forecast_hour,
                pressure_levels=pressure_levels,
                lats=lats,
                lons=lons,
                temperature=t_data.values,
            )
            ds_t.close()

            for grib_key, field_name in fields.items():
                try:
                    ds = cfgrib.open_dataset(
                        grib_file,
                        filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': grib_key},
                        backend_kwargs={'indexpath': ''},
                    )
                    if ds and len(ds.data_vars) > 0:
                        setattr(data, field_name, ds[list(ds.data_vars)[0]].values)
                    ds.close()
                except Exception:
                    pass

            try:
                ds_sp = cfgrib.open_dataset(
                    resolved_sfc,
                    filter_by_keys={'typeOfLevel': 'surface', 'shortName': 'sp'},
                    backend_kwargs={'indexpath': ''},
                )
                if ds_sp and len(ds_sp.data_vars) > 0:
                    sp_data = ds_sp[list(ds_sp.data_vars)[0]].values
                    while sp_data.ndim > 2:
                        sp_data = sp_data[0]
                    if np.isfinite(sp_data).any() and np.nanmax(sp_data) > 2000:
                        sp_data = sp_data / 100.0
                    data.surface_pressure = sp_data
                ds_sp.close()
            except Exception:
                pass

            return data

    def _load_core_eccodes() -> ForecastHourData:
        import eccodes

        target_keys = {'t', *fields.keys()}
        by_level = {k: {} for k in target_keys}
        lats = None
        lons = None

        with open(grib_file, 'rb') as f:
            while True:
                msg = eccodes.codes_grib_new_from_file(f)
                if msg is None:
                    break
                try:
                    ltype = eccodes.codes_get(msg, 'typeOfLevel')
                    if ltype != 'isobaricInhPa':
                        continue
                    short_name = eccodes.codes_get(msg, 'shortName')
                    if short_name not in target_keys:
                        continue

                    level = int(eccodes.codes_get(msg, 'level'))
                    arr2d = _decode_msg_to_2d(msg).astype(np.float32, copy=False)
                    by_level[short_name][level] = arr2d

                    if lats is None or lons is None:
                        lat_vals = np.asarray(eccodes.codes_get_array(msg, 'latitudes'), dtype=np.float32)
                        lon_vals = np.asarray(eccodes.codes_get_array(msg, 'longitudes'), dtype=np.float32)
                        lats = lat_vals.reshape(arr2d.shape)
                        lons = lon_vals.reshape(arr2d.shape)
                finally:
                    eccodes.codes_release(msg)

        t_levels = by_level.get('t', {})
        if not t_levels:
            raise RuntimeError("missing temperature (shortName='t')")

        levels_sorted = sorted(t_levels.keys(), reverse=True)
        pressure_levels = np.asarray(levels_sorted, dtype=np.float32)
        n_levels = len(levels_sorted)
        sample = t_levels[levels_sorted[0]]
        ny, nx = sample.shape

        if lats is None or lons is None:
            raise RuntimeError("missing lat/lon grid")
        if np.isfinite(lons).any() and np.nanmax(lons) > 180:
            lons = np.where(lons > 180, lons - 360, lons)

        def _stack(short_name: str) -> Optional[np.ndarray]:
            field_map = by_level.get(short_name, {})
            if not field_map:
                return None
            arr = np.full((n_levels, ny, nx), np.nan, dtype=np.float32)
            for idx, lev in enumerate(levels_sorted):
                lev_arr = field_map.get(lev)
                if lev_arr is not None:
                    arr[idx] = lev_arr
            return arr

        data = ForecastHourData(
            forecast_hour=forecast_hour,
            pressure_levels=pressure_levels,
            lats=lats,
            lons=lons,
            temperature=_stack('t'),
        )
        for grib_key, field_name in fields.items():
            arr = _stack(grib_key)
            if arr is not None:
                setattr(data, field_name, arr)

        with open(resolved_sfc, 'rb') as f:
            while True:
                msg = eccodes.codes_grib_new_from_file(f)
                if msg is None:
                    break
                try:
                    short_name = eccodes.codes_get(msg, 'shortName')
                    ltype = eccodes.codes_get(msg, 'typeOfLevel')
                    if short_name == 'sp' and ltype == 'surface':
                        sp_data = _decode_msg_to_2d(msg).astype(np.float32, copy=False)
                        if np.isfinite(sp_data).any() and np.nanmax(sp_data) > 2000:
                            sp_data = sp_data / 100.0
                        data.surface_pressure = sp_data
                        break
                finally:
                    eccodes.codes_release(msg)

        return data

    try:
        print(f"Loading F{forecast_hour:02d} from {Path(grib_file).name}...")
        start = time.perf_counter()

        fhr_data = None
        backend_used = None
        backend_errors = []

        for candidate in backend_order:
            try:
                if candidate == 'eccodes':
                    fhr_data = _load_core_eccodes()
                else:
                    fhr_data = _load_core_cfgrib()
                backend_used = candidate
                break
            except Exception as e:
                backend_errors.append((candidate, str(e)))
                print(f"  Warning: {candidate} core loader failed in worker: {e}")
                continue

        if fhr_data is None:
            detail = '; '.join(f"{name}: {err}" for name, err in backend_errors) or "no backend attempted"
            raise RuntimeError(f"All GRIB backends failed ({detail})")

        # Load smoke (MASSDEN) from wrfnat if available — keep on native hybrid levels
        try:
            if resolved_nat and Path(resolved_nat).exists():
                import eccodes

                smoke_levels = {}
                pres_levels_hyb = {}
                with open(str(resolved_nat), 'rb') as fnat:
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
                                arr2d = _decode_msg_to_2d(msg)
                                if disc == 0 and cat == 20 and num == 0:
                                    smoke_levels[lev] = arr2d
                                elif disc == 0 and cat == 3 and num == 0 and lev not in pres_levels_hyb:
                                    pres_levels_hyb[lev] = arr2d
                        except Exception:
                            pass
                        finally:
                            eccodes.codes_release(msg)

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
            scale = (1000.0 / np.asarray(fhr_data.pressure_levels, dtype=np.float32)) ** 0.286
            fhr_data.theta = fhr_data.temperature * scale[:, None, None]
            fhr_data.temp_c = fhr_data.temperature - 273.15

        duration = time.perf_counter() - start
        print(f"  Loaded F{forecast_hour:02d} in {duration:.1f}s ({fhr_data.memory_usage_mb():.0f} MB) using {backend_used}")
        return fhr_data

    except Exception as e:
        print(f"Error loading F{forecast_hour:02d}: {e}")
        return None


class InteractiveCrossSection:
    """Pre-loads HRRR data for fast interactive cross-section generation."""

    # Cached cartopy feature geometries (class-level, parsed once per process)
    _cartopy_features_cache = None

    @classmethod
    def _get_cartopy_features(cls):
        """Load and cache cartopy feature geometries once per process.

        Parsing Natural Earth shapefiles takes ~2.5s. By caching the resolved
        geometry lists, subsequent renders skip all shapefile I/O.
        """
        if cls._cartopy_features_cache is not None:
            return cls._cartopy_features_cache

        import cartopy.feature as cfeature
        cls._cartopy_features_cache = {
            'land': list(cfeature.LAND.geometries()),
            'ocean': list(cfeature.OCEAN.geometries()),
            'lakes': list(cfeature.LAKES.geometries()),
            'states': list(cfeature.STATES.geometries()),
            'borders': list(cfeature.BORDERS.geometries()),
            'coastline': list(cfeature.COASTLINE.geometries()),
        }
        return cls._cartopy_features_cache

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

    SUPPORTED_GRIB_BACKENDS = {'cfgrib', 'eccodes', 'auto'}

    def __init__(self, cache_dir: str = None, min_levels: int = 40,
                 sfc_resolver=None, nat_resolver=None, grib_backend: str = 'cfgrib'):
        """Initialize the interactive cross-section system.

        Args:
            cache_dir: Directory for NPZ cache. If provided, enables fast caching.
                      First load from GRIB takes ~25s, subsequent loads ~2s.
            min_levels: Minimum pressure levels required (40 for HRRR, 20 for GFS).
            sfc_resolver: Callable(prs_path) -> sfc_path. Resolves surface GRIB file.
            nat_resolver: Callable(prs_path) -> nat_path or None. Resolves native GRIB file.
            grib_backend: Core field loader backend: 'cfgrib' (default), 'eccodes', or
                         'auto' (try eccodes first, then fallback to cfgrib).
        """
        self.forecast_hours: Dict[int, ForecastHourData] = {}
        self._kdtree_cache = None  # Cached cKDTree for curvilinear grid interpolation
        self._kdtree_grid_id = None  # id() of the lats array used to build the tree
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Metadata for labeling
        self.model = "HRRR"
        self.init_date = None  # YYYYMMDD
        self.init_hour = None  # HH
        # Model-specific validation
        self.min_levels = min_levels
        # File resolution callbacks
        self._sfc_resolver = sfc_resolver or self._default_sfc_resolver
        self._nat_resolver = nat_resolver or self._default_nat_resolver
        self.set_grib_backend(grib_backend)

        # Climatology for anomaly mode
        self.climatology_dir = None  # Path to climo NPZ directory
        self._climo_cache: Dict[str, ClimatologyData] = {}  # "MM_HHz_FNN" -> data

    def set_grib_backend(self, grib_backend: str):
        """Set GRIB backend: cfgrib, eccodes, or auto."""
        backend = (grib_backend or 'cfgrib').strip().lower()
        if backend not in self.SUPPORTED_GRIB_BACKENDS:
            allowed = ', '.join(sorted(self.SUPPORTED_GRIB_BACKENDS))
            raise ValueError(f"Invalid grib_backend '{grib_backend}'. Expected one of: {allowed}")
        self.grib_backend = backend

    def _grib_backend_order(self) -> List[str]:
        """Backend resolution order for core GRIB loading."""
        if self.grib_backend == 'auto':
            return ['eccodes', 'cfgrib']
        return [self.grib_backend]

    @staticmethod
    def _default_sfc_resolver(prs_file: str) -> str:
        """HRRR default: replace wrfprs with wrfsfc."""
        sfc = Path(prs_file).parent / Path(prs_file).name.replace('wrfprs', 'wrfsfc')
        return str(sfc) if sfc.exists() else prs_file

    @staticmethod
    def _default_nat_resolver(prs_file: str):
        """HRRR default: replace wrfprs with wrfnat."""
        nat = Path(prs_file).parent / Path(prs_file).name.replace('wrfprs', 'wrfnat')
        return str(nat) if nat.exists() else None

    def _get_cache_stem(self, grib_file: str) -> Optional[str]:
        """Get cache name stem (without extension) for a GRIB file."""
        if not self.cache_dir:
            return None
        grib_path = Path(grib_file)
        # e.g., outputs/hrrr/20251224/19z/F00/hrrr.t19z.wrfprsf00.grib2
        # -> 20251224_19z_F00_hrrr.t19z.wrfprsf00
        parts = grib_path.parts
        try:
            date_idx = next(i for i, p in enumerate(parts) if p.isdigit() and len(p) == 8)
            return f"{parts[date_idx]}_{parts[date_idx+1]}_{parts[date_idx+2]}_{grib_path.stem}"
        except (StopIteration, IndexError):
            return grib_path.stem

    def _get_mmap_cache_dir(self, grib_file: str) -> Optional[Path]:
        """Get mmap cache directory path for a GRIB file."""
        stem = self._get_cache_stem(grib_file)
        if stem is None:
            return None
        return self.cache_dir / stem

    def _get_legacy_cache_path(self, grib_file: str) -> Optional[Path]:
        """Get legacy .npz cache path for a GRIB file (migration support)."""
        stem = self._get_cache_stem(grib_file)
        if stem is None:
            return None
        return self.cache_dir / f"{stem}.npz"

    def _cleanup_cache(self):
        """Evict oldest cache entries if cache exceeds CACHE_LIMIT_GB.

        Handles both mmap directories and legacy .npz files.
        """
        if not self.cache_dir:
            return
        try:
            import shutil
            entries = []  # (path, size_bytes, atime)

            # Legacy .npz files
            for f in self.cache_dir.glob('*.npz'):
                stat = f.stat()
                entries.append((f, stat.st_size, stat.st_atime))

            # Mmap cache directories (contain _complete marker)
            for d in self.cache_dir.iterdir():
                if d.is_dir() and (d / '_complete').exists():
                    dir_size = sum(ff.stat().st_size for ff in d.iterdir() if ff.is_file())
                    # Use _complete marker atime as directory access time
                    atime = (d / '_complete').stat().st_atime
                    entries.append((d, dir_size, atime))

            total_bytes = sum(e[1] for e in entries)
            total_gb = total_bytes / (1024 ** 3)
            if total_gb <= self.CACHE_LIMIT_GB:
                return

            target_gb = self.CACHE_LIMIT_GB * 0.85
            entries.sort(key=lambda e: e[2])  # oldest access first
            for path, size_bytes, _ in entries:
                if total_gb <= target_gb:
                    break
                size_gb = size_bytes / (1024 ** 3)
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                total_gb -= size_gb
                print(f"Cache cleanup: removed {path.name} ({size_gb:.1f}GB), {total_gb:.1f}GB remaining")
        except Exception as e:
            print(f"Cache cleanup error: {e}")

    # --- Legacy NPZ cache (for migration) ---

    def _save_to_legacy_cache(self, fhr_data: ForecastHourData, cache_path: Path):
        """Save ForecastHourData to legacy NPZ format (uncompressed)."""
        data = {'forecast_hour': np.array([fhr_data.forecast_hour])}

        for field_name in ['pressure_levels', 'lats', 'lons', 'temperature', 'u_wind', 'v_wind',
                      'rh', 'omega', 'specific_humidity', 'geopotential_height', 'vorticity',
                      'cloud', 'dew_point', 'smoke_hyb', 'smoke_pres_hyb',
                      'surface_pressure', 'theta', 'temp_c']:
            arr = getattr(fhr_data, field_name, None)
            if arr is not None:
                data[field_name] = arr

        np.savez(cache_path, **data)
        self._cleanup_cache()

    def _load_from_legacy_cache(self, cache_path: Path) -> Optional[ForecastHourData]:
        """Load ForecastHourData from legacy NPZ format."""
        try:
            data = np.load(cache_path)

            fhr_data = ForecastHourData(
                forecast_hour=int(data['forecast_hour'][0]),
                pressure_levels=data['pressure_levels'],
                lats=data['lats'],
                lons=data['lons'],
            )

            for field_name in ['temperature', 'u_wind', 'v_wind', 'rh', 'omega',
                          'specific_humidity', 'geopotential_height', 'vorticity',
                          'cloud', 'dew_point', 'smoke_hyb', 'smoke_pres_hyb',
                          'surface_pressure', 'theta', 'temp_c']:
                if field_name in data:
                    setattr(fhr_data, field_name, data[field_name])

            return fhr_data
        except Exception as e:
            print(f"Error loading from legacy cache: {e}")
            return None

    # --- Mmap cache (per-field .npy files in float16) ---

    # Fields saved as float16 (3.3 decimal digits — sufficient for visualization)
    _FLOAT16_FIELDS = {
        'temperature', 'u_wind', 'v_wind', 'rh', 'omega',
        'specific_humidity', 'vorticity', 'cloud', 'dew_point',
        'surface_pressure', 'theta', 'temp_c',
        'smoke_hyb', 'smoke_pres_hyb',
    }
    # Fields kept at float32 (need precision for derived calculations)
    _FLOAT32_FIELDS = {'geopotential_height'}
    # Coordinate fields kept at float64 (tiny, loaded into RAM)
    _COORD_FIELDS = {'pressure_levels', 'lats', 'lons'}

    def _save_to_mmap_cache(self, fhr_data: ForecastHourData, cache_dir: Path):
        """Save ForecastHourData as per-field .npy files for memory-mapped access.

        3D fields saved as float16 (~half disk size vs float64).
        geopotential_height saved as float32 (needed for shear/lapse_rate precision).
        Coordinate arrays saved as float64 (tiny, loaded into RAM).
        _complete marker written last for atomic cache creation.
        """
        import shutil

        # Write to temp directory, rename when done (atomic)
        tmp_dir = Path(str(cache_dir) + '._partial')
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        try:
            # Save forecast_hour as metadata
            np.save(tmp_dir / 'meta.npy', np.array([fhr_data.forecast_hour]))

            # Save coordinate fields (float64, tiny)
            for field_name in self._COORD_FIELDS:
                arr = getattr(fhr_data, field_name, None)
                if arr is not None:
                    np.save(tmp_dir / f'{field_name}.npy', arr)

            # Save float16 fields
            for field_name in self._FLOAT16_FIELDS:
                arr = getattr(fhr_data, field_name, None)
                if arr is not None:
                    # Read from mmap if needed before converting
                    if isinstance(arr, np.memmap):
                        arr = np.array(arr)
                    np.save(tmp_dir / f'{field_name}.npy', arr.astype(np.float16))

            # Save float32 fields
            for field_name in self._FLOAT32_FIELDS:
                arr = getattr(fhr_data, field_name, None)
                if arr is not None:
                    if isinstance(arr, np.memmap):
                        arr = np.array(arr)
                    np.save(tmp_dir / f'{field_name}.npy', arr.astype(np.float32))

            # Write _complete marker last — cache only valid if this exists
            (tmp_dir / '_complete').touch()

            # Atomic rename: remove old dir if exists, rename temp into place
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            tmp_dir.rename(cache_dir)

            self._cleanup_cache()
        except Exception as e:
            # Clean up partial write
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise e

    def _load_from_mmap_cache(self, cache_dir: Path) -> Optional[ForecastHourData]:
        """Load ForecastHourData with memory-mapped .npy files.

        Coordinate arrays (pressure_levels, lats, lons) are loaded into RAM (~30KB).
        All 3D fields are opened with mmap_mode='r' — just file handles, no data read.
        Actual data is read from NVMe on demand when cross-section slices specific levels.
        """
        try:
            if not (cache_dir / '_complete').exists():
                return None

            meta = np.load(cache_dir / 'meta.npy')
            forecast_hour = int(meta[0])

            # Coordinate arrays: load fully into RAM (tiny)
            pressure_levels = np.load(cache_dir / 'pressure_levels.npy')
            lats = np.load(cache_dir / 'lats.npy')
            lons = np.load(cache_dir / 'lons.npy')

            fhr_data = ForecastHourData(
                forecast_hour=forecast_hour,
                pressure_levels=pressure_levels,
                lats=lats,
                lons=lons,
            )

            # Memory-map all field files (no data read, just file handles)
            all_fields = self._FLOAT16_FIELDS | self._FLOAT32_FIELDS
            for field_name in all_fields:
                npy_path = cache_dir / f'{field_name}.npy'
                if npy_path.exists():
                    setattr(fhr_data, field_name, np.load(npy_path, mmap_mode='r'))

            # Touch _complete to update access time for LRU eviction
            (cache_dir / '_complete').touch()

            return fhr_data
        except Exception as e:
            print(f"Error loading from mmap cache: {e}")
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

    def _validate_fhr_data(self, fhr_data: ForecastHourData) -> Optional[str]:
        """Validate ForecastHourData. Returns error message or None if valid."""
        n_levels = len(fhr_data.pressure_levels)
        if n_levels < self.min_levels:
            return f"only {n_levels} levels (expected {self.min_levels})"
        if fhr_data.surface_pressure is None:
            return "missing surface_pressure (needed for terrain)"
        return None

    def _discard_cache(self, path: Path, reason: str):
        """Remove a stale cache entry (file or directory)."""
        import shutil
        print(f"  Cache {reason}, discarding stale cache: {path.name}")
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    def _backfill_smoke(self, fhr_data: ForecastHourData, grib_file: str, mmap_cache_dir: Optional[Path] = None):
        """Backfill smoke from native file into ForecastHourData and update cache."""
        if fhr_data.smoke_hyb is not None:
            return
        nat_path = self._nat_resolver(grib_file)
        if not nat_path or not Path(nat_path).exists():
            return
        print(f"  Backfilling smoke from native file...")
        try:
            result = self._load_smoke_from_wrfnat(nat_path)
            if result is None:
                return
            smoke_hyb, smoke_pres_hyb = result
            print(f"  Loaded PM2.5 smoke on {smoke_hyb.shape[0]} hybrid levels "
                  f"(max={np.nanmax(smoke_hyb):.1f} μg/m³)")
            # For mmap caches, write smoke .npy files directly into the cache dir
            if mmap_cache_dir and mmap_cache_dir.is_dir():
                np.save(mmap_cache_dir / 'smoke_hyb.npy', smoke_hyb.astype(np.float16))
                np.save(mmap_cache_dir / 'smoke_pres_hyb.npy', smoke_pres_hyb.astype(np.float16))
                # Re-open as mmap for consistency
                fhr_data.smoke_hyb = np.load(mmap_cache_dir / 'smoke_hyb.npy', mmap_mode='r')
                fhr_data.smoke_pres_hyb = np.load(mmap_cache_dir / 'smoke_pres_hyb.npy', mmap_mode='r')
                print(f"  Updated mmap cache with smoke data")
            else:
                fhr_data.smoke_hyb = smoke_hyb
                fhr_data.smoke_pres_hyb = smoke_pres_hyb
        except Exception as e:
            print(f"  Warning: Could not backfill smoke: {e}")

    @staticmethod
    def _grib_msg_to_2d(msg):
        """Decode one GRIB message into a 2D NumPy array."""
        import eccodes

        try:
            ni = int(eccodes.codes_get(msg, 'Ni'))
            nj = int(eccodes.codes_get(msg, 'Nj'))
        except Exception:
            ni = int(eccodes.codes_get(msg, 'Nx'))
            nj = int(eccodes.codes_get(msg, 'Ny'))

        values = eccodes.codes_get_values(msg)
        return values.reshape(nj, ni)

    def _load_core_fields_eccodes(
        self,
        grib_file: str,
        forecast_hour: int,
        cb,
        total_steps: int,
        field_labels: Dict[str, str],
    ) -> ForecastHourData:
        """Load isobaric fields + surface pressure using one-pass eccodes scans."""
        import eccodes

        cb(1, total_steps, "Reading Temperature (eccodes)...")

        target_keys = set(self.FIELDS_TO_LOAD.keys())
        fields_by_level = {k: {} for k in target_keys}  # shortName -> level -> 2D array
        lats = None
        lons = None
        scanned = 0
        matched = 0

        with open(grib_file, 'rb') as f:
            while True:
                try:
                    msg = eccodes.codes_grib_new_from_file(f)
                except Exception:
                    break  # Truncated trailing message — treat as EOF
                if msg is None:
                    break
                scanned += 1
                try:
                    ltype = eccodes.codes_get(msg, 'typeOfLevel')
                    if ltype != 'isobaricInhPa':
                        continue

                    short_name = eccodes.codes_get(msg, 'shortName')
                    if short_name not in target_keys:
                        continue

                    level = int(eccodes.codes_get(msg, 'level'))
                    arr2d = self._grib_msg_to_2d(msg).astype(np.float32, copy=False)
                    fields_by_level[short_name][level] = arr2d
                    matched += 1

                    if lats is None or lons is None:
                        lat_vals = np.asarray(eccodes.codes_get_array(msg, 'latitudes'), dtype=np.float32)
                        lon_vals = np.asarray(eccodes.codes_get_array(msg, 'longitudes'), dtype=np.float32)
                        lats = lat_vals.reshape(arr2d.shape)
                        lons = lon_vals.reshape(arr2d.shape)
                finally:
                    eccodes.codes_release(msg)

        temp_by_level = fields_by_level.get('t', {})
        if not temp_by_level:
            raise RuntimeError("eccodes loader missing temperature (shortName='t')")

        levels_sorted = sorted(temp_by_level.keys(), reverse=True)
        pressure_levels = np.asarray(levels_sorted, dtype=np.float32)
        n_levels = len(levels_sorted)
        sample = temp_by_level[levels_sorted[0]]
        ny, nx = sample.shape

        if lats is None or lons is None:
            raise RuntimeError("eccodes loader missing grid coordinates")

        if np.isfinite(lons).any() and np.nanmax(lons) > 180:
            lons = np.where(lons > 180, lons - 360, lons)

        def stack_field(short_name: str) -> Optional[np.ndarray]:
            level_map = fields_by_level.get(short_name, {})
            if not level_map:
                return None
            stacked = np.full((n_levels, ny, nx), np.nan, dtype=np.float32)
            missing = 0
            for idx, lev in enumerate(levels_sorted):
                lev_arr = level_map.get(lev)
                if lev_arr is None:
                    missing += 1
                    continue
                stacked[idx] = lev_arr
            if short_name == 't' and missing > 0:
                raise RuntimeError(f"eccodes temperature missing {missing}/{n_levels} pressure levels")
            if missing > 0:
                print(f"  Warning: eccodes missing {short_name} on {missing}/{n_levels} pressure levels")
            return stacked

        fhr_data = ForecastHourData(
            forecast_hour=forecast_hour,
            pressure_levels=pressure_levels,
            lats=lats,
            lons=lons,
            temperature=stack_field('t'),
        )

        step = 2
        for grib_key, field_name in self.FIELDS_TO_LOAD.items():
            if grib_key == 't':
                continue
            cb(step, total_steps, f"Reading {field_labels.get(grib_key, field_name)} (eccodes)...")
            arr = stack_field(grib_key)
            if arr is not None:
                setattr(fhr_data, field_name, arr)
            step += 1

        cb(11, total_steps, "Reading Surface Pressure...")
        sp_file = self._sfc_resolver(grib_file)
        surface_pressure = None
        sp_scanned = 0
        with open(sp_file, 'rb') as f:
            while True:
                try:
                    msg = eccodes.codes_grib_new_from_file(f)
                except Exception:
                    break  # Truncated trailing message — treat as EOF
                if msg is None:
                    break
                sp_scanned += 1
                try:
                    short_name = eccodes.codes_get(msg, 'shortName')
                    ltype = eccodes.codes_get(msg, 'typeOfLevel')
                    if short_name == 'sp' and ltype == 'surface':
                        surface_pressure = self._grib_msg_to_2d(msg).astype(np.float32, copy=False)
                        break
                finally:
                    eccodes.codes_release(msg)

        if surface_pressure is not None:
            if np.isfinite(surface_pressure).any() and np.nanmax(surface_pressure) > 2000:
                surface_pressure = surface_pressure / 100.0
            fhr_data.surface_pressure = surface_pressure
        else:
            print("  Warning: Could not load surface pressure via eccodes")

        print(f"  eccodes core scan: prs_msgs={scanned}, matched={matched}, sfc_msgs={sp_scanned}")
        return fhr_data

    def _load_core_fields_cfgrib(
        self,
        grib_file: str,
        forecast_hour: int,
        cb,
        total_steps: int,
        field_labels: Dict[str, str],
    ) -> ForecastHourData:
        """Load isobaric fields + surface pressure via cfgrib (legacy path)."""
        import cfgrib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cb(1, total_steps, "Reading Temperature...")
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

            step = 2
            for grib_key, field_name in self.FIELDS_TO_LOAD.items():
                if grib_key == 't':
                    continue

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

            cb(11, total_steps, "Reading Surface Pressure...")
            try:
                sp_file = self._sfc_resolver(grib_file)

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
            except Exception as e:
                print(f"  Warning: Could not load surface pressure: {e}")

        return fhr_data

    def load_forecast_hour(self, grib_file: str, forecast_hour: int, progress_callback=None) -> bool:
        """Load all fields for a forecast hour.

        Load order: mmap cache dir → legacy .npz cache → GRIB file.
        New saves always go to mmap format. Legacy .npz caches work indefinitely
        but are not updated — they convert to mmap when the FHR is reloaded from GRIB.

        Args:
            grib_file: Path to wrfprs GRIB2 file
            forecast_hour: Forecast hour number
            progress_callback: Optional callback(step, total, detail) for progress reporting

        Returns:
            True if successful
        """
        cb = progress_callback or (lambda s, t, d: None)

        # --- Try mmap cache first ---
        mmap_dir = self._get_mmap_cache_dir(grib_file)
        if mmap_dir and mmap_dir.is_dir():
            cb(1, 2, "Loading from mmap cache...")
            print(f"Loading F{forecast_hour:02d} from mmap cache...")
            start = time.perf_counter()
            fhr_data = self._load_from_mmap_cache(mmap_dir)
            if fhr_data is not None:
                err = self._validate_fhr_data(fhr_data)
                if err:
                    self._discard_cache(mmap_dir, err)
                    fhr_data = None
            if fhr_data is not None:
                fhr_data.grib_file = grib_file
                self.forecast_hours[forecast_hour] = fhr_data
                duration = time.perf_counter() - start
                print(f"  Loaded F{forecast_hour:02d} from mmap cache in {duration:.3f}s ({fhr_data.memory_usage_mb():.0f} MB heap)")
                cb(2, 2, "Done")
                return True

        # --- Fallback: legacy .npz cache ---
        legacy_path = self._get_legacy_cache_path(grib_file)
        if legacy_path and legacy_path.exists():
            cb(1, 2, "Loading from cache...")
            print(f"Loading F{forecast_hour:02d} from legacy cache...")
            start = time.perf_counter()
            fhr_data = self._load_from_legacy_cache(legacy_path)
            if fhr_data is not None:
                err = self._validate_fhr_data(fhr_data)
                if err:
                    self._discard_cache(legacy_path, err)
                    fhr_data = None
            if fhr_data is not None:
                # Migrate legacy cache to mmap format
                if mmap_dir:
                    try:
                        print(f"  Migrating to mmap cache...")
                        self._save_to_mmap_cache(fhr_data, mmap_dir)
                        # Re-load as mmap to get memory-mapped arrays
                        fhr_data_mmap = self._load_from_mmap_cache(mmap_dir)
                        if fhr_data_mmap is not None:
                            fhr_data = fhr_data_mmap
                            # Remove legacy .npz
                            legacy_path.unlink(missing_ok=True)
                            print(f"  Migrated to mmap, removed legacy .npz")
                    except Exception as e:
                        print(f"  Warning: Could not migrate to mmap: {e}")
                fhr_data.grib_file = grib_file
                self.forecast_hours[forecast_hour] = fhr_data
                duration = time.perf_counter() - start
                print(f"  Loaded F{forecast_hour:02d} from cache in {duration:.1f}s ({fhr_data.memory_usage_mb():.0f} MB)")
                cb(2, 2, "Done")
                return True

        # --- Load from GRIB ---
        field_labels = {
            't': 'Temperature', 'u': 'U-Wind', 'v': 'V-Wind', 'r': 'RH',
            'w': 'Omega', 'q': 'Sp. Humidity', 'gh': 'Geopotential',
            'absv': 'Vorticity', 'clwmr': 'Cloud Water', 'dpt': 'Dew Point',
        }
        total_steps = 13  # 10 fields + surface pressure + smoke + derived

        try:
            print(f"Loading F{forecast_hour:02d} from {Path(grib_file).name}...")
            start = time.perf_counter()
            fhr_data = None
            backend_used = None
            backend_errors = []

            for backend in self._grib_backend_order():
                try:
                    if backend == 'eccodes':
                        fhr_data = self._load_core_fields_eccodes(
                            grib_file=grib_file,
                            forecast_hour=forecast_hour,
                            cb=cb,
                            total_steps=total_steps,
                            field_labels=field_labels,
                        )
                    else:
                        fhr_data = self._load_core_fields_cfgrib(
                            grib_file=grib_file,
                            forecast_hour=forecast_hour,
                            cb=cb,
                            total_steps=total_steps,
                            field_labels=field_labels,
                        )
                    backend_used = backend
                    break
                except Exception as e:
                    backend_errors.append((backend, str(e)))
                    print(f"  Warning: {backend} core loader failed: {e}")
                    continue

            if fhr_data is None:
                detail = '; '.join(f"{name}: {err}" for name, err in backend_errors) or "no backend attempted"
                raise RuntimeError(f"All GRIB backends failed ({detail})")

            # Load smoke (MASSDEN PM2.5) from native file if available
            cb(12, total_steps, "Reading Smoke (PM2.5)...")
            try:
                nat_path = self._nat_resolver(grib_file)
                if nat_path and Path(nat_path).exists():
                    nat_file = Path(nat_path)
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
                scale = (1000.0 / np.asarray(fhr_data.pressure_levels, dtype=np.float32)) ** 0.286
                fhr_data.theta = fhr_data.temperature * scale[:, None, None]
                fhr_data.temp_c = fhr_data.temperature - 273.15

            # Validate before storing — don't cache incomplete data
            err = self._validate_fhr_data(fhr_data)
            if err:
                print(f"  WARNING: {err} — GRIB may be incomplete, skipping")
                return False

            # Store
            fhr_data.grib_file = grib_file
            self.forecast_hours[forecast_hour] = fhr_data

            duration = time.perf_counter() - start
            mem_mb = fhr_data.memory_usage_mb()
            print(f"  Loaded F{forecast_hour:02d} in {duration:.1f}s ({mem_mb:.0f} MB) using {backend_used}")

            # Save to mmap cache, then reload as mmap to free RAM
            if mmap_dir:
                try:
                    self._save_to_mmap_cache(fhr_data, mmap_dir)
                    fhr_mmap = self._load_from_mmap_cache(mmap_dir)
                    if fhr_mmap is not None:
                        fhr_mmap.grib_file = grib_file
                        self.forecast_hours[forecast_hour] = fhr_mmap
                        print(f"  Cached to {mmap_dir.name}/ (mmap, {fhr_mmap.memory_usage_mb():.0f} MB heap)")
                    else:
                        print(f"  Cached to {mmap_dir.name}/ (mmap)")
                except Exception as e:
                    print(f"  Warning: Could not cache: {e}")

            return True

        except Exception as e:
            print(f"Error loading F{forecast_hour:02d}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_run(self, run_dir: str, max_hours: int = 18, workers: int = 1,
                 prs_pattern: str = "*wrfprs*.grib2") -> int:
        """Load all forecast hours from a run directory.

        Args:
            run_dir: Path to run directory (e.g., outputs/hrrr/20251224/19z)
            prs_pattern: Glob pattern for pressure GRIB files (model-specific)
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
            prs_files = list(fhr_dir.glob(prs_pattern))
            if prs_files:
                files_to_load.append((str(prs_files[0]), fhr))

        if not files_to_load:
            print(f"No GRIB files found matching {prs_pattern}")
            return 0

        print(f"Loading {len(files_to_load)} forecast hours with {workers} workers...")
        start_time = time.perf_counter()

        if workers <= 1:
            # Sequential loading
            for grib_file, fhr in files_to_load:
                self.load_forecast_hour(grib_file, fhr)
        else:
            # Parallel loading with multiprocessing (bypasses GIL)
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _load_hour_process,
                        grib_file,
                        fhr,
                        self.grib_backend,
                        self._sfc_resolver(grib_file),
                        self._nat_resolver(grib_file),
                    ): fhr
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

        duration = time.perf_counter() - start_time
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
            start = time.perf_counter()

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
                    sp_file = self._sfc_resolver(grib_file)

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
                    scale = (1000.0 / np.asarray(pressure_levels, dtype=np.float32)) ** 0.286
                    fhr_data.theta = fhr_data.temperature * scale[:, None, None]
                    fhr_data.temp_c = fhr_data.temperature - 273.15

                duration = time.perf_counter() - start
                mem_mb = fhr_data.memory_usage_mb()
                print(f"  Loaded F{forecast_hour:02d} in {duration:.1f}s ({mem_mb:.0f} MB)")

                return fhr_data

        except Exception as e:
            print(f"Error loading F{forecast_hour:02d}: {e}")
            return None

    # ── Climatology methods for anomaly mode ──

    def set_climatology_dir(self, path: str):
        """Set directory containing climatology NPZ files."""
        self.climatology_dir = Path(path) if path else None
        self._climo_cache.clear()

    def has_climatology(self, month: int, init_hour: int) -> bool:
        """Check if any climatology exists for this month/init."""
        if not self.climatology_dir or not self.climatology_dir.exists():
            return False
        return bool(list(self.climatology_dir.glob(
            f"climo_{month:02d}_{init_hour:02d}z_F*.npz"
        )))

    def get_climatology(self, month: int, init_hour: int, fhr: int) -> Optional[ClimatologyData]:
        """Load climatology for given month/init/fhr. Caches in RAM."""
        if not self.climatology_dir:
            return None

        key = f"{month:02d}_{init_hour:02d}z_F{fhr:02d}"
        if key in self._climo_cache:
            return self._climo_cache[key]

        climo_path = self.climatology_dir / f"climo_{key}.npz"
        if not climo_path.exists():
            # Fall back to nearest available FHR
            candidates = sorted(self.climatology_dir.glob(
                f"climo_{month:02d}_{init_hour:02d}z_F*.npz"
            ))
            if not candidates:
                return None
            best = min(candidates,
                       key=lambda p: abs(int(p.stem.split('_F')[1]) - fhr))
            climo_path = best
            key = best.stem.replace('climo_', '')
            if key in self._climo_cache:
                return self._climo_cache[key]

        try:
            data = np.load(climo_path)
            climo = ClimatologyData(
                month=month,
                init_hour=init_hour,
                fhr=fhr,
                pressure_levels=data['pressure_levels'],
                lats=data['lats'],
                lons=data['lons'],
                n_samples=int(data['n_samples'][0]),
                years=data.get('years', np.array([])).tolist(),
            )
            for field_name in ('temperature', 'u_wind', 'v_wind', 'rh', 'omega',
                               'specific_humidity', 'geopotential_height', 'vorticity'):
                if field_name in data:
                    setattr(climo, field_name, data[field_name])

            self._climo_cache[key] = climo
            return climo
        except Exception as e:
            print(f"Failed to load climatology {climo_path.name}: {e}")
            return None

    def _interpolate_climatology_to_path(
        self,
        climo: ClimatologyData,
        path_lats: np.ndarray,
        path_lons: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Interpolate coarsened climatology grids to cross-section path.

        Returns dict of field_name -> (n_levels, n_points) arrays.
        """
        from scipy.interpolate import RegularGridInterpolator

        n_points = len(path_lats)
        pts = np.column_stack([path_lats, path_lons])

        climo_lats = climo.lats if climo.lats.ndim == 1 else climo.lats[:, 0]
        climo_lons = climo.lons if climo.lons.ndim == 1 else climo.lons[0, :]

        # Ensure lats are ascending for RegularGridInterpolator
        lat_ascending = climo_lats[0] < climo_lats[-1]

        result = {}
        for field_name in ('temperature', 'u_wind', 'v_wind', 'rh', 'omega',
                           'specific_humidity', 'geopotential_height', 'vorticity'):
            field_3d = getattr(climo, field_name, None)
            if field_3d is None:
                continue
            n_levels = field_3d.shape[0]
            interp_result = np.full((n_levels, n_points), np.nan)
            for lev in range(n_levels):
                lev_data = field_3d[lev]
                lat_coords = climo_lats
                if not lat_ascending:
                    lev_data = lev_data[::-1, :]
                    lat_coords = climo_lats[::-1]
                try:
                    interp = RegularGridInterpolator(
                        (lat_coords, climo_lons), lev_data,
                        method='linear', bounds_error=False, fill_value=np.nan
                    )
                    interp_result[lev, :] = interp(pts)
                except Exception:
                    pass
            result[field_name] = interp_result

        return result

    def _apply_anomaly(self, data: Dict, climo_path: Dict, style: str) -> Dict:
        """Subtract climatology from forecast data to produce anomalies.

        Stores result in data['anomaly']. Raw fields are NOT modified,
        so theta contours, freezing level, wind barbs, terrain stay absolute.
        """
        if style not in ANOMALY_STYLES:
            return data

        pressure_levels = data.get('pressure_levels')

        if style == 'temp':
            if 'temp_c' in data and 'temperature' in climo_path:
                climo_c = climo_path['temperature'] - 273.15
                data['anomaly'] = data['temp_c'] - climo_c

        elif style == 'wind_speed':
            if 'u_wind' in data and 'v_wind' in data:
                fcst_wspd = np.sqrt(data['u_wind']**2 + data['v_wind']**2) * 1.944
                if 'u_wind' in climo_path and 'v_wind' in climo_path:
                    climo_wspd = np.sqrt(climo_path['u_wind']**2 + climo_path['v_wind']**2) * 1.944
                    data['anomaly'] = fcst_wspd - climo_wspd

        elif style == 'rh':
            if 'rh' in data and 'rh' in climo_path:
                data['anomaly'] = data['rh'] - climo_path['rh']

        elif style == 'omega':
            if 'omega' in data and 'omega' in climo_path:
                data['anomaly'] = (data['omega'] - climo_path['omega']) * 36.0

        elif style == 'theta_e':
            if 'theta_e' in data and 'temperature' in climo_path and 'specific_humidity' in climo_path:
                climo_T = climo_path['temperature']
                climo_q = climo_path['specific_humidity']
                if pressure_levels is not None:
                    climo_theta = np.zeros_like(climo_T)
                    for lev_idx, p in enumerate(pressure_levels):
                        if lev_idx < climo_T.shape[0]:
                            climo_theta[lev_idx] = climo_T[lev_idx] * (1000.0 / p) ** 0.286
                    Lv, cp = 2.5e6, 1004.0
                    with np.errstate(invalid='ignore', divide='ignore'):
                        climo_theta_e = climo_theta * np.exp(Lv * climo_q / (cp * climo_T))
                    data['anomaly'] = data['theta_e'] - climo_theta_e

        elif style == 'q':
            if 'specific_humidity' in data and 'specific_humidity' in climo_path:
                data['anomaly'] = (data['specific_humidity'] - climo_path['specific_humidity']) * 1000

        elif style == 'vorticity':
            if 'vorticity' in data and 'vorticity' in climo_path:
                data['anomaly'] = (data['vorticity'] - climo_path['vorticity']) * 1e5

        elif style == 'shear':
            if 'shear' in data and 'u_wind' in climo_path and 'v_wind' in climo_path and 'geopotential_height' in climo_path:
                gh_c = climo_path['geopotential_height']
                u_c, v_c = climo_path['u_wind'], climo_path['v_wind']
                n_levels = gh_c.shape[0]
                climo_shear = np.full_like(gh_c, np.nan)
                for lev in range(n_levels - 1):
                    dz = gh_c[lev, :] - gh_c[lev + 1, :]
                    dz = np.where(np.abs(dz) < 10, np.nan, dz)
                    du = u_c[lev, :] - u_c[lev + 1, :]
                    dv = v_c[lev, :] - v_c[lev + 1, :]
                    dwind = np.sqrt(du**2 + dv**2)
                    climo_shear[lev, :] = (dwind / np.abs(dz)) * 1000
                climo_shear[-1, :] = climo_shear[-2, :] if n_levels > 1 else 0
                data['anomaly'] = data['shear'] - climo_shear

        elif style == 'lapse_rate':
            if 'lapse_rate' in data and 'temperature' in climo_path and 'geopotential_height' in climo_path:
                T_c = climo_path['temperature']
                gh_c = climo_path['geopotential_height']
                n_levels = T_c.shape[0]
                climo_lapse = np.full_like(T_c, np.nan)
                for lev in range(n_levels - 1):
                    dz = (gh_c[lev, :] - gh_c[lev + 1, :]) / 1000.0
                    dz = np.where(np.abs(dz) < 0.01, np.nan, dz)
                    dT = T_c[lev, :] - T_c[lev + 1, :]
                    climo_lapse[lev, :] = -dT / dz
                climo_lapse[-1, :] = climo_lapse[-2, :] if n_levels > 1 else 0
                data['anomaly'] = data['lapse_rate'] - climo_lapse

        elif style == 'wetbulb':
            if 'wetbulb' in data and 'temperature' in climo_path and 'rh' in climo_path:
                climo_tc = climo_path['temperature'] - 273.15
                climo_rh = climo_path['rh']
                climo_tw = (climo_tc * np.arctan(0.151977 * np.sqrt(climo_rh + 8.313659))
                           + np.arctan(climo_tc + climo_rh)
                           - np.arctan(climo_rh - 1.676331)
                           + 0.00391838 * (climo_rh ** 1.5) * np.arctan(0.023101 * climo_rh)
                           - 4.686035)
                data['anomaly'] = data['wetbulb'] - climo_tw

        elif style == 'vpd':
            if 'vpd' in data and 'temperature' in climo_path and 'rh' in climo_path:
                climo_tc = climo_path['temperature'] - 273.15
                climo_rh = climo_path['rh']
                climo_es = 6.1078 * np.exp(17.27 * climo_tc / (climo_tc + 237.3))
                climo_vpd = climo_es * (1.0 - climo_rh / 100.0)
                data['anomaly'] = data['vpd'] - climo_vpd

        elif style == 'dewpoint_dep':
            if 'dewpoint_dep' in data and 'temperature' in climo_path and 'dew_point' in climo_path:
                climo_tc = climo_path['temperature'] - 273.15
                climo_td_c = climo_path['dew_point'] - 273.15
                climo_dd = climo_tc - climo_td_c
                data['anomaly'] = data['dewpoint_dep'] - climo_dd

        elif style == 'moisture_transport':
            if 'moisture_transport' in data and 'specific_humidity' in climo_path and 'u_wind' in climo_path and 'v_wind' in climo_path:
                climo_q = climo_path['specific_humidity']
                climo_u = climo_path['u_wind']
                climo_v = climo_path['v_wind']
                climo_ws = np.sqrt(climo_u**2 + climo_v**2)
                climo_mt = climo_q * 1000.0 * climo_ws
                data['anomaly'] = data['moisture_transport'] - climo_mt

        return data

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
        temp_cmap: str = "standard",
        metadata: Dict = None,
        anomaly: bool = False,
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
            anomaly: If True, subtract climatological mean and use diverging colormap

        Returns:
            PNG image bytes, or data dict if return_image=False
        """
        if forecast_hour not in self.forecast_hours:
            print(f"Forecast hour {forecast_hour} not loaded")
            return None

        start = time.perf_counter()

        # Get pre-loaded data
        fhr_data = self.forecast_hours[forecast_hour]

        # Create path
        path_lats = np.linspace(start_point[0], end_point[0], n_points)
        path_lons = np.linspace(start_point[1], end_point[1], n_points)

        # Interpolate all needed fields to path
        data = self._interpolate_to_path(fhr_data, path_lats, path_lons, style)

        t_interp = time.perf_counter() - start

        if not return_image:
            return data

        # Override terrain for consistent GIF frames
        ref_pressure_levels = None
        if terrain_data is not None:
            for key in ('surface_pressure', 'surface_pressure_hires', 'distances_hires'):
                if key in terrain_data:
                    data[key] = terrain_data[key]
            # Lock y-axis range to first frame's pressure levels so axis doesn't shift between GIF frames
            ref_pressure_levels = terrain_data.get('pressure_levels')

        # Build metadata for labels
        if metadata is None:
            metadata = {
                'model': self.model,
                'init_date': self.init_date,
                'init_hour': self.init_hour,
                'forecast_hour': fhr_data.forecast_hour,
            }

        # Anomaly mode: subtract climatological mean
        climo_info = None
        if anomaly and style in ANOMALY_STYLES:
            init_date = metadata.get('init_date', self.init_date)
            init_hr = metadata.get('init_hour', self.init_hour)
            if init_date and init_hr is not None:
                month = int(str(init_date)[4:6])
                real_fhr = metadata.get('forecast_hour', forecast_hour)
                climo = self.get_climatology(month, int(init_hr), real_fhr)
                if climo is not None:
                    climo_along_path = self._interpolate_climatology_to_path(
                        climo, path_lats, path_lons
                    )
                    self._apply_anomaly(data, climo_along_path, style)
                    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                                   'Jul','Aug','Sep','Oct','Nov','Dec']
                    climo_info = {
                        'month': month,
                        'month_name': month_names[month - 1],
                        'n_samples': climo.n_samples,
                        'years': climo.years,
                    }

        # Render
        img_bytes = self._render_cross_section(data, style, dpi, metadata, y_axis, vscale, y_top, units=units, temp_cmap=temp_cmap, ref_pressure_levels=ref_pressure_levels, anomaly=anomaly, climo_info=climo_info)

        t_total = time.perf_counter() - start
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

        def _ensure_float32(arr):
            """Cast float16/memmap to float32 for scipy interpolation."""
            if arr.dtype == np.float16:
                return np.array(arr, dtype=np.float32)
            if isinstance(arr, np.memmap):
                return np.array(arr)  # read from disk into RAM
            return arr

        # Build interpolator (curvilinear vs regular grid)
        if lats_grid.ndim == 2:
            # Curvilinear grid - use KDTree (cached per grid)
            grid_id = id(lats_grid)
            if self._kdtree_cache is not None and self._kdtree_grid_id == grid_id:
                tree = self._kdtree_cache
            else:
                src_pts = np.column_stack([lats_grid.ravel(), lons_grid.ravel()])
                tree = cKDTree(src_pts)
                self._kdtree_cache = tree
                self._kdtree_grid_id = grid_id
            tgt_pts = np.column_stack([path_lats, path_lons])
            _, indices = tree.query(tgt_pts, k=1)

            def interp_3d(field_3d):
                result = np.full((n_levels, n_points), np.nan)
                for lev in range(min(field_3d.shape[0], n_levels)):
                    level_data = _ensure_float32(field_3d[lev])
                    result[lev, :] = level_data.ravel()[indices]
                return result

            def interp_2d(field_2d):
                return _ensure_float32(field_2d).ravel()[indices]
        else:
            # Regular grid - use bilinear interpolation
            lats_1d = lats_grid if lats_grid.ndim == 1 else lats_grid[:, 0]
            lons_1d = lons_grid if lons_grid.ndim == 1 else lons_grid[0, :]

            # Handle GFS-style 0-360 longitudes: convert to -180..180
            _lon_shifted = False
            if np.any(lons_1d > 180):
                lons_1d = np.where(lons_1d > 180, lons_1d - 360, lons_1d)
                _lon_shifted = True

            # Ensure monotonically ascending for both axes
            _lat_flip = False
            if lats_1d.size > 1 and lats_1d[0] > lats_1d[-1]:
                lats_1d = lats_1d[::-1]
                _lat_flip = True

            _lon_sort_idx = None
            if lons_1d.size > 1 and not np.all(np.diff(lons_1d) > 0):
                _lon_sort_idx = np.argsort(lons_1d)
                lons_1d = lons_1d[_lon_sort_idx]

            pts = np.column_stack([path_lats, path_lons])

            def _reorder_field(field):
                """Flip/sort a 2D field to match the reordered lat/lon axes."""
                f = field
                if _lat_flip:
                    f = f[::-1, :]
                if _lon_sort_idx is not None:
                    f = f[:, _lon_sort_idx]
                return f

            def interp_3d(field_3d):
                result = np.full((n_levels, n_points), np.nan)
                for lev in range(min(field_3d.shape[0], n_levels)):
                    level_data = _reorder_field(_ensure_float32(field_3d[lev]))
                    interp = RegularGridInterpolator(
                        (lats_1d, lons_1d), level_data,
                        method='linear', bounds_error=False, fill_value=np.nan
                    )
                    result[lev, :] = interp(pts)
                return result

            def interp_2d(field_2d):
                level_data = _reorder_field(_ensure_float32(field_2d))
                interp = RegularGridInterpolator(
                    (lats_1d, lons_1d), level_data,
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

            sp_f32 = _ensure_float32(fhr_data.surface_pressure)
            if lats_grid.ndim == 2:
                # Curvilinear - use same tree
                tgt_pts_hires = np.column_stack([path_lats_hires, path_lons_hires])
                _, indices_hires = tree.query(tgt_pts_hires, k=1)
                sp_hires = sp_f32.ravel()[indices_hires]
            else:
                # Regular grid - bilinear interpolation
                pts_hires = np.column_stack([path_lats_hires, path_lons_hires])
                interp_sp = RegularGridInterpolator(
                    (lats_1d, lons_1d), _reorder_field(sp_f32),
                    method='linear', bounds_error=False, fill_value=np.nan
                )
                sp_hires = interp_sp(pts_hires)

            result['surface_pressure_hires'] = sp_hires
            result['distances_hires'] = self._calculate_distances(path_lats_hires, path_lons_hires)

        # Style-specific fields
        if style in ['rh', 'q'] and fhr_data.rh is not None:
            result['rh'] = interp_3d(fhr_data.rh)

        if style == 'smoke':
            # Lazy smoke backfill: load from wrfnat on first smoke request
            if fhr_data.smoke_hyb is None and fhr_data.grib_file:
                mmap_dir = self._get_mmap_cache_dir(fhr_data.grib_file)
                self._backfill_smoke(fhr_data, fhr_data.grib_file,
                                     mmap_cache_dir=mmap_dir if mmap_dir and mmap_dir.is_dir() else None)
            if fhr_data.smoke_hyb is not None:
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

        if style == 'vpd' and fhr_data.rh is not None:
            RH = interp_3d(fhr_data.rh)
            result['rh'] = RH
            T_c = result['temp_c']
            # Tetens formula for saturation vapor pressure (hPa)
            es = 6.1078 * np.exp(17.27 * T_c / (T_c + 237.3))
            result['vpd'] = es * (1.0 - RH / 100.0)

        if style == 'dewpoint_dep' and fhr_data.dew_point is not None:
            td = interp_3d(fhr_data.dew_point)
            td_c = td - 273.15
            result['dewpoint_dep'] = result['temp_c'] - td_c

        if style == 'moisture_transport' and fhr_data.specific_humidity is not None:
            q = interp_3d(fhr_data.specific_humidity)
            result['specific_humidity'] = q
            u = result.get('u_wind')
            v = result.get('v_wind')
            if u is not None and v is not None:
                wind_speed = np.sqrt(u**2 + v**2)
                result['moisture_transport'] = q * 1000.0 * wind_speed  # g/kg * m/s

        if style == 'pv' and fhr_data.vorticity is not None:
            vort = interp_3d(fhr_data.vorticity)
            result['vorticity'] = vort
            theta = result['theta']
            p_levels = fhr_data.pressure_levels  # hPa
            n_lev = len(p_levels)
            # Centered finite difference for dθ/dp
            dtheta_dp = np.zeros_like(theta)
            for lev in range(1, n_lev - 1):
                dp = (p_levels[lev + 1] - p_levels[lev - 1]) * 100.0  # hPa → Pa
                dtheta_dp[lev, :] = (theta[lev + 1, :] - theta[lev - 1, :]) / dp
            dtheta_dp[0, :] = dtheta_dp[1, :]
            dtheta_dp[-1, :] = dtheta_dp[-2, :]
            g = 9.81
            pv = -g * vort * dtheta_dp  # K m² kg⁻¹ s⁻¹
            result['pv'] = pv * 1e6  # Convert to PVU

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
    def _build_temp_colormap(name: str = "standard"):
        """Build a temperature colormap by name.

        Options:
            standard:     Indigo -> blue -> cyan -> teal -> pale green (0°C) -> yellow -> orange -> red -> maroon
            white_zero:   White (0°C) with purples below and warm colors above
            nws_ndfd:     Classic NWS blue -> cyan -> yellow (0°C) -> orange -> red
            green_purple: Legacy green (0°C) -> yellow -> orange -> red -> purple
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
        elif name == "green_purple":
            # Legacy: blue-teal below freezing, green at 0°C, warm above
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
        else:
            # standard (default): smooth met colormap
            # Deep indigo (tropopause) -> blue -> cyan -> teal (0°C) ->
            # green -> yellow -> orange -> red -> maroon
            # Freezing sits in cool teal; warm colors start above ~10°C
            return from_f_anchors([
                (-80, ( 15,   0,  60)),  # Near-black indigo (tropopause)
                (-65, ( 30,   5, 110)),  # Deep indigo
                (-50, ( 45,  20, 170)),  # Dark violet-blue
                (-35, ( 20,  55, 210)),  # Royal blue
                (-20, ( 15,  95, 235)),  # Bright blue
                ( -5, ( 25, 150, 250)),  # Azure
                ( 12, ( 40, 190, 235)),  # Sky blue
                ( 24, ( 60, 210, 210)),  # Cyan-teal
                ( 32, ( 80, 220, 190)),  # Teal (freezing — clearly cool)
                ( 42, (110, 210, 140)),  # Cool green
                ( 52, (170, 215,  80)),  # Yellow-green
                ( 62, (230, 210,  40)),  # Yellow (warm — above ~17°C)
                ( 72, (255, 175,  20)),  # Gold-orange
                ( 82, (255, 130,  10)),  # Orange
                ( 92, (240,  75,  10)),  # Red-orange
                (100, (215,  30,  15)),  # True red
                (108, (170,  10,  25)),  # Crimson
                (118, (115,   5,  35)),  # Dark red
                (125, ( 70,   0,  40)),  # Deep maroon
            ])

    def _render_cross_section(self, data: Dict, style: str, dpi: int, metadata: Dict = None,
                               y_axis: str = "pressure", vscale: float = 1.0, y_top: int = 100,
                               units: str = "km", temp_cmap: str = "standard",
                               ref_pressure_levels: np.ndarray = None,
                               anomaly: bool = False, climo_info: Dict = None) -> bytes:
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
                    'icing', 'wetbulb', 'lapse_rate',
                    'vpd', 'dewpoint_dep', 'moisture_transport', 'pv']:
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

        # Anomaly labels per style
        _ANOMALY_LABELS = {
            'temp': ('Temperature Anomaly (°C)', 'Temperature Anomaly'),
            'wind_speed': ('Wind Speed Anomaly (kts)', 'Wind Speed Anomaly'),
            'rh': ('RH Anomaly (%)', 'RH Anomaly'),
            'omega': ('Omega Anomaly (μb/s)', 'Omega Anomaly'),
            'theta_e': ('θe Anomaly (K)', 'θe Anomaly'),
            'q': ('Specific Humidity Anomaly (g/kg)', 'q Anomaly'),
            'vorticity': ('Vorticity Anomaly (×10⁻⁵ s⁻¹)', 'Vorticity Anomaly'),
            'shear': ('Shear Anomaly (kt/kft)', 'Shear Anomaly'),
            'lapse_rate': ('Lapse Rate Anomaly (°C/km)', 'Lapse Rate Anomaly'),
            'wetbulb': ('Wet Bulb Anomaly (°C)', 'Wet Bulb Anomaly'),
        }

        if anomaly and 'anomaly' in data:
            # Anomaly mode: diverging colormap centered at 0
            anomaly_field = data['anomaly']

            # Symmetric auto-scaling from 98th percentile of |anomaly|
            finite_vals = anomaly_field[np.isfinite(anomaly_field)]
            if len(finite_vals) > 0:
                vmax = np.percentile(np.abs(finite_vals), 98)
                vmax = max(vmax, 0.1)  # Floor to avoid degenerate range
            else:
                vmax = 1.0

            # Build symmetric levels centered at 0
            n_levels_anom = 20
            levels_anom = np.linspace(-vmax, vmax, n_levels_anom * 2 + 1)

            cf = ax.contourf(X, Y, anomaly_field, levels=levels_anom,
                            cmap='RdBu_r', extend='both')
            cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
            cbar = plt.colorbar(cf, cax=cbar_ax)

            cbar_label, shading_label = _ANOMALY_LABELS.get(
                style, (f'{style} Anomaly', f'{style} Anomaly'))
            cbar.set_label(cbar_label)

        elif style == "wind_speed" and wind_speed is not None:
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
                cf = ax.contourf(X, Y, temp_c, levels=np.arange(-66, 56, 2), cmap=cmap_obj, extend='both')
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
        elif style == "vpd":
            vpd = data.get('vpd')
            if vpd is not None:
                # Green (moist) → Yellow → Orange → Red (dry/fire risk)
                vpd_colors = ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b',
                              '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026']
                vpd_cmap = mcolors.LinearSegmentedColormap.from_list('vpd', vpd_colors, N=256)
                cf = ax.contourf(X, Y, vpd, levels=np.arange(0, 65, 5), cmap=vpd_cmap, extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('VPD (hPa)')
                shading_label = "VPD"
        elif style == "dewpoint_dep":
            dd = data.get('dewpoint_dep')
            if dd is not None:
                # Green (saturated) → Yellow → Brown (very dry)
                dd_colors = ['#006837', '#1a9850', '#66bd63', '#a6d96a', '#d9ef8b',
                             '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026']
                dd_cmap = mcolors.LinearSegmentedColormap.from_list('dd', dd_colors, N=256)
                cf = ax.contourf(X, Y, dd, levels=np.arange(0, 42, 2), cmap=dd_cmap, extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Dewpoint Depression (°C)')
                # Saturation line (dd=0)
                try:
                    cs_sat = ax.contour(X, Y, dd, levels=[0], colors='cyan', linewidths=2.5)
                    ax.clabel(cs_sat, inline=True, fontsize=9, fmt='SAT', colors='black')
                except:
                    pass
                shading_label = "T-Td"
        elif style == "moisture_transport":
            mt = data.get('moisture_transport')
            if mt is not None:
                cf = ax.contourf(X, Y, mt, levels=np.linspace(0, 200, 21), cmap='YlGnBu', extend='max')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Moisture Transport (g·m/kg/s)')
                shading_label = "q·V"
        elif style == "pv":
            pv = data.get('pv')
            if pv is not None:
                # Stratospheric PV is large positive; tropospheric is small
                pv_colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3',
                             '#f5f5f5',
                             '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']
                pv_cmap = mcolors.LinearSegmentedColormap.from_list('pv', pv_colors, N=256)
                cf = ax.contourf(X, Y, pv, levels=np.arange(-2, 10.5, 0.5), cmap=pv_cmap, extend='both')
                cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.68])
                cbar = plt.colorbar(cf, cax=cbar_ax)
                cbar.set_label('Potential Vorticity (PVU)')
                # 2 PVU dynamical tropopause
                try:
                    cs_trop = ax.contour(X, Y, pv, levels=[2], colors='magenta', linewidths=2.5)
                    ax.clabel(cs_trop, inline=True, fontsize=9, fmt='2 PVU', colors='black')
                except:
                    pass
                shading_label = "PV"
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
                # Use ref levels for consistent terrain fill bottom across GIF frames
                ref_max = ref_pressure_levels[ref_pressure_levels >= y_top].max() if ref_pressure_levels is not None else pressure_levels_filtered.max()
                max_p = max(ref_max, np.nanmax(sp_hires)) + 20
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
            # Use ref_pressure_levels for consistent y-axis across GIF frames
            ylim_levels = ref_pressure_levels if ref_pressure_levels is not None else pressure_levels_filtered
            ylim_levels_f = ylim_levels[ylim_levels >= y_top]
            ax.set_ylim(max(ylim_levels_f), min(ylim_levels_f))
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

        # Anomaly subtitle
        if anomaly and climo_info:
            n_yrs = len(climo_info.get('years', []))
            n_samples = climo_info.get('n_samples', 0)
            month_name = climo_info.get('month_name', '')
            fig.text(0.06, 0.82, f'Departure from {n_yrs}-yr HRRR Mean ({month_name}, n={n_samples})',
                    fontsize=10, style='italic', color='#B71C1C',
                    transform=fig.transFigure)

        # A/B endpoint labels below the secondary axis
        total_dist = distances[-1]
        a_label = f"A\n{abs(lats[0]):.2f}°{'N' if lats[0]>=0 else 'S'}, {abs(lons[0]):.2f}°{'W' if lons[0]<0 else 'E'}"
        b_label = f"B\n{abs(lats[-1]):.2f}°{'N' if lats[-1]>=0 else 'S'}, {abs(lons[-1]):.2f}°{'W' if lons[-1]<0 else 'E'}"
        fig.text(0.06, 0.045, a_label, ha='left', va='top', fontsize=8, fontweight='bold',
                color='#333', transform=fig.transFigure)
        fig.text(0.88, 0.045, b_label, ha='right', va='top', fontsize=8, fontweight='bold',
                color='#333', transform=fig.transFigure)

        # Lat/lon labels along x-axis
        n_ticks = min(8, max(3, int(total_dist / 200)))  # ~1 tick per 200km
        tick_indices = np.linspace(0, len(distances) - 1, n_ticks + 2, dtype=int)[1:-1]  # Skip A/B endpoints

        tick_positions = []
        tick_labels = []
        for idx in tick_indices:
            d = distances[idx]
            lat_i, lon_i = lats[idx], lons[idx]
            label = f"{abs(lat_i):.1f}°{'N' if lat_i >= 0 else 'S'}, {abs(lon_i):.1f}°{'W' if lon_i < 0 else 'E'}"
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
            from cartopy.feature import ShapelyFeature

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
            pc = ccrs.PlateCarree()
            axins = fig.add_axes([0.88 - inset_width, 0.83, inset_width, inset_height],
                                projection=pc)
            axins.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)

            # Use cached geometries (parsed once per process, not per render)
            cached = self._get_cartopy_features()

            # Add terrain background from cached geometries
            axins.add_feature(ShapelyFeature(cached['land'], pc, facecolor='#C4B896', edgecolor='none'), zorder=0)
            axins.add_feature(ShapelyFeature(cached['ocean'], pc, facecolor='#97B6C8', edgecolor='none'), zorder=0)
            axins.add_feature(ShapelyFeature(cached['lakes'], pc, facecolor='#97B6C8', edgecolor='#6090A0', linewidth=0.3), zorder=1)
            axins.add_feature(ShapelyFeature(cached['states'], pc, facecolor='none', edgecolor='#666666', linewidth=0.4), zorder=2)
            axins.add_feature(ShapelyFeature(cached['borders'], pc, facecolor='none', edgecolor='#333333', linewidth=0.6), zorder=2)
            axins.add_feature(ShapelyFeature(cached['coastline'], pc, facecolor='none', edgecolor='#444444', linewidth=0.5), zorder=2)

            # Draw cross-section path
            axins.plot(lons, lats, 'r-', linewidth=2.5, transform=pc, zorder=10)
            # Start point - A label
            axins.text(lons[0], lats[0], 'A', transform=pc, zorder=11,
                      fontsize=10, fontweight='bold', ha='center', va='center',
                      color='white', bbox=dict(boxstyle='round,pad=0.15', facecolor='#38bdf8',
                                               edgecolor='white', linewidth=1.5))
            # End point - B label
            axins.text(lons[-1], lats[-1], 'B', transform=pc, zorder=11,
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
        fig.text(0.5, 0.005, 'Produced by drewsny  |  Contributors: @jasonbweather, justincat66, Sequoiagrove, California Wildfire Tracking & others',
                 ha='center', va='bottom', fontsize=7, color='#888888',
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
