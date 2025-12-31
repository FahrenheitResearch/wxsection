# Cross-Section Dashboard Context

## Current State (Dec 30, 2025)

### Branch: `focused/cross-section-dashboard`

Focused repo for interactive HRRR cross-section visualization.

### What Works
- **Dashboard**: `tools/unified_dashboard.py` - Flask + Leaflet interactive map
- **Cross-section engine**: `core/cross_section_interactive.py` - Sub-second generation
- **NPZ caching**: `cache/dashboard/xsect/` (~2s per hour from cache vs 25s from GRIB)
- **14 styles**: wind_speed, temp, theta_e, rh, q, omega, vorticity, shear, lapse_rate, cloud, cloud_total, wetbulb, icing, frontogenesis
- **Split & Chip UI**: Model Run dropdown + Forecast Hour chips (F00, F06, F12, F18)
- **Smart preloading**: Prefers complete cycles (all 4 FHRs available)
- **GRIB downloads**: `smart_hrrr/orchestrator.py` with parallel threading

### Files (14 Python files)
```
config/colormaps.py
core/__init__.py
core/cross_section_interactive.py   # Main engine - get_cross_section()
core/cross_section_production.py    # Batch generation
core/downloader.py
core/grib_loader.py
model_config.py
smart_hrrr/__init__.py
smart_hrrr/availability.py
smart_hrrr/io.py
smart_hrrr/orchestrator.py
smart_hrrr/utils.py
tools/auto_update.py
tools/unified_dashboard.py
```

### Key Features
- **Frontogenesis**: Petterssen kinematic formula with Gaussian smoothing ("Winter Bander" mode)
- **Wind barbs**: Rotated to cross-section orientation
- **Terrain masking**: Below surface pressure
- **Metadata labels**: Init time, valid time, path distance
- **Memory management**: ~1.7 GB per forecast hour, max 2 cycles loaded

### Key APIs
```python
# Interactive cross-section
from core.cross_section_interactive import InteractiveCrossSection
ixs = InteractiveCrossSection(cache_dir="cache/dashboard/xsect")
ixs.load_forecast_hour(data_dir, forecast_hour)
img_bytes = ixs.get_cross_section(start_point, end_point, style, forecast_hour)

# Parallel downloads
from smart_hrrr.orchestrator import download_gribs_parallel
download_gribs_parallel(model, date_str, cycle_hour, forecast_hours)
```

### Running Dashboard
```bash
python tools/unified_dashboard.py --auto-update --port 5559
python tools/unified_dashboard.py --auto-update --port 5559 --production  # with rate limiting
```

### Data Location
- GRIB files: `outputs/hrrr/YYYYMMDD/HHz/F##/`
- NPZ cache: `cache/dashboard/xsect/`
