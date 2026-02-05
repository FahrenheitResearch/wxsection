# Cross-Section Dashboard Context

## Current State (Feb 4, 2026)

### Branch: `focused/cross-section-dashboard`

Focused repo for interactive HRRR cross-section visualization.

### What Works
- **Dashboard**: `tools/unified_dashboard.py` - Flask + Leaflet interactive map
- **Cross-section engine**: `core/cross_section_interactive.py` - Sub-second generation
- **NPZ caching**: `cache/dashboard/xsect/` (~2s per hour from cache vs 25s from GRIB, 400GB limit with eviction)
- **15 styles**: wind_speed, temp, theta_e, rh, q, omega, vorticity, shear, lapse_rate, cloud, cloud_total, wetbulb, icing, frontogenesis, smoke
- **Color-coded chip UI**: Grey (on disk) / Green (in RAM) / Blue (viewing) / Yellow (loading) / Shift+click to unload
- **Plot annotations**: A/B labels, ~100+ city labels, lat/lon secondary axis, legend box, inset map with A/B badges
- **Distance units**: km/mi toggle at render time
- **Even FHR preloading**: Only even hours (F00, F02, ..., F18) loaded to RAM for latest 2 cycles (~53GB)
- **Smart memory management**: 117GB hard cap, LRU eviction at 115GB
- **Disk management**: 500GB GRIB limit + 400GB NPZ cache limit with separate eviction
- **Auto-update daemon**: `tools/auto_update.py` downloads latest cycles continuously
- **Custom date requests**: Calendar button downloads any date from NOMADS/AWS with live progress toast
- **GRIB downloads**: `smart_hrrr/orchestrator.py` with parallel threading
- **Community favorites**: Save/load cross-section configs, 12h expiry
- **Cloudflare Tunnel**: Public access via `cloudflared`
- **GIF animation**: `/api/xsect_gif` generates animated GIF of all loaded FHRs, terrain locked to F00, normal/slow speed
- **Terrain masking**: Theta contours, freezing level, and isotherms masked below terrain surface via `np.ma.masked_where`
- **Temperature colormaps**: 3 options (green_purple, white_zero, nws_ndfd) via `_build_temp_colormap()`, selectable in UI dropdown

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

### Key Architecture Details
- **Engine key system**: `_engine_key_map` maps `(cycle_key, fhr)` to auto-incrementing int for `self.forecast_hours` dict
- **Real FHR passthrough**: Dashboard sets `self.xsect._real_forecast_hour = fhr` before calling engine, engine reads via `getattr(self, '_real_forecast_hour', ...)` for metadata labels
- **Colorbar positioning**: Manual `fig.add_axes([0.90, 0.12, 0.012, 0.68])` with `cax=cbar_ax` (15 instances) - avoids `plt.colorbar(ax=ax)` stealing axes space
- **City labels**: `ax.secondary_xaxis(-0.08)` for aligned secondary axis, 120km search radius, `used_cities` set for dedup
- **Unit conversion**: `dist_scale = KM_TO_MI if use_miles else 1.0` applied to distances at render time, all internal math stays in km
- **Figure size**: `figsize=(17, 11)`, axes at `[0.06, 0.12, 0.82, 0.68]`
- **Terrain masking**: `terrain_mask` built from `pressure_levels > surface_pressure[i]`, applied to theta/temperature contours via `np.ma.masked_where`. Wind barbs also masked. contourf left unmasked (terrain fill zorder=5 covers it)
- **Temperature colormaps**: `_build_temp_colormap(name)` static method returns one of 3 colormaps. All defined as °F anchor arrays `(°F, (R,G,B))`, converted to °C internally. 512-bin `LinearSegmentedColormap`. Param threaded through `get_cross_section()` → `_render_cross_section()` as `temp_cmap`
- **GIF terrain lock**: GIF endpoint extracts terrain from first FHR via `get_terrain_data()`, passes as `terrain_data` override to all subsequent frames so terrain doesn't jitter with surface pressure changes
- **Smoke loading**: Uses eccodes (not cfgrib) to read MASSDEN (disc=0, cat=20, num=0) from wrfnat — cfgrib can't identify this field (shows as `unknown`). Kept on **native hybrid levels** (50 levels, ~10-15 packed in lowest 2km) with per-column pressure coordinate — NOT interpolated to isobaric. This preserves boundary layer vertical detail where smoke concentrates. Stored as `smoke_hyb` + `smoke_pres_hyb` in ForecastHourData and NPZ cache. Units: kg/m³ × 10⁹ = μg/m³. Plotted with its own X/Y mesh (pressure varies per column due to terrain).
- **Smoke backfill**: When loading from NPZ cache, if `smoke_hyb` is missing but wrfnat file exists, smoke is automatically loaded from wrfnat and cache is updated. Handles transition from pre-smoke caches.
- **Auto-update wrfnat awareness**: `auto_update.py` checks for wrfnat completeness — an FHR is only "downloaded" if wrfprs AND wrfnat both exist. Downloads wrfprs, wrfsfc, and wrfnat for all cycles.

### Key APIs
```python
# Interactive cross-section
from core.cross_section_interactive import InteractiveCrossSection
ixs = InteractiveCrossSection(cache_dir="cache/dashboard/xsect")
ixs.load_forecast_hour(data_dir, forecast_hour)
img_bytes = ixs.get_cross_section(start_point, end_point, style, forecast_hour, units='km')

# Parallel downloads
from smart_hrrr.orchestrator import download_gribs_parallel
download_gribs_parallel(model, date_str, cycle_hour, forecast_hours)
```

### Running Dashboard
```bash
python tools/unified_dashboard.py --port 5559
python tools/unified_dashboard.py --port 5559 --production  # with rate limiting

# Auto-update daemon (separate process)
python tools/auto_update.py --interval 2 --max-hours 18

# Public access
cloudflared tunnel --url http://localhost:5559
```

### Data Location
- GRIB files: `outputs/hrrr/YYYYMMDD/HHz/F##/`
- NPZ cache: `cache/dashboard/xsect/`
- Favorites: `data/favorites.json`
- Votes: `data/votes.json`
- Disk metadata: `data/disk_meta.json`
