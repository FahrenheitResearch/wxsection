# HRRR Cross-Section Generator

Interactive vertical atmospheric cross-section visualization from HRRR (High-Resolution Rapid Refresh) weather model data. Draw a line on a map, get an instant cross-section showing the vertical structure of the atmosphere.

**Live demo:** Deployed via Cloudflare Tunnel (URL changes on restart)

## What It Does

Cross-sections slice through the atmosphere along a path between two geographic points, revealing:
- Jet streams and wind maxima
- Temperature inversions and frontal boundaries
- Moisture transport and dry air intrusions
- Rising/sinking motion (omega)
- Snow banding potential (frontogenesis)
- Icing hazards for aviation
- Wildfire smoke plumes and air quality (PM2.5)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the auto-update daemon (downloads latest HRRR data continuously)
python tools/auto_update.py --interval 2 --max-hours 18 &

# Run the dashboard
python tools/unified_dashboard.py --port 5559

# Expose publicly (optional)
cloudflared tunnel --url http://localhost:5559
```

## Features

### Interactive Web Dashboard
- **Leaflet map** with click-to-place markers and draggable endpoints
- **15 visualization styles** via dropdown with community voting
- **19 forecast hours** (F00-F18) with color-coded chip system:
  - **Grey** = downloaded, not loaded (click to load, ~15s)
  - **Green** = loaded in RAM (click for instant view)
  - **Blue** = currently viewing
  - **Yellow pulse** = loading in progress
  - **Shift+click** to unload (prevents accidental unloads)
- **Model run picker** grouped by date with `<optgroup>`, shows load status and FHR count
- **Height/pressure toggle** - view Y-axis as hPa or km
- **Vertical scaling** - 1x, 1.5x, 2x exaggeration
- **Vertical range selector** - full atmosphere (100 hPa), mid (300), low (500), boundary layer (700)
- **Distance units toggle** - km or miles
- **Community favorites** - save/load cross-section configs by name, auto-expire after 12h
- **Feature request system** - users can submit and vote on feature ideas

### Plot Annotations
- **A/B endpoint labels** on the cross-section and inset map
- **City/location labels** along the x-axis (~100+ US cities, 120km search radius, deduplicated)
- **Lat/lon coordinates** on a secondary x-axis below the main plot
- **Legend box** with color-coded entries for theta, freezing level, and terrain
- **Inset map** with matplotlib (no external tile dependency) showing the cross-section path with A/B badges
- **Credit text** with research collaborative attribution

### Continuous Auto-Updating
- **Progressive download daemon** (`auto_update.py`) checks every 2 minutes
- Maintains latest 2 init cycles with F00-F18 (full forecast set)
- Client-side auto-refresh polls every 60s for newly available data
- Background server rescan detects new downloads without restart
- **Parallel loading** with 4 worker threads for faster data loading

### Memory Management
- **Even FHR preloading** - latest 2 cycles pre-loaded with even forecast hours only (F00, F02, ..., F18) to balance RAM usage (~53 GB) vs coverage
- **LRU eviction** starting at 115 GB, hard cap at 117 GB
- Unique engine key mapping allows multiple init cycles loaded simultaneously
- Odd forecast hours load on-demand when requested

### Disk Storage
- **NPZ cache** with configurable limit (default 400 GB) - separate from GRIB storage
- **GRIB disk limit** of 500 GB with space-based eviction
- **Popularity tracking** via `data/disk_meta.json` (last accessed time + access count)
- Protected cycles: latest 2 auto-update targets + anything accessed in last 2 hours
- Disk usage checked every 10 minutes, evicts least-recently-accessed first

### Custom Date Requests
- **Calendar button** lets users download F00-F18 for any date/init cycle
- Downloads from NOMADS (recent, <48h) or AWS archive (older data)
- **Live progress toast** showing download count (e.g., "7/19 FHRs") with progress bar, polls every 10s

### Performance
- **Sub-second generation** after data is loaded (~0.3s typical)
- **NPZ caching** - first GRIB load ~25s, subsequent loads ~2s from cache
- **Parallel GRIB download** with configurable thread count
- Non-blocking startup - Flask serves immediately while data loads in background

### Production Ready
- Rate limiting (60 req/min) for public deployment
- REST API for programmatic access
- Cloudflare Tunnel integration for public access
- Batch generation for animations

## Visualization Styles

### Core Meteorology
| Style | Shows | Use For |
|-------|-------|---------|
| `wind_speed` | Horizontal wind (kt) | Jet streams, wind maxima |
| `temp` | Temperature (°C) with NWS NDFD colormap | Inversions, frontal zones |
| `theta_e` | Equivalent potential temp (K) | Warm/cold advection, instability |
| `omega` | Vertical velocity | Rising (blue) / sinking (red) motion |
| `vorticity` | Absolute vorticity | Cyclonic/anticyclonic patterns |

### Moisture & Clouds
| Style | Shows | Use For |
|-------|-------|---------|
| `rh` | Relative humidity (%) | Dry slots, moisture plumes |
| `q` | Specific humidity (g/kg) | Moisture transport |
| `cloud` | Cloud water (g/kg) | Cloud layers |
| `cloud_total` | All hydrometeors | Full precipitation picture |

### Smoke & Air Quality
| Style | Shows | Use For |
|-------|-------|---------|
| `smoke` | PM2.5 concentration (μg/m³) | **Wildfire smoke plumes, air quality** |

Smoke data comes from HRRR-Smoke MASSDEN field in wrfnat files (~670MB each), read via eccodes (cfgrib can't identify this field). Smoke is kept on **native hybrid levels** (50 levels) rather than interpolated to isobaric — this preserves the fine vertical resolution (~10-15 levels in the lowest 2 km) where smoke concentrates. Each column has its own pressure coordinate that follows terrain. Auto-scaled colormap adjusts to fire intensity with 50 smooth contour levels. AQI reference contours at 35 (Moderate), 55 (USG), 150 (Unhealthy), 250 (Very Unhealthy) μg/m³. Requires wrfnat GRIB files — downloaded automatically alongside wrfprs/wrfsfc.

### Winter Weather & Aviation
| Style | Shows | Use For |
|-------|-------|---------|
| `frontogenesis` | Petterssen frontogenesis | **Snow banding potential** |
| `wetbulb` | Wet-bulb temperature (°C) | Rain/snow transition |
| `icing` | Supercooled liquid water | Aircraft icing hazard |
| `shear` | Wind shear (1/s) | Turbulence, jet cores |
| `lapse_rate` | Temp lapse rate (°C/km) | Stability analysis |

### All Styles Include
- **Theta contours** (black lines) - atmospheric stability
- **Wind barbs** with actual U and V components
- **Freezing level** (magenta line) - 0°C isotherm
- **Terrain fill** (brown) - contourf fills entire grid, terrain covers underground (standard met practice)
- **Legend box** identifying overlays
- **A/B endpoint markers** on plot and inset map
- **City labels** with lat/lon and distance along path

### Temperature Colormap
Uses the NWS NDFD color table with 0°C (freezing) = yellow:
- Purple (-60°C) -> Blue (-30°C) -> Cyan (-10°C) -> **Yellow (0°C)** -> Orange (20°C) -> Red (40°C)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/api/cycles` | GET | List available cycles (grouped by date) |
| `/api/status` | GET | Memory/load status |
| `/api/progress` | GET | Loading progress for UI progress bar |
| `/api/load` | POST | Load specific cycle + forecast hour |
| `/api/load_cycle` | POST | Load entire cycle (all FHRs) |
| `/api/unload` | POST | Unload a forecast hour |
| `/api/xsect` | GET | Generate cross-section PNG |
| `/api/request_cycle` | POST | Request download of a specific date/init |
| `/api/favorites` | GET | List community favorites |
| `/api/favorite` | POST | Save a favorite |
| `/api/favorite/<id>` | DELETE | Delete a favorite |
| `/api/votes` | GET | Style vote counts |
| `/api/vote` | POST | Vote for a style |

### Generate Cross-Section via API

```
GET /api/xsect?start_lat=40.0&start_lon=-100.0&end_lat=35.0&end_lon=-90.0&style=frontogenesis&cycle=20260204_04z&fhr=6&y_axis=pressure&vscale=1.5&y_top=300&units=km
```

Parameters:
- `start_lat`, `start_lon`, `end_lat`, `end_lon` - Cross-section endpoints
- `style` - Visualization style (see table above)
- `cycle` - Init cycle key (e.g., `20260204_04z`)
- `fhr` - Forecast hour (0-18)
- `y_axis` - `pressure` (hPa) or `height` (km)
- `vscale` - Vertical exaggeration (1.0, 1.5, 2.0)
- `y_top` - Top of plot in hPa (100, 200, 300, 500, 700)
- `units` - Distance axis units (`km` or `mi`)

## Architecture

```
tools/
├── unified_dashboard.py      # Flask server + Leaflet UI + data management
│   ├── CrossSectionManager   # Handles loading, eviction, engine key mapping
│   ├── Memory management     # 117GB cap, LRU eviction at 115GB
│   ├── Disk management       # 500GB GRIB + 400GB NPZ cache limits
│   ├── Community favorites   # Save/load/delete with 12h expiry
│   ├── FHR chip system       # Color-coded load states, shift+click unload
│   └── Thread-safe parallel loading
│
└── auto_update.py            # Continuous download daemon
    ├── Progressive download  # Latest 2 cycles, F00-F18 (wrfprs + wrfsfc + wrfnat)
    └── Space-based cleanup   # Evicts least-popular when disk full

core/
├── cross_section_interactive.py  # Fast interactive engine
│   ├── Pre-loads 3D fields into RAM
│   ├── NPZ caching layer (400GB limit with eviction)
│   ├── <1s cross-section generation
│   ├── City label proximity matching (120km radius)
│   ├── A/B endpoint labels and legend
│   ├── km/mi unit conversion at render time
│   ├── Progress callback for field-level tracking
│   ├── NWS NDFD temperature colormap
│   ├── Smoke on native hybrid levels (50 levels, not isobaric)
│   └── Smoke backfill: auto-loads from wrfnat when cache is stale
│
└── cross_section_production.py   # Batch processing

smart_hrrr/
├── orchestrator.py    # Parallel GRIB download coordination
├── availability.py    # Check NOMADS/AWS for available cycles
├── io.py              # Output directory structure
└── utils.py           # Shared utilities

data/
├── favorites.json     # Community favorites
├── votes.json         # Style votes
├── requests.json      # Feature requests
└── disk_meta.json     # Disk usage tracking (access times, counts)
```

## Memory Usage

| Loaded | RAM Usage |
|--------|-----------|
| 1 forecast hour (without smoke) | ~3.7 GB |
| 1 forecast hour (with smoke/wrfnat) | ~4.4 GB |
| Even FHRs, 1 cycle (F00,F02,...,F18) | ~27-44 GB |
| Even FHRs, 2 cycles (preloaded) | ~53-88 GB |
| Max before eviction | 115 GB |
| Hard cap | 117 GB |

The dashboard pre-loads even forecast hours from the latest 2 cycles at startup. Odd hours load on-demand (~15s from NPZ cache). LRU eviction kicks in at 115 GB. Smoke adds ~0.76 GB per FHR (50 hybrid levels × 2 arrays) when wrfnat files are available — a 21% increase over base.

## Command Line Options

### Dashboard
```
python tools/unified_dashboard.py [OPTIONS]

--port PORT          Server port (default: 5559)
--host HOST          Server host (default: 0.0.0.0)
--preload N          Cycles to pre-load at startup (default: 0)
--production         Enable rate limiting
--auto-update        Download latest data before starting
--max-hours N        Max forecast hour to download
```

### Auto-Update Daemon
```
python tools/auto_update.py [OPTIONS]

--interval N         Check interval in minutes (default: 2)
--max-hours N        Max forecast hour (default: 18)
--once               Run once and exit
--no-cleanup         Don't clean up old data
```

## Data Requirements

HRRR GRIB2 files with pressure-level data:

```
outputs/hrrr/{YYYYMMDD}/{HH}z/F{XX}/
├── hrrr.t{HH}z.wrfprsf{XX}.grib2  # Pressure levels (required)
├── hrrr.t{HH}z.wrfsfcf{XX}.grib2  # Surface (for terrain)
└── hrrr.t{HH}z.wrfnatf{XX}.grib2  # Native levels (for smoke, ~670MB)
```

Required fields: Temperature, U/V wind, RH, geopotential height, specific humidity, vorticity, cloud water, dew point on isobaric levels. Surface pressure from surface file for terrain. MASSDEN (smoke PM2.5, GRIB2 disc=0/cat=20/num=0) from native-level file for smoke style — read via eccodes because cfgrib can't identify this field (shows as `unknown` with paramId=0). Stored on native hybrid levels with per-column pressure, not interpolated to isobaric.

Data is automatically downloaded from NOAA NOMADS (recent, <48h) or AWS archive (older) by the auto-update daemon or on-demand via the calendar request button.

## Credits

Contributors: @jasonbweather, Sequoiagrove & others

## Dependencies

```
numpy
scipy
matplotlib
cfgrib
eccodes
flask
imageio
Pillow
```

Install with: `pip install -r requirements.txt`

For public access: `cloudflared` (Cloudflare Tunnel client)

## References

- [HRRR Model](https://rapidrefresh.noaa.gov/hrrr/) - NOAA's 3km CONUS model
- [NWS NDFD Color Tables](https://www.weather.gov/media/mdl/ndfd/NDFDelem_fullres.pdf) - Temperature colormap reference
- [Petterssen Frontogenesis](https://glossary.ametsoc.org/wiki/Frontogenesis) - AMS Glossary
- [cfgrib](https://github.com/ecmwf/cfgrib) - GRIB file reader
