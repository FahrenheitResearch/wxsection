# HRRR Cross-Section Generator

Interactive vertical atmospheric cross-section visualization from HRRR (High-Resolution Rapid Refresh) weather model data. Draw a line on a map, get an instant cross-section showing the vertical structure of the atmosphere.

**Live:** [wxsection.com](https://wxsection.com)

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

# Production startup (dashboard + auto-update + Cloudflare tunnel)
export WXSECTION_KEY=your_admin_key
./deploy/run_production.sh

# Or run manually:
python tools/auto_update.py --interval 2 --max-hours 18 &
python tools/unified_dashboard.py --port 5559 --preload 2
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
- **Vertical scaling** - 0.5x, 1x, 1.5x, 2x exaggeration
- **Vertical range selector** - full atmosphere (100 hPa), mid (300), low (500), boundary layer (700)
- **Distance units toggle** - km or miles
- **Community favorites** - save/load cross-section configs by name, auto-expire after 12h
- **GIF animation** - generate animated GIF of every-3rd-hour forecast hours with normal/slow speed control
- **Temperature colormap picker** - 3 color tables (Green-Purple, White at 0°C, NWS Classic) switchable on the fly
- **Admin key system** - lock icon (🔒) for archive access, stored in browser localStorage

### Plot Annotations
- **A/B endpoint labels** on the cross-section and inset map
- **City/location labels** along the x-axis (~100+ US cities, 120km search radius, deduplicated)
- **Lat/lon coordinates** on a secondary x-axis below the main plot
- **Legend box** with color-coded entries for theta, freezing level, and terrain
- **Inset map** with matplotlib (no external tile dependency) showing the cross-section path with A/B badges
- **Credit text** with producer and contributor attribution

### Continuous Auto-Updating
- **Progressive download daemon** (`auto_update.py`) checks every 2 minutes
- Maintains latest 2 init cycles with F00-F18 (full forecast set)
- Client-side auto-refresh polls every 60s for newly available data
- Background server rescan detects new downloads without restart
- **Auto-load**: new FHRs for latest 2 cycles automatically load into RAM as they're downloaded (every 60s check)
- **Parallel loading** with 4 worker threads for preloading and on-demand cycle loads

### Memory Management
- **Every-3rd-hour preloading** - latest 2 cycles pre-loaded with F00, F03, F06, F09, F12, F15, F18 via 4 parallel workers to balance RAM (~52-62 GB) vs coverage
- **Protected cycles** - latest 2 init cycles cannot be unloaded by regular users or evicted by LRU
- **LRU eviction** starting at 115 GB, hard cap at 117 GB (skips protected cycles)
- Unique engine key mapping allows multiple init cycles loaded simultaneously
- Non-preloaded forecast hours load on-demand when requested
- **Render semaphore** - caps concurrent matplotlib renders at 4 to prevent CPU/memory thrash under load

### Admin Key System
- Set via `WXSECTION_KEY` environment variable (never stored in code)
- **Required for**: loading archive/older cycles into RAM, downloading custom dates, unloading protected cycles
- **Not required for**: viewing cross-sections from already-loaded data, loading FHRs in latest 2 cycles
- UI: click 🔒 icon, enter key, saved to browser localStorage
- Validates via `/api/check_key` endpoint

### Disk Storage
- **NPZ cache** with configurable limit (default 400 GB) - separate from GRIB storage
- **GRIB disk limit** of 500 GB with space-based eviction
- **Popularity tracking** via `data/disk_meta.json` (last accessed time + access count)
- Protected cycles: latest 2 auto-update targets + anything accessed in last 2 hours
- Disk usage checked every 10 minutes, evicts least-recently-accessed first

### Custom Date Requests
- **Calendar button** (📅) lets admin users download F00-F18 for any date/init cycle
- Downloads from NOMADS (recent, <48h) or AWS archive (older data)
- **Live progress toast** showing download count (e.g., "7/19 FHRs") with progress bar, polls every 10s

### Performance
- **Sub-second generation** after data is loaded (~0.3s typical)
- **NPZ caching** - first GRIB load ~25s, subsequent loads ~2s from cache
- **Parallel GRIB download** with configurable thread count
- Non-blocking startup - Flask serves immediately while data loads in background
- **Render semaphore** prevents server overload under concurrent use (10-25 users)

### Production Ready
- Rate limiting (60 req/min) for public deployment
- REST API for programmatic access
- Named Cloudflare Tunnel (wxsection.com) for permanent public access
- Startup script (`deploy/run_production.sh`) manages all 3 services
- Batch generation for animations

## Visualization Styles

### Core Meteorology
| Style | Shows | Use For |
|-------|-------|---------|
| `wind_speed` | Horizontal wind (kt) | Jet streams, wind maxima |
| `temp` | Temperature (°C) with 3 selectable colormaps | Inversions, frontal zones |
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

Smoke data comes from HRRR-Smoke MASSDEN field in wrfnat files (~670MB each), read via eccodes (cfgrib can't identify this field). Smoke is kept on **native hybrid levels** (50 levels) rather than interpolated to isobaric — this preserves the fine vertical resolution (~10-15 levels in the lowest 2 km) where smoke concentrates. Each column has its own pressure coordinate that follows terrain. Auto-scaled colormap adjusts to fire intensity with 50 smooth contour levels. Requires wrfnat GRIB files — downloaded automatically alongside wrfprs/wrfsfc.

### Winter Weather & Aviation
| Style | Shows | Use For |
|-------|-------|---------|
| `frontogenesis` | Petterssen frontogenesis | **Snow banding potential** |
| `wetbulb` | Wet-bulb temperature (°C) | Rain/snow transition |
| `icing` | Supercooled liquid water | Aircraft icing hazard |
| `shear` | Wind shear (1/s) | Turbulence, jet cores |
| `lapse_rate` | Temp lapse rate (°C/km) | Stability analysis |

### All Styles Include
- **Theta contours** (black lines) - atmospheric stability, masked below terrain
- **Wind barbs** with actual U and V components, masked below terrain
- **Freezing level** (magenta line) - 0°C isotherm, masked below terrain
- **Terrain fill** (brown) - hi-res ~1.5km sampling with bilinear interpolation
- **Legend box** identifying overlays
- **A/B endpoint markers** on plot and inset map
- **City labels** with lat/lon and distance along path

### Temperature Colormaps
Three selectable color tables (dropdown appears when Temperature style is active):

| Colormap | 0°C (32°F) | Cold End | Hot End | Best For |
|----------|-----------|----------|---------|----------|
| **Green-Purple** | Green | Blue-teal | Deep purple | General use, intuitive warm/cold |
| **White at 0°C** | White | Purple tones | Deep purple (hot) | Freezing boundary emphasis |
| **NWS Classic** | Yellow | Indigo/blue | Deep purple | Traditional NWS style |

All defined with °F anchor points, converted to °C internally. 512-bin interpolation with 2°C contour steps (even numbers, 0°C always a contour level).

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | | Dashboard UI |
| `/api/cycles` | GET | | List available cycles (grouped by date) |
| `/api/status` | GET | | Memory/load status |
| `/api/progress` | GET | | Loading progress for UI progress bar |
| `/api/check_key` | GET | | Validate admin key, get protected cycles |
| `/api/load` | POST | Archive* | Load specific cycle + forecast hour |
| `/api/load_cycle` | POST | Archive* | Load entire cycle (all FHRs) |
| `/api/unload` | POST | Protected* | Unload a forecast hour |
| `/api/xsect` | GET | | Generate cross-section PNG |
| `/api/xsect_gif` | GET | | Generate animated GIF of every-3rd-hour FHRs |
| `/api/request_cycle` | POST | Admin | Request download of a specific date/init |
| `/api/favorites` | GET | | List community favorites |
| `/api/favorite` | POST | | Save a favorite |
| `/api/favorite/<id>` | DELETE | | Delete a favorite |
| `/api/votes` | GET | | Style vote counts |
| `/api/vote` | POST | | Vote for a style |

*Archive = admin key required only for non-latest-2 cycles. Protected = admin key required to unload latest 2 cycles.

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
- `vscale` - Vertical exaggeration (0.5, 1.0, 1.5, 2.0)
- `y_top` - Top of plot in hPa (100, 200, 300, 500, 700)
- `units` - Distance axis units (`km` or `mi`)
- `temp_cmap` - Temperature colormap (`green_purple`, `white_zero`, `nws_ndfd`)

## Architecture

```
deploy/
└── run_production.sh        # Start/stop all 3 services (dashboard + auto-update + tunnel)

tools/
├── unified_dashboard.py      # Flask server + Leaflet UI + data management
│   ├── CrossSectionManager   # Handles loading, eviction, engine key mapping
│   ├── Memory management     # 117GB cap, LRU eviction at 115GB, protected cycles
│   ├── Disk management       # 500GB GRIB + 400GB NPZ cache limits
│   ├── Admin key system      # WXSECTION_KEY env var, gates archive access
│   ├── Render semaphore      # Caps concurrent matplotlib renders at 4
│   ├── Auto-load             # Background thread loads new FHRs for latest 2 cycles
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
│   ├── 3 selectable temperature colormaps (Green-Purple, White-Zero, NWS)
│   ├── Terrain masking for contour lines (theta, freezing, isotherms)
│   ├── Smoke on native hybrid levels (50 levels, not isobaric)
│   ├── Smoke backfill: auto-loads from wrfnat when cache is stale
│   └── GIF animation with terrain-locked frames
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
| Every-3rd FHR, 1 cycle (F00,F03,...,F18) | ~26-31 GB |
| Every-3rd FHR, 2 cycles (preloaded) | ~52-62 GB |
| Max before eviction | 115 GB |
| Hard cap | 117 GB |

The dashboard pre-loads every 3rd forecast hour (F00, F03, F06, F09, F12, F15, F18) from the latest 2 cycles at startup using 4 parallel workers. As the auto-update daemon downloads new FHRs, the background auto-load thread picks them up within 60 seconds. Other hours load on-demand (~15s from NPZ cache). LRU eviction kicks in at 115 GB but skips the latest 2 protected cycles. Smoke adds ~0.76 GB per FHR (50 hybrid levels × 2 arrays) when wrfnat files are available — a 21% increase over base.

## Command Line Options

### Dashboard
```
WXSECTION_KEY=secret python tools/unified_dashboard.py [OPTIONS]

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

### Production Startup
```bash
export WXSECTION_KEY=your_secret
./deploy/run_production.sh        # Start all services
./deploy/run_production.sh stop   # Stop all services
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

Data is automatically downloaded from NOAA NOMADS (recent, <48h) or AWS archive (older) by the auto-update daemon or on-demand via the calendar request button (admin key required).

## Credits

Produced by drewsny

Contributors: @jasonbweather, justincat66, Sequoiagrove, California Wildfire Tracking & others

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
- [Petterssen Frontogenesis](https://glossary.ametsoc.org/wiki/Frontogenesis) - AMS Glossary
- [cfgrib](https://github.com/ecmwf/cfgrib) - GRIB file reader
