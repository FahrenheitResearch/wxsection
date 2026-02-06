# Multi-Model Cross-Section Generator

Interactive vertical atmospheric cross-section visualization from HRRR, GFS, and RRFS weather model data. Draw a line on a map, get an instant cross-section showing the vertical structure of the atmosphere.

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

# Production startup (mount VHD, start auto-update + dashboard + cloudflared)
cd ~/hrrr-maps && ./start.sh

# Or run manually:
sudo mount /dev/sde /mnt/hrrr
python tools/auto_update.py --interval 2 --models hrrr,gfs,rrfs &
XSECT_GRIB_BACKEND=auto WXSECTION_KEY=your_key python tools/unified_dashboard.py --port 5561 --models hrrr,gfs,rrfs
```

## Features

### Interactive Web Dashboard
- **Leaflet map** (OpenTopoMap) with click-to-place markers and draggable endpoints
- **Multi-model support**: HRRR (3km), GFS (0.25deg), RRFS (3km) with model selector dropdown
- **19 visualization styles** via dropdown with community voting
- **Up to 48 forecast hours** with color-coded chip system (synoptic HRRR cycles: F00-F48, others: F00-F18):
  - **Grey** = on disk, not opened (click to open via mmap, ~14ms if cached)
  - **Green** = opened (mmap handles ready, click for instant view)
  - **Blue** = currently viewing
  - **Yellow pulse** = loading in progress
  - **Shift+click** to unload (prevents accidental unloads)
- **Model run picker** filtered to preload window + loaded archive cycles
- **Height/pressure toggle** - view Y-axis as hPa or km
- **Vertical scaling** - 0.5x, 1x, 1.5x, 2x exaggeration
- **Vertical range selector** - full atmosphere (100 hPa), mid (300), low (500), boundary layer (700)
- **Distance units toggle** - km or miles
- **Community favorites** - save/load cross-section configs by name, auto-expire after 12h
- **GIF animation** - animated GIF with speed options, Pillow rendering with `disposal=2` for Discord compatibility
- **Temperature colormap picker** - 4 color tables (Standard, Green-Purple, White at 0C, NWS Classic)
- **Time slider + auto-play** - play/pause with 0.5x-4x speed, pre-render for instant scrubbing
- **Cycle comparison mode** - side-by-side Same FHR or Valid Time matching across different init cycles
- **Activity panel** - real-time progress for all operations (preload, auto-load, download, prerender, auto-update)
- **Auto-update progress** - per-model download status (HRRR/GFS/RRFS) via status file IPC with auto_update daemon
- **RAM status modal** - shows all models (HRRR + GFS + RRFS) with per-cycle FHR counts and memory estimates
- **Cancel operations** - admin can cancel pre-render and download jobs via X button in progress panel
- **Archive request modal** - calendar date picker, hour selector, FHR range for downloading historical data
- **Load All button** - loads all FHRs for the current cycle at once (available to all users)
- **Step buttons** - prev/next frame buttons flanking the play button for frame-by-frame navigation

### Performance
- **0.5s warm renders** - cartopy geometry cache + KDTree cache eliminate repeated I/O
- **~15s GRIB-to-mmap conversion** with eccodes backend (~35% faster than cfgrib)
- **<0.1s cached FHR loads** - mmap from NVMe, instant page faults
- **~4s cached preload** for 176 FHRs across all models (HRRR+GFS+RRFS)
- **Lazy smoke loading** - wrfnat (652MB, 50 hybrid levels) loaded on first smoke request, not during preload
- **Frame prerender cache** - 500-entry server-side cache, ~20ms cached vs ~0.5s live render
- **Parallel prerender** - ThreadPool(8) batch rendering, ~4s for 19 frames (was ~10s sequential)
- **Two-phase preload**: cached FHRs load instantly (Phase 1), GRIB conversions run in background (Phase 2)
- **Render semaphore** - caps concurrent matplotlib renders at 12 (8 prerender + 4 live)
- **Configurable workers** - `--grib-workers N` / `--preload-workers N` (or env `XSECT_GRIB_WORKERS` / `XSECT_PRELOAD_WORKERS`)

### Mmap Cache Architecture
- **Memory-mapped cache on NVMe** - per-field raw arrays, ~2.3GB per FHR on disk
- **Tiny RAM footprint** - mmap only pages in accessed slices (~100MB resident per FHR, ~29MB heap)
- **~125 FHRs in preload window** = ~290GB on NVMe, ~12GB in RAM
- **Two-tier NVMe eviction**:
  - Tier 1: Rotated preload cycles always evicted from cache when they leave target window
  - Tier 2: Archive request caches persist up to 670GB limit, oldest evicted first when over
- **Per-model memory budgets**: HRRR 48GB, GFS 8GB, RRFS 8GB

### Auto-Update (Slot-Based Concurrent)
- **Parallel download slots** - 3 HRRR + 1 GFS + 1 RRFS downloading simultaneously via ThreadPoolExecutor
- **Per-model lanes** - slow RRFS downloads can't block HRRR/GFS progress
- **HRRR fail-fast** - unavailable FHRs prune higher FHRs from same cycle
- **HRRR refresh** - re-scans for newly published FHRs every 45s while other models download
- **Status file IPC** - writes progress to `/tmp/auto_update_status.json` for dashboard activity panel
- **Single-cycle targeting** - only downloads latest available cycle per model (no handoff)
- **Extended 48h** for HRRR synoptic cycles (00/06/12/18z)
- **Configurable**: `--hrrr-slots 3 --gfs-slots 1 --rrfs-slots 1`

### Admin Key System
- Set via `WXSECTION_KEY` environment variable (never stored in code)
- **Required for**: downloading archive data, cancelling operations
- **Not required for**: loading/unloading, Load All, GIF, viewing cross-sections
- UI: click lock icon, enter key, saved to browser localStorage

### Disk Storage
- **GRIB sources on VHD** (`/mnt/hrrr/`) - 20TB external VHDX, 500GB GRIB limit with LRU eviction
- **Mmap cache on NVMe** (`cache/xsect/{model}/`) - ~400GB preload + ~245GB archive headroom
- **GRIB disk limit** of 500GB with popularity-based eviction
- **NVMe cache limit** of 670GB with two-tier eviction

## Visualization Styles

### Core Meteorology
| Style | Shows | Use For |
|-------|-------|---------|
| `wind_speed` | Horizontal wind (kt) | Jet streams, wind maxima |
| `temp` | Temperature (C) with 4 selectable colormaps | Inversions, frontal zones |
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
| `smoke` | PM2.5 concentration (ug/m3) | Wildfire smoke plumes, air quality |

### Winter Weather & Aviation
| Style | Shows | Use For |
|-------|-------|---------|
| `frontogenesis` | Petterssen frontogenesis | Snow banding potential |
| `wetbulb` | Wet-bulb temperature (C) | Rain/snow transition |
| `icing` | Supercooled liquid water | Aircraft icing hazard |
| `shear` | Wind shear (1/s) | Turbulence, jet cores |
| `lapse_rate` | Temp lapse rate (C/km) | Stability analysis |

### Derived / Diagnostic
| Style | Shows | Use For |
|-------|-------|---------|
| `vpd` | Vapor pressure deficit (hPa) | Fire weather, plant stress |
| `dewpoint_dep` | T minus Td (C) | Dry layers, cloud base ID |
| `moisture_transport` | q times wind speed (g·m/kg/s) | Atmospheric rivers, moisture plumes |
| `pv` | Potential vorticity (PVU) | Tropopause folds, jet dynamics |

### All Styles Include
- **Theta contours** (black lines) - atmospheric stability, masked below terrain
- **Wind barbs** with actual U and V components, masked below terrain
- **Freezing level** (magenta line) - 0C isotherm, masked below terrain
- **Terrain fill** (brown) - hi-res ~1.5km sampling with bilinear interpolation
- **A/B endpoint markers** on plot and inset map
- **Lat/lon tick labels** with distance along path

## Architecture

```
start.sh                             # Production startup (mount VHD, start all services)
model_config.py                      # Model registry (HRRR/GFS/RRFS metadata)

tools/
├── unified_dashboard.py             # Flask server + Leaflet UI + data management
│   ├── CrossSectionManager          # Loading, eviction, engine key mapping
│   ├── Two-phase preload            # Cache-first (Phase 1) + GRIB conversion (Phase 2)
│   ├── Two-tier NVMe eviction       # Preload rotation + size-based archive eviction
│   ├── Cancel system                # CANCEL_FLAGS + admin API for pre-render/download abort
│   ├── Archive request modal        # Calendar + hour + FHR range, download + auto-load
│   ├── Activity panel               # Real-time progress for all ops (preload/autoload/download/autoupdate)
│   ├── Auto-update injection        # Reads /tmp/auto_update_status.json for download progress
│   ├── RAM status modal             # Multi-model memory view (HRRR + GFS + RRFS)
│   ├── Frame prerender cache        # 500-entry server-side PNG cache
│   ├── Community favorites          # Save/load/delete with 12h expiry
│   └── Admin key system             # WXSECTION_KEY env var
│
├── auto_update.py                   # Slot-based concurrent download daemon
│   ├── Concurrent slots             # 3 HRRR + 1 GFS + 1 RRFS in parallel
│   ├── HRRR fail-fast               # Prunes unavailable FHRs from queue
│   ├── Extended 48h                 # Synoptic HRRR cycles get F19-F48
│   ├── Status file IPC              # Atomic JSON writes to /tmp/auto_update_status.json
│   └── Space-based cleanup          # Evicts least-popular when disk full
│
├── build_climatology.py             # Build monthly climatology NPZ from archived HRRR
└── bulk_download.py                 # Bulk HRRR archive downloader for VHD

core/
├── cross_section_interactive.py     # Fast interactive engine (0.5s warm renders)
│   ├── Cartopy geometry cache       # Parsed once per process
│   ├── KDTree cache                 # Per-grid, reused across all FHRs/paths
│   ├── Mmap-based field loading     # ~14ms per FHR from NVMe cache
│   ├── City label proximity matching
│   ├── 4 temperature colormaps
│   └── Smoke on native hybrid levels (50 levels)
│
└── cross_section_production.py      # Batch processing

smart_hrrr/
├── orchestrator.py                  # Parallel GRIB downloads (on_complete, on_start, should_cancel)
├── availability.py                  # Check NOMADS/AWS for available cycles
├── io.py                            # Output directory structure
└── utils.py                         # Shared utilities

data/
├── favorites.json                   # Community favorites
├── votes.json                       # Style votes
├── requests.json                    # Feature requests
└── disk_meta.json                   # Disk usage tracking
```

### Disk Layout
```
/dev/sdd (2TB NVMe VHD) mounted at /
  ~/hrrr-maps/                       — code
  ~/hrrr-maps/cache/xsect/          — ACTIVE mmap cache (NVMe)

/dev/sde (20TB external VHD) mounted at /mnt/hrrr
  /mnt/hrrr/hrrr-live/              — live HRRR GRIBs
  /mnt/hrrr/gfs/                    — live GFS GRIBs
  /mnt/hrrr/rrfs/                   — live RRFS GRIBs
  /mnt/hrrr/YYYYMMDD/              — archived HRRR GRIBs
  /mnt/hrrr/climatology/           — monthly mean NPZ files
```

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | | Dashboard UI |
| `/api/v1/cross-section` | GET | | Generate cross-section PNG (agent-friendly) |
| `/api/v1/products` | GET | | List available products |
| `/api/v1/cycles` | GET | | List available cycles |
| `/api/v1/status` | GET | | Server health check |
| `/api/xsect` | GET | | Generate cross-section PNG (internal) |
| `/api/xsect_gif` | GET | | Generate animated GIF |
| `/api/cycles` | GET | | List available cycles |
| `/api/status` | GET | | Memory/load status |
| `/api/progress` | GET | | Loading progress |
| `/api/load` | POST | Archive* | Load specific cycle + FHR |
| `/api/load_cycle` | POST | Archive* | Load entire cycle |
| `/api/unload` | POST | Protected* | Unload a forecast hour |
| `/api/request_cycle` | POST | Admin | Download archive cycle with FHR range |
| `/api/cancel` | POST | Admin | Cancel pre-render or download operation |
| `/api/prerender` | POST | | Pre-render frames for time slider |
| `/api/favorites` | GET | | List community favorites |
| `/api/favorite` | POST | | Save a favorite |
| `/api/check_key` | GET | | Validate admin key |

*Archive = admin key required for non-target cycles. Protected = admin required to unload target cycles.

See [API_GUIDE.md](API_GUIDE.md) for the full v1 API documentation.

## Command Line Options

### Dashboard
```
WXSECTION_KEY=secret python tools/unified_dashboard.py [OPTIONS]

--port PORT          Server port (default: 5561)
--host HOST          Server host (default: 0.0.0.0)
--models M           Comma-separated models (default: hrrr)
--grib-workers N     GRIB conversion threads (default: 4, env XSECT_GRIB_WORKERS)
--preload-workers N  Cached mmap load threads (default: 20, env XSECT_PRELOAD_WORKERS)
```

Environment: `XSECT_GRIB_BACKEND=auto` (default) tries eccodes, falls back to cfgrib.

### Auto-Update Daemon
```
python tools/auto_update.py [OPTIONS]

--interval N         Check interval in minutes (default: 2)
--max-hours N        Max forecast hour (default: 18)
--models M           Comma-separated models (default: hrrr) e.g. hrrr,gfs,rrfs
--once               Run once and exit
--no-cleanup         Don't clean up old data
```

## Dependencies

```
numpy, scipy, matplotlib, cfgrib, eccodes, flask, imageio, Pillow
```

Install with: `pip install -r requirements.txt`

For public access: `cloudflared` (Cloudflare Tunnel client)

## Credits

Produced by drewsny

Contributors: @jasonbweather, justincat66, Sequoiagrove, California Wildfire Tracking & others

## References

- [HRRR Model](https://rapidrefresh.noaa.gov/hrrr/) - NOAA's 3km CONUS model
- [GFS Model](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast) - NOAA's 0.25deg global model
- [RRFS Model](https://gsl.noaa.gov/focus-areas/unified_forecast_system) - NOAA's next-gen 3km CONUS model
- [eccodes](https://github.com/ecmwf/eccodes) - ECMWF GRIB library (default backend via `auto`)
- [cfgrib](https://github.com/ecmwf/cfgrib) - GRIB file reader (fallback backend)
