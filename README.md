# HRRR Cross-Section Dashboard

Interactive vertical cross-section visualization for HRRR weather model data. Draw lines on a map to generate real-time atmospheric profiles.

## Quick Start

```bash
# Basic usage - auto-discovers latest data
python tools/unified_dashboard.py --auto-update --port 5559

# Production mode with rate limiting
python tools/unified_dashboard.py --auto-update --port 5559 --production

# Specific data directory
python tools/unified_dashboard.py --data-dir outputs/hrrr/20251229/04z
```

Open `http://localhost:5559` in your browser.

## Features

### Split & Chip UI
- **Model Run Dropdown**: Select from available init cycles (grouped by date)
- **Forecast Hour Chips**: Click chips to toggle F00, F06, F12, F18
- **Max 2 Cycles**: Memory-efficient design limits to 2 loaded cycles at once
- **FIFO Behavior**: Loading a 3rd cycle automatically unloads the oldest

### 14 Cross-Section Styles
| Style | Description |
|-------|-------------|
| `wind_speed` | Horizontal wind speed (kt) with barbs |
| `temp` | Temperature (C) |
| `theta_e` | Equivalent potential temperature (K) |
| `rh` | Relative humidity (%) |
| `q` | Specific humidity (g/kg) |
| `omega` | Vertical velocity (Pa/s) - blue=rising, red=sinking |
| `vorticity` | Absolute vorticity |
| `shear` | Wind shear magnitude |
| `lapse_rate` | Temperature lapse rate (C/km) |
| `cloud` | Cloud water mixing ratio |
| `cloud_total` | Total condensate (cloud + ice) |
| `wetbulb` | Wet-bulb temperature |
| `icing` | Icing potential |
| `frontogenesis` | Petterssen kinematic frontogenesis ("Winter Bander") |

### Frontogenesis ("Winter Bander")
The frontogenesis style uses the Petterssen kinematic formula to identify regions of frontogenesis (banding potential) and frontolysis:

- **Red**: Frontogenesis (temperature gradient intensifying) - potential snow banding
- **Blue**: Frontolysis (temperature gradient weakening)
- Gaussian smoothing applied to reduce noise from 3km HRRR grid
- Units: K/100km/hr

### Smart Memory Management
- **Pre-loads**: 2 most recent COMPLETE cycles (all 4 FHRs available) at startup
- **On-demand**: Older cycles available but not loaded until requested
- **~1.7 GB per forecast hour**: 8 pre-loaded hours = ~14 GB RAM

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard UI |
| `/api/cycles` | GET | List available cycles with load status |
| `/api/status` | GET | Current memory/load status |
| `/api/load` | POST | Load a specific cycle+fhr |
| `/api/unload` | POST | Unload a specific cycle+fhr |
| `/api/load_cycle` | POST | Load all FHRs for a cycle |
| `/api/xsect` | GET | Generate cross-section image |
| `/api/info` | GET | Get point data for coordinates |

### Example: Generate Cross-Section

```
GET /api/xsect?start_lat=40.0&start_lon=-100.0&end_lat=35.0&end_lon=-90.0&style=frontogenesis&cycle=20251229_04z&fhr=6
```

## Command Line Options

```
--data-dir DIR       Specific data directory to load
--auto-update        Auto-discover and load latest available data
--port PORT          Server port (default: 5559)
--production         Enable rate limiting (60 req/min, 10 burst)
--max-hours N        Max forecast hours to download (for auto-update)
```

## Data Requirements

The dashboard expects HRRR pressure-level GRIB2 files in:
```
outputs/hrrr/{YYYYMMDD}/{HH}z/F{XX}/
```

Required fields for cross-sections:
- Temperature (TMP) on isobaric levels
- U/V wind components on isobaric levels
- Relative humidity (RH) on isobaric levels
- Geopotential height (HGT) on isobaric levels

The dashboard uses NPZ caching for fast subsequent loads (~10x faster than raw GRIB parsing).

## Architecture

```
tools/unified_dashboard.py    # Flask server + Leaflet UI
core/cross_section_interactive.py  # Cross-section engine
  - GRIB2 parsing with cfgrib
  - NPZ caching layer
  - Matplotlib rendering
  - Derived calculations (theta-e, frontogenesis, etc.)
```

## Branch Goals

This `focused/cross-section-dashboard` branch aims to:
1. Provide a standalone, focused cross-section visualization tool
2. Maintain compatibility with the core HRRR download/processing pipeline
3. Support winter weather analysis (frontogenesis, icing, wet-bulb)
4. Be memory-efficient for deployment on modest hardware

## Related

- Main HRRR processing: See `README.md`
- Setup guide: See `SETUP.md`
