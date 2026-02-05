# Cross-Section Dashboard Context

## Current State (Feb 5, 2026)

### Branch: `hrrr-maps-cross-section`

Focused repo for interactive HRRR cross-section visualization.
Live at **wxsection.com** via Cloudflare Tunnel (`cloudflared` running in WSL).

### What Works
- **Dashboard**: `tools/unified_dashboard.py` - Flask + Leaflet interactive map
- **Cross-section engine**: `core/cross_section_interactive.py` - Sub-second generation
- **NPZ caching**: `cache/dashboard/xsect/` (~2s per hour from cache vs 25s from GRIB, 400GB limit with eviction)
- **15 styles**: wind_speed, temp, theta_e, rh, q, omega, vorticity, shear, lapse_rate, cloud, cloud_total, wetbulb, icing, frontogenesis, smoke
- **Anomaly/departure mode**: "Raw / 5yr Dep" toggle subtracts 5-year HRRR climatological mean from current forecast. RdBu_r diverging colormap centered at 0. Works for 10 anomaly-eligible styles (temp, wind_speed, rh, omega, theta_e, q, vorticity, shear, lapse_rate, wetbulb). Requires pre-built climatology NPZ files
- **Climatology pipeline**: `tools/build_climatology.py` builds monthly mean NPZ files from archived HRRR data. Coarsened grid (every 5th point, 212x360 vs 1059x1799, ~15km). ~30MB per file. Nearest-FHR fallback (FHR 01 uses FHR 00's climo, FHR 02 uses FHR 03's, etc.)
- **Color-coded chip UI**: Grey (on disk) / Green (in RAM) / Blue (viewing) / Yellow (loading) / Shift+click to unload
- **Plot annotations**: A/B labels, ~100+ city labels, lat/lon secondary axis, legend box, inset map with A/B badges
- **Distance units**: km/mi toggle at render time
- **Every-3rd-hour preloading**: F00, F03, F06, F09, F12, F15, F18 loaded to RAM for latest 2 cycles (~52-62GB) via 2 parallel workers with startup progress bar. Newest cycle loads first. Other hours load on-demand. Loading mutex prevents preload/auto-load/Load All from overlapping
- **Auto-load**: Background thread (every 60s) detects new FHRs downloaded for latest 2 cycles and loads matching PRELOAD_FHRS into RAM automatically — no restart needed
- **Smart memory management**: 117GB hard cap, LRU eviction at 115GB. Latest 2 cycles are protected from eviction
- **Render semaphore**: Limits concurrent matplotlib renders to 4, returns 503 if queue full (10s timeout for single, 30s for GIF)
- **Disk management**: 500GB GRIB limit + 400GB NPZ cache limit with separate eviction
- **Auto-update daemon**: `tools/auto_update.py` downloads latest cycles continuously
- **Custom date requests**: Calendar button downloads any date from NOMADS/AWS with live progress toast (admin key required)
- **GRIB downloads**: `smart_hrrr/orchestrator.py` with parallel threading
- **Community favorites**: Save/load cross-section configs, 12h expiry
- **Cloudflare Tunnel**: Named tunnel `wxsection` → wxsection.com + www.wxsection.com. `cloudflared` binary at `/usr/local/bin/cloudflared`, config at `~/.cloudflared/config.yml`, tunnel ID `13c6556c-b8bb-4a81-8730-f57005819544`
- **GIF animation**: `/api/xsect_gif` generates animated GIF with Pillow (`disposal=2` for Discord). 4 speed options (0.25x/0.5x/0.75x/1x). Admin gets all loaded FHRs (up to 19 frames), regular users get every-3rd-hour only. Terrain + pressure levels locked to first frame for consistency
- **Terrain masking**: Theta contours, freezing level, and isotherms masked below terrain surface via `np.ma.masked_where`
- **Temperature colormaps**: 3 options (green_purple, white_zero, nws_ndfd) via `_build_temp_colormap()`, selectable in UI dropdown
- **Admin key system**: `WXSECTION_KEY` env var gates archive loading, downloading, unloading protected cycles, Load All button, and full-frame GIF. UI lock icon (🔒/🔓) with key stored in localStorage. Server-side `.strip()` on key to tolerate trailing whitespace
- **Load All button**: Admin-only button loads all available FHRs for current cycle. Waits up to 2 min if preload is running, then loads remaining FHRs
- **Protected cycles**: Latest 2 init cycles cannot be unloaded by regular users or evicted by LRU. Shifts automatically as new inits arrive
- **Download progress**: `/api/request_cycle` reports per-FHR completion via `on_complete` callback in `download_gribs_parallel`
- **VHD archive storage**: 20TB dynamic VHDX on external HDD mounted at `/mnt/hrrr` via `wsl --mount --vhd`. Bypasses 9p/DrvFS bottleneck (183 MB/s vs 19.7 MB/s writes). Holds historical GRIB archive + climatology NPZ files

### Files (16 Python files)
```
config/colormaps.py
core/__init__.py
core/cross_section_interactive.py   # Main engine - get_cross_section() + anomaly subtraction
core/cross_section_production.py    # Batch generation
core/downloader.py
core/grib_loader.py
model_config.py
smart_hrrr/__init__.py
smart_hrrr/availability.py
smart_hrrr/io.py
smart_hrrr/orchestrator.py          # Parallel GRIB downloads (on_complete callback)
smart_hrrr/utils.py
tools/auto_update.py
tools/build_climatology.py          # Build monthly climatology NPZ from archived HRRR
tools/bulk_download.py              # Bulk HRRR archive downloader for VHD/external drives
tools/unified_dashboard.py
```

### Key Architecture Details
- **Engine key system**: `_engine_key_map` maps `(cycle_key, fhr)` to auto-incrementing int for `self.forecast_hours` dict
- **Metadata passthrough**: Dashboard builds metadata dict (model, init_date, init_hour, real FHR) and passes directly to `get_cross_section(metadata=...)` — thread-safe, no shared state
- **Colorbar positioning**: Manual `fig.add_axes([0.90, 0.12, 0.012, 0.68])` with `cax=cbar_ax` (15 instances) - avoids `plt.colorbar(ax=ax)` stealing axes space
- **City labels**: `ax.secondary_xaxis(-0.08)` for aligned secondary axis, 120km search radius, `used_cities` set for dedup
- **Unit conversion**: `dist_scale = KM_TO_MI if use_miles else 1.0` applied to distances at render time, all internal math stays in km
- **Figure size**: `figsize=(17, 11)`, axes at `[0.06, 0.12, 0.82, 0.68]`
- **Terrain masking**: `terrain_mask` built from `pressure_levels > surface_pressure[i]`, applied to theta/temperature contours via `np.ma.masked_where`. Wind barbs also masked. contourf left unmasked (terrain fill zorder=5 covers it)
- **Temperature colormaps**: `_build_temp_colormap(name)` static method returns one of 3 colormaps. All defined as °F anchor arrays `(°F, (R,G,B))`, converted to °C internally. 512-bin `LinearSegmentedColormap`. Param threaded through `get_cross_section()` → `_render_cross_section()` as `temp_cmap`. Contour levels: `np.arange(-66, 56, 2)` — even numbers so 0°C is always a contour
- **GIF frame lock**: GIF endpoint extracts terrain + pressure levels from first FHR via `get_terrain_data()`, passes as `terrain_data` override (includes `pressure_levels`) to all subsequent frames. `ref_pressure_levels` param in `_render_cross_section()` locks y-axis limits and terrain fill across frames. Prevents jitter from surface pressure and level availability differences between FHRs
- **GIF rendering**: Uses Pillow with `disposal=2` (frame replace) for Discord compatibility. Speed dict: `{'1': 250, '0.75': 500, '0.5': 1000, '0.25': 2000}` ms. Admin GIF semaphore timeout 90s vs 30s regular
- **Render semaphore**: `threading.Semaphore(4)` wraps all matplotlib render calls. Single xsect gets 10s timeout, GIF gets 30s (admin: 90s). Returns 503 "Server busy" if semaphore can't be acquired
- **Admin key**: `WXSECTION_KEY` env var checked via `check_admin_key()` against `?key=` query param or `X-Admin-Key` header (with `.strip()`). Gates: `/api/load` (archive cycles), `/api/load_cycle` (archive), `/api/unload` (protected cycles), `/api/request_cycle` (all downloads), full-frame GIF, Load All button visibility. `/api/check_key` validates key and returns protected cycle list
- **Protected cycles**: `get_protected_cycles()` returns newest 2 cycle keys from `available_cycles`. Eviction skips these. Unload rejects non-admin requests for these
- **Auto-load**: `auto_load_latest()` called every 60s from background rescan thread. Checks latest 2 cycles for PRELOAD_FHRS on disk but not in RAM, loads with 2 parallel workers. Skips gracefully if `_loading` mutex is held by preload or load_cycle
- **Smoke loading**: Uses eccodes (not cfgrib) to read MASSDEN (disc=0, cat=20, num=0) from wrfnat — cfgrib can't identify this field (shows as `unknown`). Kept on **native hybrid levels** (50 levels, ~10-15 packed in lowest 2km) with per-column pressure coordinate — NOT interpolated to isobaric. This preserves boundary layer vertical detail where smoke concentrates. Stored as `smoke_hyb` + `smoke_pres_hyb` in ForecastHourData and NPZ cache. Units: kg/m³ × 10⁹ = μg/m³. Plotted with its own X/Y mesh (pressure varies per column due to terrain).
- **Smoke backfill**: When loading from NPZ cache, if `smoke_hyb` is missing but wrfnat file exists, smoke is automatically loaded from wrfnat and cache is updated. Handles transition from pre-smoke caches.
- **Stale cache validation**: On NPZ load, if pressure levels < 35 (expected 40), cache is discarded and reloaded from GRIB. Handles partially-downloaded GRIBs that got cached before full download completed.
- **Duplicate prevention**: All 4 `loaded_items.append()` sites (preload, auto-load, load_cycle, load_forecast_hour) check `if (cycle_key, fhr) not in self.loaded_items` before appending, preventing race condition duplicates
- **Loading mutex**: `self._loading` lock (threading.Lock) prevents preload, auto-load, and load_cycle from running simultaneously. Preload and load_cycle block on it; auto-load skips if locked. Prevents resource thrashing on startup
- **Bulk downloader**: `tools/bulk_download.py` downloads historical HRRR from AWS to VHD at `/mnt/hrrr`. Supports date ranges, configurable inits/FHRs, parallel threads, resume (skip-if-exists), dry run. Default: wrfprs + wrfsfc only (no smoke). HRRRv4 available from Dec 2020
- **Auto-update wrfnat awareness**: `auto_update.py` checks for wrfnat completeness — an FHR is only "downloaded" if wrfprs AND wrfnat both exist. Downloads wrfprs, wrfsfc, and wrfnat for all cycles.
- **Anomaly engine**: `ClimatologyData` dataclass holds coarsened climo arrays. `set_climatology_dir()` registers path. `get_climatology(month, init_hour, fhr)` loads + caches NPZ with nearest-FHR fallback. `_compute_anomaly()` interpolates coarse climo to cross-section path via `scipy.interpolate.RegularGridInterpolator`, subtracts from forecast fields. Returns anomaly array + `climo_info` dict (years, n_samples, month_name)
- **Anomaly rendering**: `_ANOMALY_LABELS` dict maps style → (colorbar_label, shading_label). `if anomaly and 'anomaly' in data:` block before style-specific shading. RdBu_r colormap, symmetric auto-scaling (98th percentile of |anomaly|), `np.linspace(-vmax, vmax, 41)` levels. Subtitle: "Departure from N-yr HRRR Mean (Month, n=samples)" in dark red italic
- **Anomaly UI**: Dashboard has "Mode: Raw / 5yr Dep" toggle group (hidden when style not in ANOMALY_STYLES or no climo available). JS fetches `/api/climatology_status` on load, `updateAnomalyVisibility()` shows/hides toggle. `&anomaly=1` appended to xsect and GIF URLs. `.anomaly-active` CSS class (orange)

### VHD Archive Infrastructure
- **VHDX file**: `D:\hrrr-archive.vhdx` (20TB dynamic, grows as data fills)
- **WSL mount**: `wsl --mount --vhd "D:\hrrr-archive.vhdx" --bare` from PowerShell (admin), then `sudo mount /dev/sde /mnt/hrrr` in WSL
- **Why VHD**: WSL2 9p/DrvFS protocol bottleneck — writing to `/mnt/d/` (Windows drive) measured at 19.7 MB/s. VHD bypasses 9p as direct SCSI device, achieving 183 MB/s (9x faster, near HDD max)
- **Thread tuning**: 12 threads downloading at 450 Mbps but only 19.7 MB/s write through 9p = wasted bandwidth buffering in RAM. On VHD, 4 threads is optimal (network-limited, not IO-limited)
- **Remount after WSL restart**: Required every time — PowerShell admin: `wsl --mount --vhd "D:\hrrr-archive.vhdx" --bare`, WSL: `sudo mount /dev/sde /mnt/hrrr`
- **Format notes**: ext4 with `lazy_itable_init=1,lazy_journal_init=1` to avoid freezing WSL during format of large filesystems (defers inode table zeroing)

### Key APIs
```python
# Interactive cross-section
from core.cross_section_interactive import InteractiveCrossSection
ixs = InteractiveCrossSection(cache_dir="cache/dashboard/xsect")
ixs.set_climatology_dir("/mnt/hrrr/climatology")
ixs.load_forecast_hour(data_dir, forecast_hour)
img_bytes = ixs.get_cross_section(start_point, end_point, style, forecast_hour, units='km', metadata={...}, anomaly=True)

# Parallel downloads
from smart_hrrr.orchestrator import download_gribs_parallel
download_gribs_parallel(model, date_str, cycle_hour, forecast_hours)

# Build climatology
# python tools/build_climatology.py --archive /mnt/hrrr --output /mnt/hrrr/climatology --month 2 --min-samples 3
```

### Running Dashboard
```bash
# Production (all 3 services)
export WXSECTION_KEY=your_secret
./deploy/run_production.sh        # start dashboard + auto-update + tunnel
./deploy/run_production.sh stop   # stop all

# Manual
WXSECTION_KEY=secret python tools/unified_dashboard.py --port 5559 --preload 2 --production
python tools/auto_update.py --interval 2 --max-hours 18
cloudflared tunnel run wxsection

# Remount VHD after WSL restart (PowerShell admin first)
# wsl --mount --vhd "D:\hrrr-archive.vhdx" --bare
sudo mount /dev/sde /mnt/hrrr
```

### Process Management
Startup order matters — dashboard must bind port 5559 before cloudflared starts forwarding:
1. Remount VHD if WSL restarted
2. Start dashboard (binds :5559)
3. Start cloudflared tunnel (forwards wxsection.com → localhost:5559)
4. Start auto_update daemon

Common issue: stale port after crash. Fix: `fuser -k -9 5559/tcp`, wait 3s, then restart dashboard.

Cloudflare tunnel config at `~/.cloudflared/config.yml`:
```yaml
tunnel: 13c6556c-b8bb-4a81-8730-f57005819544
credentials-file: /home/drew/.cloudflared/13c6556c-...json
ingress:
  - hostname: wxsection.com
    service: http://localhost:5559
  - hostname: www.wxsection.com
    service: http://localhost:5559
  - service: http_status:404
```

### Data Locations
- **Live GRIB files**: `outputs/hrrr/YYYYMMDD/HHz/F##/` (auto-update writes here, 500GB limit)
- **NPZ cache**: `cache/dashboard/xsect/` (400GB limit with eviction)
- **Archive GRIBs**: `/mnt/hrrr/YYYYMMDD/HHz/F##/` (VHD, bulk_download.py writes here)
- **Climatology NPZ**: `/mnt/hrrr/climatology/climo_MM_HHz_FNN.npz` (~30MB each)
- **Favorites**: `data/favorites.json`
- **Votes**: `data/votes.json`
- **Feature requests**: `data/requests.json`
- **Disk metadata**: `data/disk_meta.json`
