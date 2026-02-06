# CLAUDE.md — Agent Context for wxsection.com

## Project Overview

Multi-model atmospheric cross-section generator. Users draw a line on a map, get an instant vertical cross-section from HRRR, GFS, or RRFS weather model data. Live at **wxsection.com**. 19 visualization styles, all derived from cached fields.

## Architecture Summary

```
tools/unified_dashboard.py    — Flask server + Leaflet UI + all API endpoints (~5000 lines)
core/cross_section_interactive.py  — Rendering engine (matplotlib + cartopy, 0.5s warm)
tools/auto_update.py          — GRIB download daemon (slot-based concurrent)
model_config.py               — Model registry (HRRR/GFS/RRFS metadata, URLs, grids)
smart_hrrr/orchestrator.py    — Parallel GRIB download with callbacks
start.sh                      — Production startup (mount VHD, start services, cloudflared)
```

## Key Design Decisions

- **Mmap cache on NVMe**: GRIB files are converted to raw numpy arrays (2.3GB/FHR on disk, ~100MB resident RAM). This is what makes instant cross-sections possible.
- **eccodes `auto` backend**: Default GRIB backend tries eccodes direct (one-pass scan, ~35% faster), falls back to cfgrib. Configurable via `XSECT_GRIB_BACKEND` env var.
- **Lazy smoke loading**: wrfnat files (652MB, 50 hybrid levels) loaded on-demand on first `smoke` style request, not during preload. `ForecastHourData.grib_file` stores source path for deferred resolution.
- **Single-process, threaded**: WSL2 folio contention breaks ProcessPoolExecutor. Everything runs in one process with ThreadPoolExecutor.
- **Slot-based concurrent auto-update**: 3 HRRR + 1 GFS + 1 RRFS download slots in parallel via ThreadPoolExecutor. Each model has its own lane — slow RRFS can't block HRRR. HRRR fail-fast prunes unavailable FHRs.
- **Status file IPC**: auto_update.py writes `/tmp/auto_update_status.json` atomically; dashboard reads it to show download progress in the activity panel. No shared memory or sockets.
- **No handoff cycles**: Only one synoptic (48h) HRRR cycle, only one GFS/RRFS cycle. Previous cycles are evicted.
- **Two-tier NVMe eviction**: Rotated preload cycles always evicted. Archive request caches persist up to 670GB.

## Running Locally

```bash
cd ~/hrrr-maps && ./start.sh
# Or:
sudo mount /dev/sde /mnt/hrrr
python tools/auto_update.py --interval 2 --models hrrr,gfs,rrfs &
XSECT_GRIB_BACKEND=auto WXSECTION_KEY=cwtc python3 tools/unified_dashboard.py --port 5561 --models hrrr,gfs,rrfs
```

Logs: `/tmp/dashboard.log`, `/tmp/auto_update.log`, `/tmp/cloudflared.log`

## Environment

- **WSL2** on Windows, 32 cores, 118GB RAM
- **NVMe** (2TB VHD at `/`): code + mmap cache (`~/hrrr-maps/cache/xsect/`)
- **External VHD** (20TB at `/mnt/hrrr`): GRIB source files
- **Cloudflare Tunnel**: `cloudflared tunnel run wxsection` routes wxsection.com to localhost:5561

## Performance Characteristics

| Operation | Time |
|-----------|------|
| Warm render (cached FHR) | ~0.5s |
| Cached prerender (from frame cache) | ~20ms |
| GRIB-to-mmap conversion (eccodes) | ~15s/FHR HRRR, ~11s GFS, ~28s RRFS |
| Mmap load (cached on NVMe) | <0.1s |
| Cached preload (176 FHRs) | ~4s total |
| Parallel prerender (19 frames) | ~4s |
| HRRR FHR download (1.17GB) | ~170s |
| GFS FHR download (516MB) | ~83s |
| RRFS FHR download (795MB) | ~124s |

## Critical Constraints

1. **GRIB conversion is GIL-bound**: eccodes/cfgrib can't parallelize beyond 3-4 threads. Don't try ProcessPoolExecutor — folio contention on WSL2.
2. **matplotlib is not thread-safe**: The Agg backend works with ThreadPool but font cache can throw warnings under load. Non-fatal.
3. **Memory budget**: HRRR 48GB, GFS 8GB, RRFS 8GB. Mmap keeps resident small but monitor with `/api/status`.
4. **NVMe space**: 670GB cache limit enforced by `cache_evict_old_cycles()`. Monitor with `df -h /`.
5. **VHD must be mounted**: `/mnt/hrrr` needs `sudo mount /dev/sde /mnt/hrrr` after every WSL restart.
6. **Smoke loading is lazy**: wrfnat (652MB, 50 hybrid levels) only loaded on first `style=smoke` request, not during preload. Saves ~100s on startup.

## Key Constants

```python
# unified_dashboard.py
RENDER_SEMAPHORE = 12      # Max concurrent matplotlib renders
PRERENDER_WORKERS = 8      # Parallel prerender threads
PRELOAD_WORKERS = 20       # Cached mmap load threads (--preload-workers / XSECT_PRELOAD_WORKERS)
GRIB_WORKERS = 4           # GRIB conversion threads (--grib-workers / XSECT_GRIB_WORKERS)
CACHE_LIMIT_GB = 670       # NVMe cache size limit
HRRR_HOURLY_CYCLES = 3     # Non-synoptic cycles in preload window
GRIB_BACKEND = 'auto'      # XSECT_GRIB_BACKEND: auto (eccodes→cfgrib fallback), eccodes, cfgrib

# auto_update.py (slot-based concurrent)
HRRR_SLOTS = 3             # --hrrr-slots: concurrent HRRR downloads
GFS_SLOTS = 1              # --gfs-slots: concurrent GFS downloads
RRFS_SLOTS = 1             # --rrfs-slots: concurrent RRFS downloads
DISK_LIMIT_GB = 500        # GRIB source disk limit on VHD
STATUS_FILE = '/tmp/auto_update_status.json'  # IPC to dashboard
```

## API Quick Reference

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/cross-section` | Generate cross-section PNG (agent-friendly) |
| `GET /api/v1/products` | List available visualization styles |
| `GET /api/v1/cycles` | List available model cycles |
| `GET /api/v1/status` | Server health check |
| `GET /api/xsect` | Generate cross-section PNG (internal) |
| `GET /api/cycles` | List cycles with load status |
| `GET /api/status` | Memory/load status |
| `POST /api/load` | Load specific cycle + FHR |
| `POST /api/prerender` | Batch pre-render frames |

See [API_GUIDE.md](API_GUIDE.md) for full documentation.

## Common Tasks

### Restart dashboard
```bash
pkill -f unified_dashboard; sleep 2
XSECT_GRIB_BACKEND=auto WXSECTION_KEY=cwtc nohup python3 tools/unified_dashboard.py --port 5561 --models hrrr,gfs,rrfs > /tmp/dashboard.log 2>&1 &
```

### Restart auto-update
```bash
pkill -f auto_update; sleep 2
nohup python tools/auto_update.py --interval 2 --models hrrr,gfs,rrfs \
  --hrrr-slots 3 --gfs-slots 1 --rrfs-slots 1 > /tmp/auto_update.log 2>&1 &
```

### Check what's loaded
```bash
curl -s localhost:5561/api/cycles | python3 -m json.tool
curl -s localhost:5561/api/status | python3 -m json.tool
```

### Check NVMe usage
```bash
df -h /
du -sh ~/hrrr-maps/cache/xsect/*/
```

## Download Architecture

Auto-update uses slot-based concurrency (`run_download_pass_concurrent`):
- Each model gets dedicated ThreadPoolExecutor slots (3 HRRR + 1 GFS + 1 RRFS)
- Models download in parallel — slow RRFS can't block HRRR
- HRRR fail-fast: if an FHR isn't published, prunes higher FHRs from same cycle
- HRRR queue refreshes every 45s for newly published FHRs
- `download_forecast_hour` requires ALL file types to succeed (wrfprs + wrfsfc + wrfnat for HRRR)
- Writes progress to `/tmp/auto_update_status.json` on every schedule/completion event (atomically via tmp+rename)

**Availability lag**: HRRR 50min, GFS 180min (3h), RRFS 120min after init time. Set in `MODEL_AVAILABILITY_LAG`.

**NOMADS is the bottleneck**: ~6-7 MB/s per connection regardless of local bandwidth.
5 concurrent connections = ~265 Mbps sustained. Safe up to ~7-8 before throttling risk.

### Archive Downloads (dashboard `request_cycle`)
- Triggered via `/api/request_cycle` (admin-gated)
- Downloads from AWS archive (NOAA Big Data Program), not NOMADS
- Uses `download_gribs_parallel` with `max_threads=8`
- Progress shown in real-time via `/api/progress`
- Cancellable via `/api/cancel`
- Downloaded cycles tracked in `ARCHIVE_CACHE_KEYS` set for NVMe cache persistence

## Activity Panel (Progress Tracking)

Dashboard tracks operations via `PROGRESS` dict + `/api/progress` endpoint. Frontend polls every 1.5s.

**Operation types with visual identity:**
| Op Type | Icon | Color | Source |
|---------|------|-------|--------|
| `preload` | ▶ | indigo | Startup preload of target cycles |
| `autoload` | ▶ | light-indigo (#818cf8) | Background rescan auto-load (every 30-60s) |
| `load` | ↑ | default | Manual FHR load |
| `download` | ↓ | amber | Archive cycle download (admin-gated) |
| `prerender` | ● | purple | Batch frame rendering |
| `autoupdate` | ↻ | cyan (#06b6d4) | auto_update download progress (from status file) |

Auto-update progress is injected into `/api/progress` by reading `/tmp/auto_update_status.json`. Stale files (>5min) are ignored. Completed models are hidden.

## RAM Status Modal

Memory button shows all models (HRRR + GFS + RRFS) grouped by model. Each model section shows: cycle → forecast hours → ~RAM estimate. Fetches `/api/models` then `/api/status?model=X` for each registered model. Shows aggregate total at bottom.

## Known Performance Bottlenecks

1. **GRIB-to-mmap conversion (~15s/FHR)**: Still the #1 bottleneck. eccodes `auto` backend is ~35% faster than cfgrib but still GIL-bound. 3-4 workers optimal (knee at 3, regresses >4). See CONTEXT.md for detailed analysis.
2. **NOMADS per-connection speed (~6-7 MB/s)**: Main download bottleneck. More slots help linearly up to ~8 connections.
3. **Matplotlib render (0.5s)**: CPU-bound. Agg backend releases GIL for some C work but font/text rendering is Python.
4. **WSL2 VHD I/O**: All disk goes through Hyper-V virtual block layer. Native Linux would be faster.
5. **GFS data volume**: 65 FHRs x 6h intervals = F00-F384. Full cycle download is large.
