# Cross-Section Dashboard Context

## Current State (Feb 6, 2026)

### Branch: `hrrr-maps-cross-section`

Live at **wxsection.com** via Cloudflare Tunnel. WSL2 on Windows, 32 cores, 118GB RAM.

### Startup

```bash
cd ~/hrrr-maps && ./start.sh
# Mounts VHD, starts auto_update, dashboard (port 5561), cloudflared tunnel
```

Or manually:
```bash
sudo mount /dev/sde /mnt/hrrr
nohup python tools/auto_update.py --interval 2 --models hrrr,gfs,rrfs \
    --hrrr-slots 3 --gfs-slots 1 --rrfs-slots 1 > /tmp/auto_update.log 2>&1 &
XSECT_GRIB_BACKEND=auto WXSECTION_KEY=cwtc nohup python3 tools/unified_dashboard.py --port 5561 --models hrrr,gfs,rrfs > /tmp/dashboard.log 2>&1 &
nohup cloudflared tunnel run wxsection > /tmp/cloudflared.log 2>&1 &
```

---

## GRIB-to-mmap Conversion Performance

### Current State (~15s/FHR with eccodes, was ~23s with cfgrib)
- **Default backend switched to `auto`** (tries eccodes direct, falls back to cfgrib)
- eccodes one-pass scan is ~35% faster than cfgrib+xarray for HRRR
- **4 ThreadPool workers** remains the sweet spot (3-4 optimal, >4 regresses due to GIL)
- **Smoke loading is lazy** — wrfnat (652MB, 50 hybrid levels) only loaded on first smoke style request, not during preload
- Worker counts configurable at runtime: `--grib-workers N` / `--preload-workers N` (or env `XSECT_GRIB_WORKERS` / `XSECT_PRELOAD_WORKERS`)

### Measured Performance (this machine)
```
Single FHR GRIB-to-mmap (uncached):
  HRRR: ~15s (eccodes/auto) vs ~23s (cfgrib) vs ~33s (cfgrib, measured)
  GFS:  ~11s (eccodes) vs ~17s (cfgrib)
  RRFS: ~28s (eccodes) vs ~41s (cfgrib)

Scaling (HRRR, eccodes, 6 FHR batch wall time):
  1 worker: 184s, 2: 73s, 3: 67s, 4: 68s, 5: 79s, 6: 74s
  Knee at 3-4 workers; going higher regresses.

Cached preload (176 FHRs across HRRR+GFS+RRFS): ~4s total
  (was ~100s+ before lazy smoke — 26 wrfnat backfills blocked preload)
```

### What the conversion does (`load_forecast_hour` in cross_section_interactive.py)
```
1. Check mmap cache dir → if exists, load in <0.1s (done)
2. Check legacy .npz cache → if exists, migrate to mmap (rare)
3. Load from GRIB (the slow path, ~15s with eccodes):
   a. One-pass eccodes scan for 10 pressure-level fields (t, u, v, r, w, q, gh, absv, clwmr, dpt)
   b. Second scan for surface pressure
   c. Compute derived fields (theta via vectorized numpy, temp_c)
   d. Save all fields to mmap cache (14 fields x 40 levels x 1059x1799)
   e. Smoke PM2.5 NOT loaded here — deferred to first smoke render request
```

### Lazy Smoke Loading
- wrfnat files are 652MB each with 50 hybrid levels — expensive to decode
- Previously loaded during every mmap cache load, causing 99-104s stalls when 20 threads hit VHD concurrently
- Now loaded on-demand: first `style=smoke` request triggers `_backfill_smoke()`, which loads wrfnat, saves smoke_hyb.npy to mmap cache
- Subsequent loads pick up smoke from mmap cache automatically (included in `_FLOAT16_FIELDS`)
- `ForecastHourData.grib_file` field stores source path for lazy resolution

### WSL2 VHD Folio Contention (blocks ProcessPoolExecutor)
- Both `/dev/sdd` (NVMe VHD) and `/dev/sde` (external VHD) go through Hyper-V virtual block layer
- Multiple *processes* doing concurrent I/O trigger `folio_wait_bit_common` in WSL2 page cache locking
- ThreadPoolExecutor (single process) avoids folio; ProcessPoolExecutor triggers it
- This is NOT disk speed — it's WSL2-specific kernel contention

### Remaining optimization paths
1. **Pre-extract in auto_update** — convert GRIB→mmap immediately after download, before dashboard needs it
2. **Rust/C GRIB reader** (e.g., wgrib2 subprocess) — bypass Python GIL entirely
3. **Per-field parallelism** — split the eccodes scan across threads (risky with GIL)
4. **Lazy field loading** — only decode fields needed for the requested style (skip cloud for temp, etc.)

---

## Architecture

### Preload Window (5 HRRR cycles, ~125 FHRs)
```
Priority order:
  1. Latest init (e.g. 14z) — 19 FHRs, always first
  2. Newest synoptic 48h (e.g. 12z) — 49 FHRs
  3. 3 recent hourlies (e.g. 13z, 11z, 10z) — 19 FHRs each

No previous synoptic handoff — only one 48h cycle kept at a time.
HRRR_HOURLY_CYCLES = 3
```

Only these cycles appear in the run picker dropdown. Archive-requested cycles appear when downloading/loaded.

### Two-Phase Preload (cache-first)
```
Phase 1: ThreadPoolExecutor(20 workers) — cached mmap loads (<0.1s each, no smoke)
Phase 2: ThreadPoolExecutor(4 workers)  — GRIB conversion (~15s each with eccodes)
```
Partitions FHRs by checking if mmap cache dir exists. Users see data immediately from Phase 1 instead of waiting for all GRIB conversions.

### Cache Eviction (RAM + NVMe disk)
```
RAM eviction:
  - _auto_load_latest_inner(): evicts loaded FHRs whose cycle is no longer in target window
  - _evict_if_needed(): memory-pressure backstop at 48GB HRRR / 8GB GFS,RRFS limits
  - Protected cycles (current target window) are never evicted from RAM

NVMe cache eviction — two-tier (cache_evict_old_cycles):
  Tier 1 (always, every ~10 min):
    - Rotated preload cycles: if a cycle falls out of target window and
      wasn't an archive request, its cache is deleted immediately
    - Example: 03z hourly rotates out when 09z appears → 03z cache deleted
  Tier 2 (size-based, CACHE_LIMIT_GB = 670):
    - Archive request caches persist on NVMe for fast re-loading
    - Only evicted when total cache exceeds 670GB, oldest archive first
    - Evicts down to 85% (~570GB) to avoid thrashing
  - ARCHIVE_CACHE_KEYS set tracks which cycles were archive-requested
  - Target/loaded cycles are never evicted at either tier
  - Budget: ~425GB preload window + ~245GB archive headroom
```

### Auto-Update (Slot-Based Concurrent)
```
Slot-based concurrency via ThreadPoolExecutor + wait(FIRST_COMPLETED):
  --hrrr-slots 3   (3 HRRR FHRs downloading in parallel)
  --gfs-slots 1    (1 GFS FHR at a time)
  --rrfs-slots 1   (1 RRFS FHR at a time)
  Total: 5 concurrent NOMADS/AWS connections

Each model runs in its own lane — slow RRFS can't block HRRR.
HRRR fail-fast: if Fxx fails, prunes higher FHRs from same cycle.
HRRR refresh: re-scans for newly published FHRs every 45s while
  other models are still downloading.

Status file: auto_update writes /tmp/auto_update_status.json with per-model
  progress (cycle, total, done, in_flight FHRs). Dashboard reads this to
  show download progress in the activity panel.

Availability lag (minutes after init before checking):
  HRRR: 50min, GFS: 180min (3h), RRFS: 120min (2h)

Cycle targeting:
  - HRRR: latest 2 cycles (for base FHRs)
  - GFS/RRFS: latest cycle only (no handoff)

Download throughput (~6-7 MB/s per NOMADS connection):
  HRRR: 1.17GB/FHR (wrfprs 375MB + wrfsfc 138MB + wrfnat 652MB) ~170s/FHR
  GFS:  516MB/FHR (pgrb2.0p25)                                   ~83s/FHR
  RRFS: 795MB/FHR (prslev)                                       ~124s/FHR
  Total bandwidth at 3+1+1: ~265 Mbps sustained

Bottleneck: NOMADS per-connection speed (~6-7 MB/s), not local bandwidth.
Safe to bump to ~7-8 total connections before NOMADS may throttle.
```

### Activity Panel (Progress Tracking)
```
Dashboard tracks operations via PROGRESS dict + /api/progress endpoint.
Frontend polls every 1.5s.

Operation types:
  preload    (▶ indigo)   — startup preload of target cycles
  autoload   (▶ lt-indigo)— background rescan auto-load (triggered every 30-60s)
  load       (↑ default)  — manual FHR load
  download   (↓ amber)    — archive cycle download (admin-gated)
  prerender  (● purple)   — batch frame rendering
  autoupdate (↻ cyan)     — auto_update download progress (read from status file)

Auto-update progress is injected into /api/progress by reading
  /tmp/auto_update_status.json (written atomically by auto_update.py).
  Stale files (>5min old) are ignored. Completed models are hidden.
```

### RAM Status Modal
```
Memory button shows all models (HRRR + GFS + RRFS) grouped by model.
Each model section shows: cycle → forecast hours → ~RAM estimate.
Fetches /api/status?model=X for each registered model.
```

### Memory Architecture
- **Mmap cache per FHR**: 2.3GB on disk (40 levels x 1059 x 1799 x 14 float16/32 fields)
- **Resident RAM per FHR**: ~100MB (mmap only pages in accessed slices)
- **~125 FHRs loaded**: ~4-6GB RAM, ~290GB on disk
- **Heap per FHR**: ~29MB (lats+lons coordinate arrays)
- **Memory limits**: 48GB HRRR hard cap, 8GB each GFS/RRFS

### Disk Layout
```
/dev/sdd (2TB NVMe VHD) mounted at /  — 1.4TB free
  ~/hrrr-maps/                         — code
  ~/hrrr-maps/cache/xsect/hrrr/       — ACTIVE mmap cache (NVMe, building up)
  ~/hrrr-maps/cache/xsect/gfs/        — GFS cache (NVMe)
  ~/hrrr-maps/outputs/                 — symlinks to VHD

/dev/sde (20TB external VHD) mounted at /mnt/hrrr  — 17TB free
  /mnt/hrrr/hrrr-live/                 — live HRRR GRIBs (491GB)
  /mnt/hrrr/gfs/                       — live GFS GRIBs (113GB)
  /mnt/hrrr/rrfs/                      — live RRFS GRIBs (83GB)
  /mnt/hrrr/cache/xsect/              — OLD VHD cache (~348GB, orphaned, can delete)
  /mnt/hrrr/YYYYMMDD/                 — archived HRRR GRIBs
  /mnt/hrrr/climatology/              — monthly mean NPZ files

/dev/shm (59GB tmpfs, pure RAM)       — unused, too small for cache
```

### Per-FHR Sizes
```
HRRR GRIB source:  ~1.2GB (wrfprs + wrfsfc + wrfnat)
HRRR mmap cache:   ~2.3GB (14 fields x 40 levels x 1059x1799)
HRRR cycle (19 FHR): ~23GB GRIB, ~44GB cache
HRRR synoptic (49 FHR): ~61GB GRIB, ~113GB cache
```

---

## Features

### What Works
- **Dashboard**: `tools/unified_dashboard.py` — Flask + Leaflet (OpenTopoMap), live at wxsection.com:5561
- **Cross-section engine**: `core/cross_section_interactive.py` — 0.5s warm renders
- **Multi-model**: HRRR, GFS, RRFS support everywhere
- **19 styles**: wind_speed, temp, theta_e, rh, q, omega, vorticity, shear, lapse_rate, cloud, cloud_total, wetbulb, icing, frontogenesis, smoke, vpd, dewpoint_dep, moisture_transport, pv
- **Run picker**: Filtered to preload window + loaded archive cycles only
- **Archive requests**: Modal with date picker, hour selector, FHR range (admin-gated)
- **Activity panel**: Real-time progress for all operations (preload, auto-load, download, prerender, auto-update)
- **Auto-update progress**: Shows per-model download status (HRRR/GFS/RRFS) via status file IPC
- **Cancel jobs**: Admin can cancel pre-render and download operations via X button in progress panel
- **Auto-update**: Slot-based concurrent (3 HRRR + 1 GFS + 1 RRFS in parallel), fail-fast on unavailable, HRRR refresh every 45s
- **Cache-first preload**: Cached FHRs load in <1s total, GRIB conversions run in background
- **Time slider + auto-play**: 0.5x-4x speed, pre-render for instant scrubbing
- **Frame prerender cache**: 500-entry server-side cache
- **Cycle comparison mode**: Side-by-side Same FHR or Valid Time matching
- **Community favorites**: Save/load cross-section configs
- **GIF animation**: `/api/xsect_gif`
- **RAM status modal**: Shows all models (HRRR + GFS + RRFS) with per-cycle FHR counts and memory
- **Admin key**: `WXSECTION_KEY=cwtc` env var gates archive downloads, cancel ops

### Controls UI
- **Primary row**: Model, Run, Style, Favorites, Swap, Clear, "More" toggle
- **Secondary row** (hidden by default): Y-Axis, V-Scale, Top, Units, Help, Request Run, GIF, Load All, Compare, Admin, Memory

---

## Files

```
start.sh                           # Startup script (mount VHD, start services)
model_config.py                     # Model registry (HRRR/GFS/RRFS metadata)
core/cross_section_interactive.py   # Main engine — mmap cache, cartopy cache, KDTree cache
smart_hrrr/orchestrator.py          # Parallel GRIB downloads (on_complete, on_start, should_cancel)
tools/auto_update.py                # Slot-based concurrent download daemon + status file
tools/unified_dashboard.py          # Flask dashboard — everything UI + API
```

### Key Constants (unified_dashboard.py)
```python
PRELOAD_WORKERS = 20   # Thread workers for cached mmap loads
GRIB_WORKERS = 4       # Thread workers for GRIB-to-mmap conversion (THE BOTTLENECK)
CACHE_BASE = '/home/drew/hrrr-maps/cache/xsect'  # NVMe — fast local storage
CACHE_LIMIT_GB = 670   # ~290GB preload + ~380GB archive headroom
RENDER_SEMAPHORE = 12  # 8 prerender + 4 live user requests
PRERENDER_WORKERS = 8  # Parallel threads for batch prerender
HRRR_HOURLY_CYCLES = 3

# auto_update.py (slot-based concurrent)
HRRR_SLOTS = 3         # --hrrr-slots: concurrent HRRR downloads
GFS_SLOTS = 1          # --gfs-slots: concurrent GFS downloads
RRFS_SLOTS = 1         # --rrfs-slots: concurrent RRFS downloads
STATUS_FILE = '/tmp/auto_update_status.json'  # IPC to dashboard
```

---

## Known Issues / TODO

1. **Delete orphaned VHD cache**: `rm -rf /mnt/hrrr/cache/xsect/` frees ~348GB on VHD (no longer used after NVMe migration)
2. **GRIB-to-mmap is the #1 bottleneck**: ~15s/FHR with eccodes, 4 threads, GIL-bound. See "GRIB-to-mmap Conversion Performance" section above.
3. **ProcessPoolExecutor broken on WSL2**: folio contention, all workers D-state. Would need native Linux or different GRIB library
4. **GFS/RRFS rendering**: Works but needs more testing at extended FHRs
5. **VHD remount required**: After every WSL/PC restart, run `start.sh` or mount manually
6. **Background rescan frequency**: HRRR every 30s, others every 60s — could be tunable
7. **Monitor NVMe space**: Full preload cache = ~400GB, eviction keeps it bounded. Monitor with `df -h /`
