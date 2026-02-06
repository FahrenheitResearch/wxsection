#!/usr/bin/env python3
"""
HRRR Cross-Section Dashboard

Interactive cross-section visualization on a Leaflet map.
Draw a line on the map to generate vertical cross-sections.

Usage:
    python tools/unified_dashboard.py --data-dir outputs/hrrr/20251227/09z
    python tools/unified_dashboard.py --auto-update
"""

import argparse
import json
import logging
import os
import sys
import time
import io
import threading
from pathlib import Path
from datetime import datetime
from functools import wraps
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio.v2 as imageio
from PIL import Image

from flask import Flask, jsonify, request, send_file, abort

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

VOTES_FILE = Path(__file__).parent.parent / 'data' / 'votes.json'
REQUESTS_FILE = Path(__file__).parent.parent / 'data' / 'requests.json'
FAVORITES_FILE = Path(__file__).parent.parent / 'data' / 'favorites.json'
DISK_META_FILE = Path(__file__).parent.parent / 'data' / 'disk_meta.json'
DISK_LIMIT_GB = 500  # Max disk usage for HRRR data (GRIB source on VHD)
CACHE_LIMIT_GB = 670  # Max NVMe cache (~425GB preload + ~245GB for archive requests)
CLIMATOLOGY_DIR = Path('/mnt/hrrr/climatology')

# Styles that support anomaly mode (must match ANOMALY_STYLES in cross_section_interactive.py)
ANOMALY_STYLES = {
    'temp', 'wind_speed', 'rh', 'omega', 'theta_e',
    'q', 'vorticity', 'shear', 'lapse_rate', 'wetbulb',
}

# --- v1 API: product name mapping and metadata ---
PRODUCT_TO_STYLE = {
    'temperature': 'temp', 'temp': 'temp',
    'wind_speed': 'wind_speed', 'wind': 'wind_speed',
    'theta_e': 'theta_e',
    'relative_humidity': 'rh', 'rh': 'rh',
    'vertical_velocity': 'omega', 'omega': 'omega',
    'specific_humidity': 'q', 'q': 'q',
    'vorticity': 'vorticity',
    'wind_shear': 'shear', 'shear': 'shear',
    'lapse_rate': 'lapse_rate',
    'cloud': 'cloud_total', 'cloud_total': 'cloud_total',
    'wetbulb': 'wetbulb', 'wet_bulb': 'wetbulb',
    'icing': 'icing',
    'frontogenesis': 'frontogenesis',
    'smoke': 'smoke',
    'vpd': 'vpd', 'vapor_pressure_deficit': 'vpd',
    'dewpoint_dep': 'dewpoint_dep', 'dewpoint_depression': 'dewpoint_dep',
    'moisture_transport': 'moisture_transport',
    'pv': 'pv', 'potential_vorticity': 'pv',
}
PRODUCTS_INFO = [
    {'id': 'temperature', 'name': 'Temperature', 'units': '\u00b0C'},
    {'id': 'wind_speed', 'name': 'Wind Speed', 'units': 'knots'},
    {'id': 'theta_e', 'name': 'Equivalent Potential Temperature', 'units': 'K'},
    {'id': 'rh', 'name': 'Relative Humidity', 'units': '%'},
    {'id': 'omega', 'name': 'Vertical Velocity', 'units': 'hPa/hr'},
    {'id': 'q', 'name': 'Specific Humidity', 'units': 'g/kg'},
    {'id': 'vorticity', 'name': 'Absolute Vorticity', 'units': '10\u207b\u2075 s\u207b\u00b9'},
    {'id': 'shear', 'name': 'Wind Shear', 'units': '10\u207b\u00b3 s\u207b\u00b9'},
    {'id': 'lapse_rate', 'name': 'Lapse Rate', 'units': '\u00b0C/km'},
    {'id': 'cloud_total', 'name': 'Cloud Total Condensate', 'units': 'g/kg'},
    {'id': 'wetbulb', 'name': 'Wet-Bulb Temperature', 'units': '\u00b0C'},
    {'id': 'icing', 'name': 'Icing Potential', 'units': 'g/kg'},
    {'id': 'frontogenesis', 'name': 'Frontogenesis', 'units': 'K/100km/3hr'},
    {'id': 'smoke', 'name': 'PM2.5 Smoke', 'units': '\u03bcg/m\u00b3'},
    {'id': 'vpd', 'name': 'Vapor Pressure Deficit', 'units': 'hPa'},
    {'id': 'dewpoint_dep', 'name': 'Dewpoint Depression', 'units': '\u00b0C'},
    {'id': 'moisture_transport', 'name': 'Moisture Transport', 'units': 'g\u00b7m/kg/s'},
    {'id': 'pv', 'name': 'Potential Vorticity', 'units': 'PVU'},
]

def load_votes():
    """Load votes from JSON file."""
    if VOTES_FILE.exists():
        try:
            with open(VOTES_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_votes(votes):
    """Save votes to JSON file."""
    VOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VOTES_FILE, 'w') as f:
        json.dump(votes, f, indent=2)

def load_requests():
    """Load feature requests from JSON file."""
    if REQUESTS_FILE.exists():
        try:
            with open(REQUESTS_FILE) as f:
                return json.load(f)
        except:
            return []
    return []

def save_request(name, text):
    """Save a new feature request."""
    REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    requests = load_requests()
    requests.append({
        'name': name,
        'text': text,
        'timestamp': datetime.now().isoformat()
    })
    with open(REQUESTS_FILE, 'w') as f:
        json.dump(requests, f, indent=2)

def load_favorites():
    """Load community favorites from JSON file."""
    if FAVORITES_FILE.exists():
        try:
            with open(FAVORITES_FILE) as f:
                favorites = json.load(f)
            # Clean up old favorites (>12 hours) but keep at least the name
            now = datetime.now()
            cleaned = []
            for fav in favorites:
                try:
                    created = datetime.fromisoformat(fav.get('created', ''))
                    age_hours = (now - created).total_seconds() / 3600
                    if age_hours < 12 or fav.get('permanent', False):
                        cleaned.append(fav)
                except:
                    cleaned.append(fav)  # Keep if can't parse date
            return cleaned
        except:
            return []
    return []

def save_favorite(name, config, label=''):
    """Save a new community favorite."""
    FAVORITES_FILE.parent.mkdir(parents=True, exist_ok=True)
    favorites = load_favorites()
    # Generate unique ID
    import hashlib
    fav_id = hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    favorites.append({
        'id': fav_id,
        'name': name,
        'label': label,
        'config': config,
        'created': datetime.now().isoformat()
    })
    with open(FAVORITES_FILE, 'w') as f:
        json.dump(favorites, f, indent=2)
    return fav_id

def delete_favorite(fav_id):
    """Delete a community favorite by ID."""
    favorites = load_favorites()
    favorites = [f for f in favorites if f.get('id') != fav_id]
    with open(FAVORITES_FILE, 'w') as f:
        json.dump(favorites, f, indent=2)

def load_disk_meta():
    """Load disk metadata (last-accessed times, request source)."""
    if DISK_META_FILE.exists():
        try:
            with open(DISK_META_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_disk_meta(meta):
    """Save disk metadata."""
    DISK_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DISK_META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)

def touch_cycle_access(cycle_key):
    """Mark a cycle as recently accessed (for popularity tracking)."""
    meta = load_disk_meta()
    if cycle_key not in meta:
        meta[cycle_key] = {}
    meta[cycle_key]['last_accessed'] = time.time()
    meta[cycle_key]['access_count'] = meta[cycle_key].get('access_count', 0) + 1
    save_disk_meta(meta)

def get_disk_usage_gb():
    """Get total disk usage of HRRR data directory in GB."""
    base = Path("outputs/hrrr")
    if not base.exists():
        return 0
    total = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
    return total / (1024 ** 3)

def disk_evict_least_popular(target_gb=None):
    """Evict least-recently-accessed cycles from disk until under target_gb.

    Never evicts cycles accessed in the last 2 hours (likely still in use).
    """
    import shutil
    if target_gb is None:
        target_gb = DISK_LIMIT_GB * 0.85  # Evict down to 85% of limit

    base = Path("outputs/hrrr")
    if not base.exists():
        return

    usage = get_disk_usage_gb()
    if usage <= target_gb:
        return

    meta = load_disk_meta()
    now = time.time()
    recent_cutoff = now - 7200  # Don't evict anything accessed in last 2 hours

    # Build list of all cycles on disk with their last access time
    disk_cycles = []
    for date_dir in base.iterdir():
        if not date_dir.is_dir() or not date_dir.name.isdigit():
            continue
        for hour_dir in date_dir.iterdir():
            if not hour_dir.is_dir() or not hour_dir.name.endswith('z'):
                continue
            hour = hour_dir.name.replace('z', '')
            cycle_key = f"{date_dir.name}_{hour}z"
            last_access = meta.get(cycle_key, {}).get('last_accessed', 0)
            if last_access > recent_cutoff:
                continue  # Skip recently used
            disk_cycles.append((last_access, cycle_key, hour_dir))

    # Sort by last access (oldest first = evict first)
    disk_cycles.sort()

    for last_access, cycle_key, hour_dir in disk_cycles:
        if get_disk_usage_gb() <= target_gb:
            break
        logger.info(f"Disk evict: {cycle_key} (last accessed {int((now - last_access)/3600)}h ago)")
        try:
            shutil.rmtree(hour_dir)
            # Clean up empty parent date dir
            parent = hour_dir.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            # Remove from meta
            meta.pop(cycle_key, None)
        except Exception as e:
            logger.warning(f"Failed to evict {cycle_key}: {e}")

    save_disk_meta(meta)


def get_cache_usage_gb(managers: dict) -> float:
    """Get total NVMe cache usage across all models in GB."""
    total = 0
    for model_name, mgr in managers.items():
        cache_dir = Path(mgr.CACHE_BASE) / model_name
        if cache_dir.exists():
            total += sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def _evict_cache_dirs(dirs, label):
    """Delete a list of cache directories."""
    import shutil
    for d in dirs:
        try:
            shutil.rmtree(d)
        except Exception as e:
            logger.warning(f"Cache evict failed for {d.name}: {e}")
    total_gb = len(dirs) * 2.3
    logger.info(f"Cache evict: {label} — removed {len(dirs)} FHRs (~{total_gb:.0f}GB)")


def cache_evict_old_cycles(managers: dict):
    """Two-tier NVMe cache eviction.

    Tier 1 (always): Rotated preload cycles — if a cycle falls out of the target
    window and wasn't an archive request, delete its cache immediately regardless
    of disk space.

    Tier 2 (size-based): Archive request caches — only evict when total cache
    exceeds CACHE_LIMIT_GB (670GB). Oldest archive caches go first.
    """
    import re

    for model_name, mgr in managers.items():
        cache_dir = Path(mgr.CACHE_BASE) / model_name
        if not cache_dir.exists():
            continue

        # Build key sets
        target_keys = {c['cycle_key'] for c in mgr._get_target_cycles()}
        with mgr._lock:
            loaded_keys = {ck for ck, _ in mgr.loaded_items}

        # Scan cache dirs and group by cycle key (format: YYYYMMDD_HHz_F##_...)
        cycle_dirs = {}
        for entry in cache_dir.iterdir():
            if not entry.is_dir():
                continue
            m = re.match(r'(\d{8}_\d{2}z)_', entry.name)
            if m:
                ck = m.group(1)
                cycle_dirs.setdefault(ck, []).append(entry)

        # Tier 1: Always evict rotated preload cycles (not target, not loaded, not archive)
        for ck, dirs in cycle_dirs.items():
            if ck in target_keys or ck in loaded_keys or ck in ARCHIVE_CACHE_KEYS:
                continue
            _evict_cache_dirs(dirs, f"{model_name} {ck} (rotated out)")

    # Tier 2: Size-based eviction of archive caches
    usage_gb = get_cache_usage_gb(managers)
    if usage_gb <= CACHE_LIMIT_GB:
        return

    logger.info(f"Cache usage {usage_gb:.0f}GB > {CACHE_LIMIT_GB}GB limit, evicting archive caches...")
    import re
    target_gb = CACHE_LIMIT_GB * 0.85

    # Collect all evictable archive caches across models, sorted oldest first
    evictable = []
    for model_name, mgr in managers.items():
        cache_dir = Path(mgr.CACHE_BASE) / model_name
        if not cache_dir.exists():
            continue
        target_keys = {c['cycle_key'] for c in mgr._get_target_cycles()}
        with mgr._lock:
            loaded_keys = {ck for ck, _ in mgr.loaded_items}

        for entry in cache_dir.iterdir():
            if not entry.is_dir():
                continue
            m = re.match(r'(\d{8}_\d{2}z)_', entry.name)
            if not m:
                continue
            ck = m.group(1)
            if ck in target_keys or ck in loaded_keys:
                continue
            evictable.append((ck, entry, model_name))

    # Sort by cycle key (oldest first)
    evictable.sort(key=lambda x: x[0])

    removed_keys = set()
    for ck, d, model_name in evictable:
        if get_cache_usage_gb(managers) <= target_gb:
            break
        try:
            import shutil
            shutil.rmtree(d)
        except Exception as e:
            logger.warning(f"Cache evict failed for {d.name}: {e}")
        if ck not in removed_keys:
            logger.info(f"Cache evict: {model_name} {ck} (archive, over size limit)")
            removed_keys.add(ck)
    # Clean up ARCHIVE_CACHE_KEYS for fully evicted cycles
    ARCHIVE_CACHE_KEYS.difference_update(removed_keys)


CONUS_BOUNDS = {
    'south': 21.14, 'north': 52.62,
    'west': -134.10, 'east': -60.92,
}

XSECT_STYLES = [
    ('wind_speed', 'Wind Speed'),
    ('temp', 'Temperature'),
    ('theta_e', 'Theta-E'),
    ('rh', 'Relative Humidity'),
    ('q', 'Specific Humidity'),
    ('omega', 'Vertical Velocity'),
    ('vorticity', 'Vorticity'),
    ('shear', 'Wind Shear'),
    ('lapse_rate', 'Lapse Rate'),
    ('cloud', 'Cloud Water'),
    ('cloud_total', 'Total Condensate'),
    ('wetbulb', 'Wet-Bulb Temp'),
    ('icing', 'Icing Potential'),
    ('frontogenesis', '❄ Frontogenesis'),  # Winter Bander mode
    ('smoke', 'PM2.5 Smoke'),
    ('vpd', 'Vapor Pressure Deficit'),
    ('dewpoint_dep', 'Dewpoint Depression'),
    ('moisture_transport', 'Moisture Transport'),
    ('pv', 'Potential Vorticity'),
]

# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    def __init__(self, rpm=60, burst=10):
        self.rpm, self.burst = rpm, burst
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, ip):
        now = time.time()
        with self.lock:
            self.requests[ip] = [t for t in self.requests[ip] if t > now - 60]
            if len(self.requests[ip]) >= self.rpm:
                return False
            if len([t for t in self.requests[ip] if t > now - 1]) >= self.burst:
                return False
            self.requests[ip].append(now)
            return True

rate_limiter = RateLimiter()

# Limit concurrent matplotlib renders to prevent CPU/memory thrash under load
# 12 = up to 8 prerender workers + 4 live user requests
RENDER_SEMAPHORE = threading.Semaphore(12)
PRERENDER_WORKERS = 8  # Parallel threads for batch prerender

# =============================================================================
# FRAME PRERENDER CACHE — stores rendered PNG bytes for slider/comparison
# =============================================================================
FRAME_CACHE = {}            # cache_key -> PNG bytes
FRAME_CACHE_LOCK = threading.Lock()
MAX_FRAME_CACHE = 500       # ~500 * 150KB = ~75MB max

def frame_cache_key(model, cycle_key, fhr, style, start, end, y_axis, vscale, y_top, units, temp_cmap, anomaly):
    """Deterministic cache key for a rendered frame."""
    return f"{model}:{cycle_key}:F{fhr:02d}:{style}:{start[0]:.4f},{start[1]:.4f}:{end[0]:.4f},{end[1]:.4f}:{y_axis}:{vscale}:{y_top}:{units}:{temp_cmap}:{anomaly}"

def frame_cache_put(key, png_bytes):
    """Store a rendered frame, evicting oldest if full."""
    with FRAME_CACHE_LOCK:
        FRAME_CACHE[key] = png_bytes
        while len(FRAME_CACHE) > MAX_FRAME_CACHE:
            oldest = next(iter(FRAME_CACHE))
            del FRAME_CACHE[oldest]

def frame_cache_get(key):
    """Retrieve cached frame or None."""
    with FRAME_CACHE_LOCK:
        return FRAME_CACHE.get(key)

# Admin key for archive access — set via WXSECTION_KEY env var
ADMIN_KEY = os.environ.get('WXSECTION_KEY', '')

def check_admin_key():
    """Check if request has valid admin key (query param or header)."""
    key = (request.args.get('key', '') or request.headers.get('X-Admin-Key', '')).strip()
    return bool(ADMIN_KEY) and key == ADMIN_KEY

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if app.config.get('PRODUCTION'):
            if not rate_limiter.is_allowed(request.remote_addr):
                return jsonify({'error': 'Rate limit exceeded'}), 429
        return f(*args, **kwargs)
    return decorated

# =============================================================================
# PROGRESS TRACKING
# =============================================================================

PROGRESS = {}  # Global progress dict: op_id -> {op, label, step, total, detail, started, done, done_at, ...}

def progress_update(op_id, step, total, detail, label=None):
    """Update progress for an operation."""
    now = time.time()
    if op_id not in PROGRESS:
        PROGRESS[op_id] = {
            'op': op_id.split(':')[0],
            'label': label or op_id,
            'step': step,
            'total': total,
            'detail': detail,
            'started': now,
            'done': False,
            'done_at': None,
            'last_step_at': now,
            'rate_history': [],  # (timestamp, step) for rate estimation
        }
    else:
        prev_step = PROGRESS[op_id]['step']
        PROGRESS[op_id]['step'] = step
        PROGRESS[op_id]['total'] = total
        PROGRESS[op_id]['detail'] = detail
        if label:
            PROGRESS[op_id]['label'] = label
        # Track rate for ETA
        if step > prev_step:
            PROGRESS[op_id]['last_step_at'] = now
            hist = PROGRESS[op_id]['rate_history']
            hist.append((now, step))
            # Keep last 20 data points
            if len(hist) > 20:
                PROGRESS[op_id]['rate_history'] = hist[-20:]

def progress_done(op_id):
    """Mark an operation as complete."""
    if op_id in PROGRESS:
        PROGRESS[op_id]['done'] = True
        PROGRESS[op_id]['done_at'] = time.time()
        PROGRESS[op_id]['step'] = PROGRESS[op_id]['total']
        PROGRESS[op_id]['detail'] = 'Done'

def progress_remove(op_id):
    """Remove a progress entry."""
    PROGRESS.pop(op_id, None)

def progress_cleanup():
    """Remove entries that finished more than 8s ago."""
    now = time.time()
    to_remove = [k for k, v in PROGRESS.items() if v.get('done') and v.get('done_at') and now - v['done_at'] > 8]
    for k in to_remove:
        del PROGRESS[k]
    # Also clean up stale cancel flags
    for k in list(CANCEL_FLAGS.keys()):
        if k not in PROGRESS:
            CANCEL_FLAGS.pop(k, None)

CANCEL_FLAGS = {}  # op_id -> True when cancellation requested
ARCHIVE_CACHE_KEYS = set()  # cycle keys (YYYYMMDD_HHz) from archive requests — persist on NVMe

def cancel_request(op_id):
    """Request cancellation of an operation."""
    CANCEL_FLAGS[op_id] = True
    if op_id in PROGRESS and not PROGRESS[op_id].get('done'):
        PROGRESS[op_id]['detail'] = 'Cancelling...'

def is_cancelled(op_id):
    """Check if an operation has been cancelled."""
    return CANCEL_FLAGS.get(op_id, False)

# =============================================================================
# PROCESS POOL WORKER — runs in separate process, own GIL, true parallelism
# =============================================================================

# =============================================================================
# DATA MANAGER - On-Demand Loading for Memory Efficiency
# =============================================================================

# Model-specific configuration for CrossSectionManager
MODEL_PRS_PATTERNS = {
    'hrrr': '*wrfprs*.grib2',
    'rrfs': '*prslev*.grib2',
    'gfs':  '*pgrb2.0p25*',  # GFS files have no .grib2 extension
}
MODEL_SFC_PATTERNS = {
    'hrrr': '*wrfsfc*.grib2',
    'rrfs': '*prslev*.grib2',   # RRFS: surface in same prslev file
    'gfs':  '*pgrb2.0p25*',    # GFS: surface in same pgrb2 file (no .grib2 ext)
}
MODEL_NEEDS_SEPARATE_SFC = {'hrrr'}  # Only HRRR has separate wrfsfc
MODEL_FORECAST_HOURS = {
    'hrrr': list(range(19)),                # F00-F18 (base; synoptic cycles extend to F48)
    'gfs':  list(range(0, 385, 6)),         # F00-F384 every 6 hours
    'rrfs': list(range(19)),                # F00-F18
}
SYNOPTIC_HOURS = {0, 6, 12, 18}

def get_max_fhr_for_cycle(model_name: str, cycle_hour: int) -> int:
    """Return max forecast hour for a given model+cycle. HRRR synoptic cycles go to 48."""
    if model_name == 'hrrr' and cycle_hour in SYNOPTIC_HOURS:
        return 48
    base = MODEL_FORECAST_HOURS.get(model_name, list(range(19)))
    return base[-1] if base else 18

def get_model_fhr_list(model_name: str, cycle_hour: int = None) -> list:
    """Return the actual FHR list for a model+cycle (handles sparse GFS FHRs)."""
    if model_name == 'hrrr' and cycle_hour is not None and cycle_hour in SYNOPTIC_HOURS:
        return list(range(49))  # F00-F48 every hour
    return MODEL_FORECAST_HOURS.get(model_name, list(range(19)))
MODEL_MIN_LEVELS = {
    'hrrr': 40,
    'gfs':  20,  # GFS has ~34 pressure levels
    'rrfs': 40,
}
MODEL_EXCLUDED_STYLES = {
    'gfs': {'smoke'},       # GFS has no PM2.5/smoke
    'rrfs': {'smoke'},      # RRFS has no smoke either
    'hrrr': set(),          # HRRR supports all styles
}

def _env_int(name: str, default: int) -> int:
    """Parse integer env var with fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class CrossSectionManager:
    """Manages cross-section data with smart pre-loading.

    Pre-loads latest N cycles at startup for instant access.
    Older cycles are available on-demand with loading indicator.
    Parameterized by model_name for multi-model support.
    """

    PRELOAD_WORKERS = 20  # Thread workers for mmap loads (fast, ~14ms each)
    GRIB_WORKERS = 4      # Workers for GRIB conversion
    CACHE_BASE = '/home/drew/hrrr-maps/cache/xsect'  # NVMe — faster I/O, no VHD folio contention

    PRELOAD_CYCLES = 0  # Don't pre-load; load on demand

    def __init__(self, model_name: str = 'hrrr', mem_limit_mb: int = 48000, mem_evict_mb: int = 46000):
        self.model_name = model_name
        self.xsect = None
        self.base_dir = Path(f"outputs/{model_name}")
        self.available_cycles = []  # List of available cycles (metadata only)
        self.loaded_cycles = set()  # Cycle keys that are fully loaded
        self.loaded_items = []  # List of (cycle_key, fhr) currently in memory (ordered by load time = LRU)
        self.current_cycle = None  # Currently selected cycle
        self._lock = threading.Lock()  # Protects all state mutations
        self._loading = threading.Lock()  # Prevents overlapping bulk loads (preload vs load_cycle)
        self._engine_key_map = {}  # (cycle_key, fhr) -> unique engine int key
        self._next_engine_key = 0  # Counter for unique keys
        # Model-specific config
        self._prs_pattern = MODEL_PRS_PATTERNS.get(model_name, '*.grib2')
        self._sfc_pattern = MODEL_SFC_PATTERNS.get(model_name, '*.grib2')
        self._needs_separate_sfc = model_name in MODEL_NEEDS_SEPARATE_SFC
        self.FORECAST_HOURS = MODEL_FORECAST_HOURS.get(model_name, list(range(19)))
        self.PRELOAD_FHRS = self.FORECAST_HOURS  # All FHRs — mmap cache makes this cheap
        self.MEM_LIMIT_MB = mem_limit_mb
        self.MEM_EVICT_MB = mem_evict_mb
        self._min_levels = MODEL_MIN_LEVELS.get(model_name, 40)
        self._display_prefix = model_name.upper()
        # Load model config for metadata
        try:
            from model_config import get_model_registry
            self.model_config = get_model_registry().get_model(model_name)
            if self.model_config:
                self._display_prefix = self.model_config.full_name.split('(')[0].strip()
        except Exception:
            self.model_config = None

    def _sfc_file_from_prs(self, prs_file: str) -> str:
        """Derive the surface GRIB path from a pressure GRIB path."""
        if self.model_name == 'hrrr':
            sfc = Path(prs_file).parent / Path(prs_file).name.replace('wrfprs', 'wrfsfc')
            return str(sfc) if sfc.exists() else prs_file
        # GFS/RRFS: surface data is in the same pressure file
        return prs_file

    def _nat_file_from_prs(self, prs_file: str):
        """Derive the native-level GRIB path (for smoke). Returns None if not applicable."""
        if self.model_name == 'hrrr':
            nat = Path(prs_file).parent / Path(prs_file).name.replace('wrfprs', 'wrfnat')
            return str(nat) if nat.exists() else None
        elif self.model_name == 'rrfs':
            nat = Path(prs_file).parent / Path(prs_file).name.replace('prslev', 'natlev')
            return str(nat) if nat.exists() else None
        return None  # GFS has no native levels

    # ── Smart cycle selection ──
    # HRRR: newest synoptic (48h) + 5 most recent hourly (18h each)
    #        Keep previous synoptic during handoff until new one is ready.
    # GFS/RRFS: newest cycle only; keep previous during handoff.
    HRRR_HOURLY_CYCLES = 3   # Number of recent hourly cycles to keep
    GFS_CYCLES = 1            # GFS cycles to keep (+ 1 during handoff)
    RRFS_CYCLES = 1           # RRFS cycles to keep (+ 1 during handoff)

    def _get_target_cycles(self) -> list:
        """Return the list of cycles we WANT loaded, in priority order (newest first).

        HRRR: newest synoptic (48h) + N most recent hourly cycles.
              Only one synoptic kept (no previous synoptic handoff).
        GFS/RRFS: newest cycle only, no handoff.
        """
        if not self.available_cycles:
            return []

        if self.model_name == 'hrrr':
            return self._get_hrrr_target_cycles()
        else:
            return self._get_simple_target_cycles()

    def _get_hrrr_target_cycles(self) -> list:
        """HRRR: latest init first, then newest synoptic, then recent hourlies.

        Priority order:
          1. Latest init cycle (whatever it is) — always #1
          2. Newest synoptic (48h) — if not already the latest init
          3. N most recent hourly cycles
        Only one synoptic cycle is kept — no previous synoptic handoff.
        """
        targets = []
        seen = set()

        newest = self.available_cycles[0]  # Overall newest init
        synoptics = [c for c in self.available_cycles if c.get('is_synoptic')]

        # 1. Latest init — always first, period
        targets.append(newest)
        seen.add(newest['cycle_key'])

        # 2. Newest synoptic (if it's not already the latest init)
        if synoptics and synoptics[0]['cycle_key'] not in seen:
            targets.append(synoptics[0])
            seen.add(synoptics[0]['cycle_key'])

        # 3. Recent hourly cycles (up to N)
        count = 0
        for c in self.available_cycles:
            if c['cycle_key'] not in seen:
                targets.append(c)
                seen.add(c['cycle_key'])
                count += 1
                if count >= self.HRRR_HOURLY_CYCLES:
                    break

        return targets

    def _get_simple_target_cycles(self) -> list:
        """GFS/RRFS: newest cycle only, no handoff."""
        return [self.available_cycles[0]]

    def get_protected_cycles(self) -> set:
        """Return cycle keys that should never be evicted — matches target cycle strategy."""
        return {c['cycle_key'] for c in self._get_target_cycles()}

    @staticmethod
    def _priority_sort_fhrs(fhrs: list) -> list:
        """Sort FHRs by temporal priority: every 6h, then 3h fill, then hourly.

        Users see the full time range at coarse resolution fast, then it fills in.
        E.g. for F00-F48: [0,6,12,18,24,30,36,42,48, 3,9,15,21,27,33,39,45, 1,2,4,5,7,8,...]
        """
        tier1 = [f for f in fhrs if f % 6 == 0]  # Every 6h
        tier2 = [f for f in fhrs if f % 3 == 0 and f % 6 != 0]  # Every 3h fill
        tier3 = [f for f in fhrs if f % 3 != 0]  # Hourly fill
        return tier1 + tier2 + tier3

    def is_archive_cycle(self, cycle_key: str) -> bool:
        """Return True if cycle_key is NOT one of the 2 latest (i.e. it's archive/old)."""
        return cycle_key not in self.get_protected_cycles()

    def _get_engine_key(self, cycle_key: str, fhr: int) -> int:
        """Get or create a unique engine key for a (cycle_key, fhr) pair."""
        pair = (cycle_key, fhr)
        if pair not in self._engine_key_map:
            self._engine_key_map[pair] = self._next_engine_key
            self._next_engine_key += 1
        return self._engine_key_map[pair]

    def _evict_if_needed(self):
        """Evict oldest loaded items if memory exceeds threshold. Protected cycles are skipped."""
        if not self.xsect:
            return
        protected = self.get_protected_cycles()
        mem_mb = self.xsect.get_memory_usage()
        while mem_mb > self.MEM_EVICT_MB and self.loaded_items:
            # Find oldest non-protected item to evict
            evict_idx = None
            for i, (ck, fhr) in enumerate(self.loaded_items):
                if ck not in protected:
                    evict_idx = i
                    break
            if evict_idx is None:
                logger.warning(f"Memory {mem_mb:.0f}MB > limit but only protected cycles loaded, cannot evict")
                break
            old_key, old_fhr = self.loaded_items.pop(evict_idx)
            logger.info(f"Memory {mem_mb:.0f}MB > {self.MEM_EVICT_MB}MB, evicting {old_key} F{old_fhr:02d}")
            self._unload_item(old_key, old_fhr)
            if not any(k == old_key for k, _ in self.loaded_items):
                self.loaded_cycles.discard(old_key)
            mem_mb = self.xsect.get_memory_usage()

    def init_engine(self):
        """Initialize the cross-section engine if needed."""
        if self.xsect is None:
            from core.cross_section_interactive import InteractiveCrossSection
            cache_dir = f'{self.CACHE_BASE}/{self.model_name}'
            grib_backend = os.environ.get('XSECT_GRIB_BACKEND', 'auto').strip().lower()
            try:
                self.xsect = InteractiveCrossSection(
                    cache_dir=cache_dir,
                    min_levels=self._min_levels,
                    sfc_resolver=lambda prs: self._sfc_file_from_prs(prs),
                    nat_resolver=lambda prs: self._nat_file_from_prs(prs),
                    grib_backend=grib_backend,
                )
            except ValueError:
                logger.warning(f"Invalid XSECT_GRIB_BACKEND='{grib_backend}', falling back to 'auto'")
                self.xsect = InteractiveCrossSection(
                    cache_dir=cache_dir,
                    min_levels=self._min_levels,
                    sfc_resolver=lambda prs: self._sfc_file_from_prs(prs),
                    nat_resolver=lambda prs: self._nat_file_from_prs(prs),
                    grib_backend='auto',
                )
            self.xsect.model = self.model_name.upper()
            logger.info(f"Cross-section GRIB backend: {self.xsect.grib_backend}")
            # Climatology/anomaly mode disabled for now

    def scan_available_cycles(self):
        """Scan for all available cycles on disk WITHOUT loading data."""
        from datetime import datetime

        self.available_cycles = []

        if not self.base_dir.exists():
            return self.available_cycles

        # Scan for date directories
        for date_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not date_dir.is_dir() or not date_dir.name.isdigit():
                continue
            if len(date_dir.name) != 8:
                continue

            # Scan for hour directories within date
            for hour_dir in sorted(date_dir.iterdir(), reverse=True):
                if not hour_dir.is_dir() or not hour_dir.name.endswith('z'):
                    continue

                hour = hour_dir.name.replace('z', '')
                if not hour.isdigit():
                    continue

                # Check what forecast hours are available on disk
                available_fhrs = []
                cycle_hour_int = int(hour)
                max_fhr = get_max_fhr_for_cycle(self.model_name, cycle_hour_int)
                expected_fhrs = get_model_fhr_list(self.model_name, cycle_hour_int)
                for fhr in expected_fhrs:
                    fhr_dir = hour_dir / f"F{fhr:02d}"
                    if fhr_dir.exists():
                        has_prs = [f for f in fhr_dir.glob(self._prs_pattern)
                                   if not f.name.endswith('.partial')]
                        if self._needs_separate_sfc:
                            has_sfc = [f for f in fhr_dir.glob(self._sfc_pattern)
                                       if not f.name.endswith('.partial')]
                            if has_prs and has_sfc:
                                available_fhrs.append(fhr)
                        else:
                            # GFS/RRFS: surface data is in the pressure file
                            if has_prs:
                                available_fhrs.append(fhr)

                if available_fhrs:
                    cycle_key = f"{date_dir.name}_{hour}z"
                    init_dt = datetime.strptime(f"{date_dir.name}{hour}", "%Y%m%d%H")

                    self.available_cycles.append({
                        'cycle_key': cycle_key,
                        'date': date_dir.name,
                        'hour': hour,
                        'path': str(hour_dir),
                        'available_fhrs': available_fhrs,
                        'init_dt': init_dt,
                        'display': f"{self._display_prefix} - {init_dt.strftime('%b %d %HZ')}",
                        'max_fhr': max_fhr,
                        'is_synoptic': cycle_hour_int in SYNOPTIC_HOURS,
                        'expected_fhrs': expected_fhrs,
                    })

        return self.available_cycles

    def get_cycles_for_ui(self):
        """Return cycles formatted for UI dropdown.

        Only includes cycles that are either:
          - In the preload target window (recent inits the dashboard manages)
          - Have loaded data (e.g. from archive requests)
          - Have an active download/load in progress
        Older cycles on disk are hidden until explicitly requested via archive.
        """
        target_keys = {c['cycle_key'] for c in self._get_target_cycles()}
        loaded_keys = {ck for ck, _ in self.loaded_items}

        # Include cycles with active download/load operations
        # op_id format: "download:hrrr:20250618/15z" or "load:hrrr:20250618/15z"
        active_keys = set()
        for op_id, info in PROGRESS.items():
            if info.get('done'):
                continue
            parts = op_id.split(':')
            if len(parts) >= 3 and parts[1] == self.model_name:
                # Convert "20250618/15z" -> "20250618_15z"
                ck = parts[2].replace('/', '_')
                active_keys.add(ck)

        visible_keys = target_keys | loaded_keys | active_keys

        return [
            {
                'key': c['cycle_key'],
                'display': c['display'],
                'date': c['date'],
                'hour': c['hour'],
                'fhrs': c['available_fhrs'],
                'fhr_count': len(c['available_fhrs']),
                'loaded': c['cycle_key'] in loaded_keys,
                'max_fhr': c.get('max_fhr', 18),
                'is_synoptic': c.get('is_synoptic', False),
                'expected_fhrs': c.get('expected_fhrs', None),
            }
            for c in self.available_cycles
            if c['cycle_key'] in visible_keys
        ]

    def preload_latest_cycles(self, n_cycles: int = None):
        """Pre-load the latest N cycles with every 3rd forecast hour, newest first, parallel."""
        if n_cycles is None:
            n_cycles = self.PRELOAD_CYCLES

        with self._loading:
            self._preload_latest_cycles_inner(n_cycles)

    def _preload_latest_cycles_inner(self, n_cycles):
        self.init_engine()

        # Smart cycle selection: load only what matters
        cycles_to_load = self._get_target_cycles()

        # Build per-cycle FHR queues, priority-sorted within each cycle
        cycle_queues = []
        for cycle in cycles_to_load:
            cycle_key = cycle['cycle_key']
            is_synoptic = cycle.get('is_synoptic', False) and self.model_name == 'hrrr'
            allowed = cycle['available_fhrs'] if is_synoptic else [f for f in cycle['available_fhrs'] if f in self.PRELOAD_FHRS]
            with self._lock:
                fhrs = [fhr for fhr in allowed if (cycle_key, fhr) not in self.loaded_items]
            if fhrs:
                fhrs = self._priority_sort_fhrs(fhrs)
                cycle_queues.append((cycle, fhrs))

        if not cycle_queues:
            return

        # Build flat list, newest-first, priority-sorted within each cycle
        interleaved = []
        for cycle, fhrs in cycle_queues:
            for fhr in fhrs:
                interleaved.append((cycle, fhr))

        # Partition into cached (instant mmap load) vs uncached (slow GRIB conversion).
        # Cached items load first so users see rapid progress instead of waiting
        # for GRIB conversions before anything appears.
        cache_dir = Path(f'{self.CACHE_BASE}/{self.model_name}')
        cached_items = []
        uncached_items = []
        for cycle, fhr in interleaved:
            prs_files = list((Path(cycle['path']) / f"F{fhr:02d}").glob(self._prs_pattern))
            if prs_files:
                stem = self.xsect._get_cache_stem(str(prs_files[0]))
                if stem and (cache_dir / stem).is_dir():
                    cached_items.append((cycle, fhr))
                else:
                    uncached_items.append((cycle, fhr))
            else:
                uncached_items.append((cycle, fhr))

        # Load cached first, then uncached
        ordered = cached_items + uncached_items
        n_cached = len(cached_items)

        total = len(ordered)
        done = [0]
        op_id = "preload:startup"
        n_cyc = len(cycle_queues)

        cycle_list = ', '.join(c['cycle_key'] for c, _ in cycle_queues)
        progress_update(op_id, 0, total, "Starting...",
                        label=f"Pre-loading {n_cyc} cycles ({total} FHRs, {n_cached} cached)")
        logger.info(f"Pre-loading {total} FHRs across {n_cyc} cycles: [{cycle_list}] ({n_cached} cached, {total - n_cached} need GRIB)")

        def _make_loader(c):
            def _load_one(fhr):
                ck = c['cycle_key']
                with self._lock:
                    if (ck, fhr) in self.loaded_items:
                        return ck, fhr, True
                    self.xsect.init_date = c['date']
                    self.xsect.init_hour = c['hour']
                    self._evict_if_needed()
                    engine_key = self._get_engine_key(ck, fhr)

                prs_files = list((Path(c['path']) / f"F{fhr:02d}").glob(self._prs_pattern))
                if prs_files and self.xsect.load_forecast_hour(str(prs_files[0]), engine_key):
                    with self._lock:
                        if (ck, fhr) not in self.loaded_items:
                            self.loaded_items.append((ck, fhr))
                    return ck, fhr, True
                return ck, fhr, False
            return _load_one

        # Phase 1: Load cached items (fast — mmap loads in <0.1s each)
        cancelled = False
        if cached_items:
            logger.info(f"  Phase 1: Loading {n_cached} cached FHRs...")
            with ThreadPoolExecutor(max_workers=self.PRELOAD_WORKERS) as pool:
                futures = {}
                for cycle, fhr in cached_items:
                    loader = _make_loader(cycle)
                    futures[pool.submit(loader, fhr)] = (cycle['cycle_key'], fhr)

                for future in as_completed(futures):
                    if is_cancelled(op_id):
                        for f in futures:
                            f.cancel()
                        cancelled = True
                        break
                    try:
                        ck, fhr, ok = future.result()
                        done[0] += 1
                        if ok:
                            logger.info(f"  Loaded {ck} F{fhr:02d} (cached)")
                            progress_update(op_id, done[0], total, f"Loaded {ck} F{fhr:02d} (cached)")
                        else:
                            progress_update(op_id, done[0], total, f"Failed {ck} F{fhr:02d}")
                    except Exception as e:
                        done[0] += 1
                        logger.warning(f"  Failed to load FHR: {e}")

            logger.info(f"  Phase 1 done: {done[0]}/{n_cached} cached FHRs loaded")

        # Phase 2: Convert uncached items from GRIB (ThreadPool — avoids WSL2 folio contention)
        if uncached_items and not cancelled:
            logger.info(f"  Phase 2: Converting {len(uncached_items)} FHRs from GRIB ({self.GRIB_WORKERS} workers)...")
            with ThreadPoolExecutor(max_workers=self.GRIB_WORKERS) as pool:
                futures = {}
                for cycle, fhr in uncached_items:
                    loader = _make_loader(cycle)
                    futures[pool.submit(loader, fhr)] = (cycle['cycle_key'], fhr)

                for future in as_completed(futures):
                    if is_cancelled(op_id):
                        for f in futures:
                            f.cancel()
                        cancelled = True
                        break
                    try:
                        ck, fhr, ok = future.result()
                        done[0] += 1
                        if ok:
                            logger.info(f"  Loaded {ck} F{fhr:02d} (GRIB)")
                            progress_update(op_id, done[0], total, f"Loaded {ck} F{fhr:02d} (GRIB)")
                        else:
                            progress_update(op_id, done[0], total, f"Failed {ck} F{fhr:02d}")
                    except Exception as e:
                        done[0] += 1
                        logger.warning(f"  Failed to load FHR: {e}")

        mem_mb = self.xsect.get_memory_usage()
        if cancelled:
            logger.info(f"  Pre-load CANCELLED at {done[0]}/{total} ({mem_mb:.0f} MB)")
            PROGRESS[op_id]['detail'] = 'Cancelled'
        else:
            logger.info(f"  Pre-load done ({mem_mb:.0f} MB total)")
        progress_done(op_id)
        CANCEL_FLAGS.pop(op_id, None)

    def auto_load_latest(self):
        """Load new FHRs from disk, newest cycle first (priority)."""
        if not self._loading.acquire(blocking=False):
            logger.info("Skipping auto-load — another load operation in progress")
            return
        try:
            self._auto_load_latest_inner()
        finally:
            self._loading.release()

    def _auto_load_latest_inner(self):
        if not self.xsect or not self.available_cycles:
            return

        # Use the same smart cycle selection as preload — only load target cycles
        priority_cycles = self._get_target_cycles()

        # Evict FHRs from cycles that are no longer targeted
        target_keys = {c['cycle_key'] for c in priority_cycles}
        with self._lock:
            to_evict = [(ck, fhr) for ck, fhr in self.loaded_items if ck not in target_keys]
        for ck, fhr in to_evict:
            logger.info(f"  Evicting {ck} F{fhr:02d} (no longer in target cycles)")
            with self._lock:
                if (ck, fhr) in self.loaded_items:
                    self.loaded_items.remove((ck, fhr))
            self._unload_item(ck, fhr)

        # Collect all FHRs to load across all target cycles for progress tracking
        all_work = []  # [(cycle, cycle_key, fhr, is_cached)]
        newest = priority_cycles[0] if priority_cycles else None
        for cycle in priority_cycles:
            cycle_key = cycle['cycle_key']
            is_synoptic = cycle.get('is_synoptic', False)

            if is_synoptic and self.model_name == 'hrrr':
                allowed_fhrs = cycle['available_fhrs']
            else:
                allowed_fhrs = [f for f in cycle['available_fhrs'] if f in self.PRELOAD_FHRS]

            with self._lock:
                fhrs_to_load = [fhr for fhr in allowed_fhrs
                                if (cycle_key, fhr) not in self.loaded_items]

            if not fhrs_to_load:
                continue

            fhrs_to_load = self._priority_sort_fhrs(fhrs_to_load)

            # Partition into cached vs uncached
            cache_dir = Path(f'{self.CACHE_BASE}/{self.model_name}')
            for fhr in fhrs_to_load:
                prs_files = list((Path(cycle['path']) / f"F{fhr:02d}").glob(self._prs_pattern))
                is_cached = False
                if prs_files:
                    stem = self.xsect._get_cache_stem(str(prs_files[0]))
                    if stem and (cache_dir / stem).is_dir():
                        is_cached = True
                all_work.append((cycle, cycle_key, fhr, is_cached))

        if not all_work:
            return

        # Sort: cached first, then uncached
        all_work.sort(key=lambda x: (not x[3], x[1], x[2]))

        n_cached = sum(1 for _, _, _, c in all_work if c)
        n_uncached = len(all_work) - n_cached

        # Progress tracking
        op_id = f"autoload:{self.model_name}"
        cycle_keys = sorted(set(ck for _, ck, _, _ in all_work))
        cycle_label = cycle_keys[0] if len(cycle_keys) == 1 else f"{len(cycle_keys)} cycles"
        progress_update(op_id, 0, len(all_work), "Starting...",
                        label=f"Loading {self.model_name.upper()} {cycle_label} ({n_cached} cached, {n_uncached} GRIB)")

        completed = [0]

        def _make_loader(c, ck):
            def _load_one(fhr):
                with self._lock:
                    if (ck, fhr) in self.loaded_items:
                        return fhr, True
                    self.xsect.init_date = c['date']
                    self.xsect.init_hour = c['hour']
                    self._evict_if_needed()
                    engine_key = self._get_engine_key(ck, fhr)

                prs_files = list((Path(c['path']) / f"F{fhr:02d}").glob(self._prs_pattern))
                if prs_files and self.xsect.load_forecast_hour(str(prs_files[0]), engine_key):
                    with self._lock:
                        if (ck, fhr) not in self.loaded_items:
                            self.loaded_items.append((ck, fhr))
                    return fhr, True
                return fhr, False
            return _load_one

        def _on_complete(cycle_key, fhr, ok, is_cached):
            completed[0] += 1
            tag = "cached" if is_cached else "GRIB"
            detail = f"{cycle_key} F{fhr:02d} {'OK' if ok else 'FAILED'} ({tag})"
            progress_update(op_id, completed[0], len(all_work), detail)
            if ok:
                logger.info(f"  Auto-loaded {cycle_key} F{fhr:02d} ({tag})")

        # Phase 1: cached (fast)
        cached_work = [(c, ck, fhr) for c, ck, fhr, is_c in all_work if is_c]
        if cached_work:
            with ThreadPoolExecutor(max_workers=self.PRELOAD_WORKERS) as pool:
                futures = {}
                for c, ck, fhr in cached_work:
                    loader = _make_loader(c, ck)
                    fut = pool.submit(loader, fhr)
                    futures[fut] = (ck, fhr)
                for future in as_completed(futures):
                    ck, fhr = futures[future]
                    try:
                        _, ok = future.result()
                        _on_complete(ck, fhr, ok, True)
                    except Exception as e:
                        _on_complete(ck, fhr, False, True)
                        logger.warning(f"  Auto-load failed: {e}")

        # Phase 2: uncached GRIB conversion (slow)
        uncached_work = [(c, ck, fhr) for c, ck, fhr, is_c in all_work if not is_c]
        if uncached_work:
            with ThreadPoolExecutor(max_workers=self.GRIB_WORKERS) as pool:
                futures = {}
                for c, ck, fhr in uncached_work:
                    loader = _make_loader(c, ck)
                    fut = pool.submit(loader, fhr)
                    futures[fut] = (ck, fhr)
                for future in as_completed(futures):
                    ck, fhr = futures[future]
                    try:
                        _, ok = future.result()
                        _on_complete(ck, fhr, ok, False)
                    except Exception as e:
                        _on_complete(ck, fhr, False, False)
                        logger.warning(f"  Auto-load failed: {e}")

        progress_done(op_id)

    def get_loaded_status(self):
        """Return current memory status."""
        mem_mb = self.xsect.get_memory_usage() if self.xsect else 0
        return {
            'loaded': self.loaded_items.copy(),
            'loaded_cycles': list(self.loaded_cycles),
            'memory_mb': round(mem_mb, 0),
            'loading': self._lock.locked(),
        }

    def resolve_cycle(self, cycle_key: str, fhr: int) -> 'Optional[str]':
        """Resolve 'latest' to an actual cycle key. Returns None if nothing available."""
        if cycle_key and cycle_key != 'latest':
            return cycle_key
        with self._lock:
            # Prefer newest loaded cycle that has this FHR
            for ck, f in reversed(self.loaded_items):
                if f == fhr:
                    return ck
            # Fall back to newest available cycle on disk
            for c in self.available_cycles:
                if fhr in c['available_fhrs']:
                    return c['cycle_key']
        return None

    def ensure_loaded(self, cycle_key: str, fhr: int) -> bool:
        """Ensure a forecast hour is loaded (auto-loads from mmap/GRIB). Returns True if ready."""
        with self._lock:
            if (cycle_key, fhr) in self.loaded_items:
                return True
        result = self.load_forecast_hour(cycle_key, fhr)
        return result.get('success', False)

    def load_cycle(self, cycle_key: str) -> dict:
        """Load an entire cycle (all available FHRs) into memory, parallel."""
        if not self._loading.acquire(timeout=120):
            return {'success': False, 'error': 'Timed out waiting for other load to finish'}
        try:
            return self._load_cycle_inner(cycle_key)
        finally:
            self._loading.release()

    def _load_cycle_inner(self, cycle_key: str) -> dict:
        with self._lock:
            cycle = next((c for c in self.available_cycles if c['cycle_key'] == cycle_key), None)
            if not cycle:
                return {'success': False, 'error': f'Cycle {cycle_key} not found'}

            # Check if ALL available FHRs are already loaded
            loaded_fhrs = {fhr for ck, fhr in self.loaded_items if ck == cycle_key}
            if loaded_fhrs >= set(cycle['available_fhrs']):
                mem_mb = self.xsect.get_memory_usage() if self.xsect else 0
                return {'success': True, 'already_loaded': True, 'loaded_fhrs': len(loaded_fhrs), 'memory_mb': round(mem_mb, 0)}

            self.init_engine()

        run_path = Path(cycle['path'])
        op_id = f"cycle:{cycle_key}"
        total_fhrs = len(cycle['available_fhrs'])
        progress_update(op_id, 0, total_fhrs, "Starting...", label=f"Loading cycle {cycle_key}")
        loaded_count = [0]  # mutable for closure

        # Count already-loaded
        with self._lock:
            fhrs_to_load = []
            for fhr in cycle['available_fhrs']:
                if (cycle_key, fhr) in self.loaded_items:
                    loaded_count[0] += 1
                else:
                    fhrs_to_load.append(fhr)

        def _load_one(fhr):
            with self._lock:
                if (cycle_key, fhr) in self.loaded_items:
                    return fhr, True
                self.xsect.init_date = cycle['date']
                self.xsect.init_hour = cycle['hour']
                self._evict_if_needed()
                engine_key = self._get_engine_key(cycle_key, fhr)

            prs_files = list((run_path / f"F{fhr:02d}").glob(self._prs_pattern))
            if prs_files and self.xsect.load_forecast_hour(str(prs_files[0]), engine_key):
                with self._lock:
                    if (cycle_key, fhr) not in self.loaded_items:
                        self.loaded_items.append((cycle_key, fhr))
                return fhr, True
            return fhr, False

        logger.info(f"Loading {cycle['display']} ({len(fhrs_to_load)} FHRs, {self.PRELOAD_WORKERS} workers)...")

        cancelled = False
        with ThreadPoolExecutor(max_workers=self.PRELOAD_WORKERS) as pool:
            futures = {pool.submit(_load_one, fhr): fhr for fhr in fhrs_to_load}
            for future in as_completed(futures):
                if is_cancelled(op_id):
                    for f in futures:
                        f.cancel()
                    cancelled = True
                    break
                try:
                    fhr, ok = future.result()
                    if ok:
                        loaded_count[0] += 1
                        progress_update(op_id, loaded_count[0], total_fhrs, f"Loaded F{fhr:02d}")
                        logger.info(f"  Loaded {cycle_key} F{fhr:02d}")
                except Exception as e:
                    logger.warning(f"  Failed to load FHR: {e}")

        with self._lock:
            self.loaded_cycles.add(cycle_key)
        mem_mb = self.xsect.get_memory_usage()
        if cancelled:
            logger.info(f"Load CANCELLED for {cycle['display']} at {loaded_count[0]}/{total_fhrs} ({mem_mb:.0f} MB)")
            PROGRESS[op_id]['detail'] = 'Cancelled'
        else:
            logger.info(f"Loaded {cycle['display']} ({loaded_count[0]} FHRs, {mem_mb:.0f} MB total)")
        progress_done(op_id)
        CANCEL_FLAGS.pop(op_id, None)

        return {
            'success': True,
            'cycle': cycle_key,
            'loaded_fhrs': loaded_count[0],
            'memory_mb': round(mem_mb, 0),
        }

    def load_forecast_hour(self, cycle_key: str, fhr: int) -> dict:
        """Load a specific forecast hour into memory."""
        from datetime import datetime, timedelta

        # Fast checks and state setup under lock
        with self._lock:
            if (cycle_key, fhr) in self.loaded_items:
                return {'success': True, 'already_loaded': True}

            cycle = next((c for c in self.available_cycles if c['cycle_key'] == cycle_key), None)
            if not cycle:
                return {'success': False, 'error': f'Cycle {cycle_key} not found'}

            if fhr not in cycle['available_fhrs']:
                return {'success': False, 'error': f'F{fhr:02d} not available for {cycle_key}'}

            self.init_engine()
            self.xsect.init_date = cycle['date']
            self.xsect.init_hour = cycle['hour']

            run_path = Path(cycle['path'])
            fhr_dir = run_path / f"F{fhr:02d}"
            prs_files = list(fhr_dir.glob(self._prs_pattern))
            if not prs_files:
                return {'success': False, 'error': f'No GRIB file found for F{fhr:02d}'}

            self._evict_if_needed()
            engine_key = self._get_engine_key(cycle_key, fhr)

        # Slow GRIB I/O runs WITHOUT lock
        op_id = f"load:{cycle_key}:F{fhr:02d}"
        progress_update(op_id, 0, 12, "Starting...", label=f"Loading {cycle_key} F{fhr:02d}")
        logger.info(f"Loading {cycle_key} F{fhr:02d} (engine key {engine_key})...")
        load_start = time.time()

        def _progress_cb(step, total, detail):
            progress_update(op_id, step, total, detail)

        success = self.xsect.load_forecast_hour(str(prs_files[0]), engine_key, progress_callback=_progress_cb)

        # State update under lock
        if success:
            load_time = time.time() - load_start
            with self._lock:
                if (cycle_key, fhr) not in self.loaded_items:
                    self.loaded_items.append((cycle_key, fhr))
                self.current_cycle = cycle_key
            mem_mb = self.xsect.get_memory_usage()
            logger.info(f"Loaded {cycle_key} F{fhr:02d} in {load_time:.1f}s (Total: {mem_mb:.0f} MB)")
            progress_done(op_id)
            return {
                'success': True,
                'loaded': (cycle_key, fhr),
                'memory_mb': round(mem_mb, 0),
                'load_time': round(load_time, 1),
            }
        else:
            progress_remove(op_id)
            return {'success': False, 'error': 'Failed to load data'}

    def _unload_item(self, cycle_key: str, fhr: int):
        """Unload a forecast hour from memory."""
        engine_key = self._engine_key_map.get((cycle_key, fhr))
        if self.xsect and engine_key is not None and engine_key in self.xsect.forecast_hours:
            self.xsect.unload_hour(engine_key)
            logger.info(f"Unloaded {cycle_key} F{fhr:02d}")

    def unload_forecast_hour(self, cycle_key: str, fhr: int, is_admin: bool = False) -> dict:
        """Explicitly unload a forecast hour."""
        with self._lock:
            if (cycle_key, fhr) not in self.loaded_items:
                return {'success': True, 'not_loaded': True}

            self._unload_item(cycle_key, fhr)
            self.loaded_items.remove((cycle_key, fhr))

        mem_mb = self.xsect.get_memory_usage() if self.xsect else 0
        return {
            'success': True,
            'unloaded': (cycle_key, fhr),
            'memory_mb': round(mem_mb, 0),
        }

    def generate_cross_section(self, start, end, cycle_key, fhr, style, y_axis='pressure', vscale=1.0, y_top=100, units='km', terrain_data=None, temp_cmap='standard', anomaly=False):
        """Generate a cross-section for a loaded forecast hour."""
        if not self.xsect:
            return None

        if (cycle_key, fhr) not in self.loaded_items:
            return None

        cycle = next((c for c in self.available_cycles if c['cycle_key'] == cycle_key), None)
        engine_key = self._engine_key_map.get((cycle_key, fhr))
        if engine_key is None:
            return None

        # Build metadata with real FHR (not engine key) — thread-safe, no shared state
        meta = {
            'model': self.model_name.upper(),
            'init_date': cycle['date'] if cycle else None,
            'init_hour': cycle['hour'] if cycle else None,
            'forecast_hour': fhr,
        }

        try:
            png_bytes = self.xsect.get_cross_section(
                start_point=start,
                end_point=end,
                forecast_hour=engine_key,
                style=style,
                return_image=True,
                dpi=100,
                y_axis=y_axis,
                vscale=vscale,
                y_top=y_top,
                units=units,
                terrain_data=terrain_data,
                temp_cmap=temp_cmap,
                metadata=meta,
                anomaly=anomaly,
            )
            if png_bytes is None:
                return None
            return io.BytesIO(png_bytes)
        except Exception as e:
            import traceback
            logger.error(f"Cross-section error: {e}\n{traceback.format_exc()}")
            return None

    def get_terrain_data(self, start, end, cycle_key, fhr, style):
        """Extract terrain data from a forecast hour for consistent GIF frames."""
        if not self.xsect:
            return None
        engine_key = self._engine_key_map.get((cycle_key, fhr))
        if engine_key is None:
            return None
        fhr_data = self.xsect.forecast_hours.get(engine_key)
        if fhr_data is None:
            return None
        n_points = 100
        import numpy as np
        path_lats = np.linspace(start[0], end[0], n_points)
        path_lons = np.linspace(start[1], end[1], n_points)
        data = self.xsect._interpolate_to_path(fhr_data, path_lats, path_lons, style)
        return {
            'surface_pressure': data.get('surface_pressure'),
            'surface_pressure_hires': data.get('surface_pressure_hires'),
            'distances_hires': data.get('distances_hires'),
            'pressure_levels': fhr_data.pressure_levels,
        }

    # Legacy compatibility methods
    def get_available_times(self):
        """Legacy: Return loaded times for old API."""
        from datetime import timedelta
        times = []
        for cycle_key, fhr in self.loaded_items:
            cycle = next((c for c in self.available_cycles if c['cycle_key'] == cycle_key), None)
            if cycle:
                valid_dt = cycle['init_dt'] + timedelta(hours=fhr)
                times.append({
                    'valid': valid_dt.strftime("%Y-%m-%d %HZ"),
                    'init': cycle['init_dt'].strftime("%Y-%m-%d %HZ"),
                    'fhr': fhr,
                    'cycle_key': cycle_key,
                })
        return times


# =============================================================================
# MODEL MANAGER REGISTRY
# =============================================================================

ENABLED_MODELS = ['hrrr']  # Default; overridden by --models CLI arg
MODEL_MEM_BUDGETS = {
    'hrrr': (58000, 56000),
    'gfs':  (8000, 7500),
    'rrfs': (8000, 7500),
}


class ModelManagerRegistry:
    """Registry of CrossSectionManagers, one per model."""

    def __init__(self):
        self.managers: Dict[str, CrossSectionManager] = {}
        self.default_model = 'hrrr'

    def register(self, model_name: str):
        limit, evict = MODEL_MEM_BUDGETS.get(model_name, (25000, 24000))
        mgr = CrossSectionManager(
            model_name=model_name,
            mem_limit_mb=limit,
            mem_evict_mb=evict,
        )
        self.managers[model_name] = mgr
        return mgr

    def get(self, model_name: str = None) -> CrossSectionManager:
        name = (model_name or self.default_model).lower()
        if name not in self.managers:
            raise ValueError(f"Unknown model: {name}. Available: {list(self.managers.keys())}")
        return self.managers[name]

    def all_managers(self):
        return self.managers.items()

    def list_models(self):
        return [
            {
                'id': name,
                'name': mgr.model_config.full_name if mgr.model_config else name.upper(),
                'resolution': mgr.model_config.resolution if mgr.model_config else 'unknown',
                'domain': mgr.model_config.domain if mgr.model_config else 'unknown',
                'cycle_count': len(mgr.available_cycles),
                'loaded_count': len(mgr.loaded_items),
                'excluded_styles': sorted(MODEL_EXCLUDED_STYLES.get(name, set())),
            }
            for name, mgr in self.managers.items()
        ]


model_registry = ModelManagerRegistry()
model_registry.register('hrrr')  # Always register HRRR at import time
data_manager = model_registry.get('hrrr')  # Backward compat alias


def get_manager_from_request() -> CrossSectionManager:
    """Extract ?model= from request and return the correct manager."""
    model = request.args.get('model', 'hrrr').lower()
    try:
        return model_registry.get(model)
    except ValueError:
        return None


# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HRRR Cross-Section Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
            --bg: #0f172a;
            --panel: #1e293b;
            --card: #334155;
            --text: #f1f5f9;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --border: #475569;
            --success: #22c55e;
            --warning: #f59e0b;
        }
        body {
            font-family: system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
        }
        #map-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        #map { flex: 1; }
        #controls {
            background: var(--panel);
            padding: 8px 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            border-bottom: 1px solid var(--border);
        }
        .control-row {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }
        .control-group { display: flex; align-items: center; gap: 4px; }
        .control-row-secondary {
            display: none;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }
        .control-row-secondary.visible {
            display: flex;
        }
        #more-toggle {
            background: none; border: 1px solid var(--border); color: var(--muted);
            padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 11px;
        }
        #more-toggle:hover { color: var(--text); border-color: var(--accent); }
        label { color: var(--muted); font-size: 12px; font-weight: 500; }
        select {
            background: var(--card);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 5px 8px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            min-width: 120px;
        }
        select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }

        /* Forecast Hour Chips */
        .chip-group {
            display: flex;
            gap: 3px;
            align-items: center;
            flex-wrap: wrap;
        }
        .chip {
            background: var(--card);
            color: var(--muted);
            border: 1px solid var(--border);
            padding: 4px 8px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.15s ease;
            user-select: none;
        }
        .chip:hover { border-color: var(--accent); color: var(--text); }
        /* Downloaded but not loaded - default look, click will trigger load */
        /* Loaded in RAM, ready for instant view */
        .chip.loaded {
            background: rgba(76, 175, 80, 0.15);
            color: #4caf50;
            border-color: #4caf50;
        }
        .chip.loaded:hover { background: rgba(76, 175, 80, 0.3); }
        /* Currently viewing this FHR */
        .chip.active {
            background: var(--accent);
            color: #000;
            border-color: var(--accent);
            font-weight: 700;
        }
        .chip.loading {
            background: var(--warning);
            color: #000;
            border-color: var(--warning);
            animation: pulse 1s infinite;
        }
        .chip:disabled, .chip.unavailable {
            opacity: 0.4;
            cursor: not-allowed;
        }
        /* Shift+click unload hint */
        .chip.loaded:active, .chip.active:active {
            opacity: 0.7;
        }
        /* Extended FHR chips (F19-F48) */
        .chip.extended {
            border-style: dashed;
            font-size: 10px;
            padding: 3px 5px;
        }
        .chip-divider {
            color: var(--muted);
            margin: 0 2px;
            font-size: 14px;
            user-select: none;
            display: flex;
            align-items: center;
        }
        /* Time slider row */
        #slider-row {
            padding: 4px 16px;
            background: rgba(0,0,0,0.15);
            border-top: 1px solid var(--border);
            gap: 8px;
        }
        #fhr-slider {
            -webkit-appearance: none;
            height: 6px;
            background: var(--card);
            border-radius: 3px;
            outline: none;
        }
        #fhr-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }

        /* Toggle button group */
        .toggle-group {
            display: flex;
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
        }
        .toggle-btn {
            background: var(--card);
            color: var(--muted);
            border: none;
            padding: 6px 12px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.15s ease;
        }
        .toggle-btn:not(:last-child) {
            border-right: 1px solid var(--border);
        }
        .toggle-btn:hover {
            color: var(--text);
        }
        .toggle-btn.active {
            background: var(--accent);
            color: #000;
        }
        .toggle-btn.anomaly-active {
            background: #FF6D00;
            color: #fff;
            font-weight: bold;
        }

        /* Smaller selects for vscale and ytop */
        #vscale-select, #ytop-select {
            min-width: 80px;
        }

        /* Favorites group */
        .favorites-group {
            display: flex;
            gap: 4px;
            align-items: center;
        }
        #favorites-select {
            min-width: 120px;
            max-width: 180px;
        }
        #save-favorite-btn {
            padding: 4px 8px;
            font-size: 14px;
        }

        /* Memory indicator */
        #memory-status {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--muted);
            font-size: 12px;
        }
        .mem-bar {
            width: 60px;
            height: 6px;
            background: var(--card);
            border-radius: 3px;
            overflow: hidden;
        }
        .mem-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.3s ease;
        }

        /* RAM status modal */
        #memory-status { cursor: pointer; }
        #memory-status:hover { color: var(--text); }
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.6);
            z-index: 10000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.visible { display: flex; }
        .modal {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            min-width: 360px;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .modal h3 {
            margin: 0 0 12px 0;
            font-size: 15px;
            color: var(--text);
        }
        .modal .close-btn {
            float: right;
            background: none;
            border: none;
            color: var(--muted);
            font-size: 18px;
            cursor: pointer;
            padding: 0 4px;
        }
        .modal .close-btn:hover { color: var(--text); }
        .modal table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .modal th {
            text-align: left;
            padding: 6px 8px;
            border-bottom: 1px solid var(--border);
            color: var(--muted);
            font-weight: 600;
        }
        .modal td {
            padding: 5px 8px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .modal .cycle-group {
            color: var(--accent);
            font-weight: 600;
        }
        .modal .summary {
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--muted);
        }

        /* Toast notifications */
        #toast-container {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .toast {
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideUp 0.3s ease;
        }
        .toast.loading { border-left: 3px solid var(--warning); }
        .toast.success { border-left: 3px solid var(--success); }
        .toast.error { border-left: 3px solid #ef4444; }
        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        @keyframes pulse { 50% { opacity: 0.6; } }

        /* Progress panel */
        #progress-panel {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0;
            min-width: 340px;
            max-width: 420px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            display: none;
            overflow: hidden;
        }
        #progress-panel.visible { display: block; animation: slideUp 0.3s ease; }
        #progress-panel.collapsed .progress-items,
        #progress-panel.collapsed .progress-footer { display: none; }
        .progress-header {
            padding: 8px 14px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--muted);
            border-bottom: 1px solid var(--border);
            background: rgba(255,255,255,0.03);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }
        .progress-header:hover { background: rgba(255,255,255,0.06); }
        .progress-header-left { display: flex; align-items: center; gap: 8px; }
        .progress-badge {
            background: var(--accent);
            color: #fff;
            font-size: 10px;
            font-weight: 700;
            min-width: 18px;
            height: 18px;
            border-radius: 9px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0 5px;
        }
        .progress-badge.done-badge { background: var(--success); }
        .progress-collapse-icon {
            font-size: 10px;
            color: var(--muted);
            transition: transform 0.2s ease;
        }
        #progress-panel.collapsed .progress-collapse-icon { transform: rotate(180deg); }
        .progress-items { padding: 4px 0; max-height: 320px; overflow-y: auto; }
        .progress-item {
            padding: 8px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .progress-item:last-child { border-bottom: none; }
        .progress-item.done { opacity: 0.5; }
        .progress-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .cancel-op-btn {
            background: none;
            border: none;
            color: #f66;
            cursor: pointer;
            font-size: 13px;
            padding: 0 0 0 6px;
            line-height: 1;
            opacity: 0.7;
        }
        .cancel-op-btn:hover { opacity: 1; }
        .progress-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 220px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .progress-label .op-icon {
            font-size: 11px;
            flex-shrink: 0;
        }
        .progress-stats {
            font-size: 11px;
            color: var(--muted);
            white-space: nowrap;
        }
        .progress-bar-bg {
            height: 4px;
            background: var(--card);
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 3px;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.4s ease;
            background: var(--accent);
        }
        .progress-item.done .progress-bar-fill { background: var(--success); }
        .progress-item[data-op="preload"] .progress-bar-fill { background: #6366f1; }
        .progress-item[data-op="download"] .progress-bar-fill { background: #f59e0b; }
        .progress-item[data-op="prerender"] .progress-bar-fill { background: #8b5cf6; }
        .progress-item[data-op="autoload"] .progress-bar-fill { background: #818cf8; }
        .progress-item[data-op="autoupdate"] .progress-bar-fill { background: #06b6d4; }
        .progress-detail {
            font-size: 11px;
            color: var(--muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: flex;
            justify-content: space-between;
        }
        .progress-detail .eta {
            color: var(--accent);
            font-weight: 500;
            flex-shrink: 0;
            margin-left: 8px;
        }
        .progress-footer {
            padding: 6px 14px;
            font-size: 11px;
            color: var(--muted);
            border-top: 1px solid var(--border);
            background: rgba(255,255,255,0.02);
            display: flex;
            justify-content: space-between;
        }

        button {
            background: var(--card);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
        }
        button:hover { background: var(--accent); color: #000; }

        #sidebar {
            width: 55%;
            min-width: 500px;
            background: var(--panel);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        #xsect-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        #active-fhr { color: var(--accent); font-size: 13px; }
        #xsect-container {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #xsect-img {
            max-width: 100%;
            max-height: 100%;
            border-radius: 4px;
        }
        #instructions {
            color: var(--muted);
            text-align: center;
            padding: 20px;
        }
        .loading-text {
            color: var(--accent);
            animation: pulse 1.5s infinite;
        }

        /* Cycle Comparison Mode */
        #xsect-panels {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        #xsect-panels.compare-active {
            flex-direction: row;
        }
        .xsect-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            min-width: 0;
        }
        .xsect-panel-label {
            padding: 4px 12px;
            font-size: 11px;
            color: var(--muted);
            border-bottom: 1px solid var(--border);
            display: none;
        }
        #xsect-panels.compare-active .xsect-panel-label {
            display: block;
        }
        #xsect-panels.compare-active .xsect-panel + .xsect-panel {
            border-left: 1px solid var(--border);
        }
        .xsect-panel-body {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px;
            overflow: hidden;
        }
        .xsect-panel-body img {
            max-width: 100%;
            max-height: 100%;
            border-radius: 4px;
        }
        #compare-controls {
            display: none;
            align-items: center;
            gap: 8px;
            padding: 0 16px 8px;
        }
        #compare-controls.visible {
            display: flex;
        }
        #compare-controls select {
            font-size: 12px;
        }
        #compare-controls .toggle-group {
            display: flex;
            gap: 2px;
        }
        #compare-controls .toggle-btn {
            font-size: 11px;
            padding: 2px 8px;
        }

        /* Inset map for cross-section path */
        /* Explainer Modal */
        #explainer-modal {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 10001;
            align-items: center;
            justify-content: center;
        }
        #explainer-modal.visible { display: flex; }
        .modal-content {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            width: 90%;
            max-width: 700px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }
        .modal-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            background: var(--panel);
            z-index: 1;
        }
        .modal-header h2 { margin: 0; font-size: 18px; }
        .modal-close {
            background: none;
            border: none;
            color: var(--muted);
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }
        .modal-close:hover { color: var(--text); background: none; }
        .modal-body { padding: 16px 20px; }

        .param-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .param-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 8px;
        }
        .param-name {
            font-weight: 600;
            color: var(--accent);
            font-size: 15px;
        }
        .param-desc {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
        }
        .param-tech {
            color: var(--text);
            font-size: 12px;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--border);
            font-family: monospace;
        }

        /* Voting */
        .vote-buttons {
            display: flex;
            gap: 4px;
            align-items: center;
        }
        .vote-btn {
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--muted);
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.15s;
        }
        .vote-btn:hover { border-color: var(--accent); color: var(--text); background: var(--bg); }
        .vote-btn.voted-up { background: #22c55e33; border-color: #22c55e; color: #22c55e; }
        .vote-btn.voted-down { background: #ef444433; border-color: #ef4444; color: #ef4444; }
        .vote-count { font-size: 12px; min-width: 20px; text-align: center; }

        .help-btn, .request-btn {
            background: var(--card);
            border: 1px solid var(--border);
            color: var(--accent);
            width: 28px;
            height: 28px;
            border-radius: 50%;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .help-btn:hover, .request-btn:hover { background: var(--accent); color: #000; }
        .request-btn { color: var(--warning); }
        .request-btn:hover { background: var(--warning); }

        /* Request Modal */
        #request-modal {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 10001;
            align-items: center;
            justify-content: center;
        }
        #request-modal.visible { display: flex; }
        #run-request-modal {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 10002;
            align-items: center;
            justify-content: center;
        }
        #run-request-modal.visible { display: flex; }
        .request-form {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .request-form textarea {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            padding: 12px;
            font-size: 14px;
            resize: vertical;
            min-height: 100px;
            font-family: inherit;
        }
        .request-form textarea:focus {
            outline: 2px solid var(--accent);
            outline-offset: 1px;
        }
        .request-form input {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            padding: 10px 12px;
            font-size: 14px;
        }
        .request-form input:focus {
            outline: 2px solid var(--accent);
            outline-offset: 1px;
        }
        .submit-btn {
            background: var(--accent);
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 14px;
        }
        .submit-btn:hover { background: #7dd3fc; }
        .submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .request-list {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 16px;
            border-top: 1px solid var(--border);
            padding-top: 16px;
        }
        .request-item {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
        }
        .request-item-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 12px;
            color: var(--muted);
        }
        .request-item-text {
            font-size: 14px;
            line-height: 1.4;
        }

        /* Credit footer */
        #credit {
            position: fixed;
            bottom: 8px;
            right: 12px;
            font-size: 11px;
            color: var(--muted);
            opacity: 0.7;
            z-index: 1000;
            text-align: right;
            line-height: 1.4;
        }
        #credit a {
            color: var(--accent);
            text-decoration: none;
        }
        #credit a:hover {
            text-decoration: underline;
        }

    </style>
</head>
<body>
    <div id="map-container">
        <div id="controls">
            <div class="control-row">
                <div class="control-group">
                    <label>Model:</label>
                    <select id="model-select"></select>
                </div>
                <div class="control-group">
                    <label>Run:</label>
                    <select id="cycle-select"></select>
                </div>
                <div class="control-group">
                    <label>Style:</label>
                    <select id="style-select"></select>
                    <select id="temp-cmap-select" style="display:none" title="Temperature color table">
                        <option value="standard">Standard</option>
                        <option value="nws_ndfd">NWS Classic</option>
                        <option value="white_zero">White at 0°C</option>
                        <option value="green_purple">Green-Purple</option>
                    </select>
                </div>
                <div class="control-group favorites-group">
                    <select id="favorites-select">
                        <option value="">Favorites</option>
                    </select>
                    <button id="save-favorite-btn" title="Save current view" style="padding:3px 6px;font-size:12px;">Save</button>
                </div>
                <button id="swap-btn" title="Swap start/end points" style="padding:3px 8px;font-size:12px;">Swap</button>
                <button id="clear-btn" style="padding:3px 8px;font-size:12px;">Clear</button>
                <button id="more-toggle" title="Show/hide more options">More &#9660;</button>
            </div>
            <div class="control-row-secondary" id="secondary-controls">
                <div class="control-group" id="anomaly-group" style="display:none">
                    <label>Mode:</label>
                    <div class="toggle-group">
                        <button class="toggle-btn active" id="anomaly-off">Raw</button>
                        <button class="toggle-btn" id="anomaly-on">5yr Dep</button>
                    </div>
                </div>
                <div class="control-group">
                    <label>Y-Axis:</label>
                    <div class="toggle-group">
                        <button class="toggle-btn active" id="yaxis-pressure">hPa</button>
                        <button class="toggle-btn" id="yaxis-height">km</button>
                    </div>
                </div>
                <div class="control-group">
                    <label>V-Scale:</label>
                    <select id="vscale-select" style="min-width:60px;">
                        <option value="0.5">0.5x</option>
                        <option value="1.0" selected>1x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2.0">2x</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Top:</label>
                    <select id="ytop-select" style="min-width:80px;">
                        <option value="100" selected>100 hPa</option>
                        <option value="200">200 hPa</option>
                        <option value="300">300 hPa</option>
                        <option value="500">500 hPa</option>
                        <option value="700">700 hPa</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Units:</label>
                    <select id="units-select" style="min-width:55px;">
                        <option value="km" selected>km</option>
                        <option value="mi">mi</option>
                    </select>
                </div>
                <button class="help-btn" id="help-btn" title="Style explanations & feedback">?</button>
                <button id="request-cycle-btn" title="Request a specific init cycle" style="padding:3px 8px;font-size:12px;">Request Run</button>
                <button id="gif-btn" title="Generate animated GIF" style="padding:3px 8px;font-size:12px;">GIF</button>
                <select id="gif-speed" title="GIF speed" style="min-width:55px;">
                    <option value="1">1x</option>
                    <option value="0.75">0.75x</option>
                    <option value="0.5" selected>0.5x</option>
                    <option value="0.25">0.25x</option>
                </select>
                <button id="load-all-btn" title="Load all FHRs for current cycle" style="padding:3px 8px;font-size:12px;">Load All</button>
                <button id="compare-btn" title="Compare two init cycles" style="padding:3px 8px;font-size:12px;">Compare</button>
                <button id="admin-key-btn" title="Enter admin key" style="padding:3px 8px;font-size:12px;">Admin</button>
                <div id="memory-status">
                    <span id="mem-text">0 MB</span>
                    <div class="mem-bar"><div class="mem-fill" id="mem-fill" style="width:0%"></div></div>
                </div>
            </div>
            <div class="control-row">
                <label>FHR:</label>
                <div class="chip-group" id="fhr-chips"></div>
            </div>
            <div class="control-row" id="slider-row" style="display:none;">
                <button id="prev-btn" title="Previous frame" style="padding:3px 6px;font-size:12px;min-width:28px;">&#9664;</button>
                <button id="play-btn" title="Auto-play" style="padding:3px 8px;font-size:14px;min-width:32px;">&#9654;</button>
                <button id="next-btn" title="Next frame" style="padding:3px 6px;font-size:12px;min-width:28px;">&#9654;</button>
                <input type="range" id="fhr-slider" min="0" max="18" value="0" style="flex:1;">
                <span id="slider-label" style="font-size:11px;color:var(--muted);min-width:32px;text-align:center;">F00</span>
                <select id="play-speed" title="Playback speed" style="min-width:50px;font-size:11px;">
                    <option value="2000">0.5x</option>
                    <option value="1000" selected>1x</option>
                    <option value="500">2x</option>
                    <option value="250">4x</option>
                </select>
                <button id="prerender-btn" title="Pre-render all frames" style="padding:3px 6px;font-size:11px;">Pre-render</button>
            </div>
        </div>
        <div id="map"></div>
    </div>
    <div id="sidebar" style="position: relative;">
        <div id="xsect-header">
            <span>Cross-Section</span>
            <span id="active-fhr"></span>
        </div>
        <div id="compare-controls">
            <label style="font-size:12px;color:var(--muted);">vs</label>
            <select id="compare-cycle-select" style="min-width:120px;"></select>
            <div class="toggle-group" id="compare-mode-toggle">
                <button class="toggle-btn active" data-value="same_fhr">Same FHR</button>
                <button class="toggle-btn" data-value="valid_time">Valid Time</button>
            </div>
            <span id="compare-fhr-label" style="font-size:11px;color:var(--muted);"></span>
        </div>
        <div id="xsect-panels">
            <div class="xsect-panel" id="panel-primary">
                <div class="xsect-panel-label" id="panel-primary-label"></div>
                <div class="xsect-panel-body" id="xsect-container">
                    <div id="instructions">
                        Click two points on the map to draw a cross-section line.<br>
                        Then select forecast hours to load.
                    </div>
                </div>
            </div>
            <div class="xsect-panel" id="panel-compare" style="display:none;">
                <div class="xsect-panel-label" id="panel-compare-label"></div>
                <div class="xsect-panel-body" id="xsect-container-compare">
                    <div style="color:var(--muted);">Select a comparison cycle</div>
                </div>
            </div>
        </div>
    </div>
    <div id="toast-container"></div>
    <div id="progress-panel">
        <div class="progress-header" id="progress-header">
            <span class="progress-header-left">Activity <span class="progress-badge" id="progress-badge">0</span></span>
            <span class="progress-collapse-icon">&#9660;</span>
        </div>
        <div class="progress-items" id="progress-items"></div>
        <div class="progress-footer" id="progress-footer"></div>
    </div>

    <!-- Explainer Modal -->
    <div id="explainer-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Cross-Section Styles - Help & Feedback</h2>
                <button class="modal-close" id="modal-close">&times;</button>
            </div>
            <div class="modal-body" id="modal-body">
                <!-- Populated by JavaScript -->
            </div>
        </div>
    </div>

    <!-- Request Modal -->
    <div id="request-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>💡 Feature Requests & Feedback</h2>
                <button class="modal-close" id="request-modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <form class="request-form" id="request-form">
                    <input type="text" id="request-name" placeholder="Your name (optional)">
                    <textarea id="request-text" placeholder="Describe your feature request, bug report, or feedback..." required></textarea>
                    <button type="submit" class="submit-btn">Submit Request</button>
                </form>
                <div class="request-list" id="request-list">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
        </div>
    </div>

    <div id="run-request-modal">
        <div class="modal-content" style="max-width:340px;padding:20px;">
            <div class="modal-header" style="margin-bottom:12px;">
                <h2 style="font-size:16px;margin:0;">Request Archive Run</h2>
                <button class="modal-close" id="req-cancel">&times;</button>
            </div>
            <div style="display:flex;flex-direction:column;gap:10px;">
                <div>
                    <label style="font-size:12px;color:var(--text-dim);display:block;margin-bottom:3px;">Date</label>
                    <input type="date" id="req-date" style="width:100%;padding:6px 8px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:14px;">
                </div>
                <div>
                    <label style="font-size:12px;color:var(--text-dim);display:block;margin-bottom:3px;">Init Hour (UTC)</label>
                    <select id="req-hour" style="width:100%;padding:6px 8px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:14px;">
                        <option value="0">00z</option><option value="1">01z</option><option value="2">02z</option>
                        <option value="3">03z</option><option value="4">04z</option><option value="5">05z</option>
                        <option value="6">06z</option><option value="7">07z</option><option value="8">08z</option>
                        <option value="9">09z</option><option value="10">10z</option><option value="11">11z</option>
                        <option value="12" selected>12z</option><option value="13">13z</option><option value="14">14z</option>
                        <option value="15">15z</option><option value="16">16z</option><option value="17">17z</option>
                        <option value="18">18z</option><option value="19">19z</option><option value="20">20z</option>
                        <option value="21">21z</option><option value="22">22z</option><option value="23">23z</option>
                    </select>
                </div>
                <div style="display:flex;gap:8px;align-items:end;">
                    <div style="flex:1;">
                        <label style="font-size:12px;color:var(--text-dim);display:block;margin-bottom:3px;">FHR Start</label>
                        <input type="number" id="req-fhr-start" value="0" min="0" max="48" style="width:100%;padding:6px 8px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:14px;">
                    </div>
                    <span style="padding-bottom:8px;color:var(--text-dim);">to</span>
                    <div style="flex:1;">
                        <label style="font-size:12px;color:var(--text-dim);display:block;margin-bottom:3px;">FHR End <span id="req-fhr-max-hint" style="opacity:0.6;">(max 18)</span></label>
                        <input type="number" id="req-fhr-end" value="18" min="0" max="48" style="width:100%;padding:6px 8px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:14px;">
                    </div>
                </div>
                <button id="req-submit" style="padding:8px;background:var(--accent,#4a9eff);border:none;border-radius:4px;color:#fff;font-size:14px;cursor:pointer;font-weight:500;margin-top:4px;">Download & Load</button>
            </div>
        </div>
    </div>

    <div class="modal-overlay" id="ram-modal">
        <div class="modal">
            <button class="close-btn" id="ram-modal-close">&times;</button>
            <h3>RAM Status</h3>
            <div id="ram-modal-body"></div>
        </div>
    </div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const styles = ''' + json.dumps(XSECT_STYLES) + ''';
        const MAX_SELECTED = 4;  // Maximum forecast hours that can be loaded at once

        // State
        let startMarker = null, endMarker = null, line = null;
        let cycles = [];           // Available cycles from server
        let currentCycle = null;   // Currently selected cycle key
        let selectedFhrs = [];     // Currently selected/loaded forecast hours
        let activeFhr = null;      // Which FHR is currently displayed in cross-section
        let isAdmin = false;
        let protectedCycles = [];
        let currentModel = 'hrrr'; // Currently selected model
        let modelExcludedStyles = {};  // model -> Set of excluded style keys

        // Slider + playback state
        let isPlaying = false;
        let playInterval = null;
        let prerenderedFrames = {};  // fhr -> blobUrl

        // Comparison mode state
        let compareActive = false;
        let compareCycle = null;     // Cycle key for comparison panel
        let compareMode = 'same_fhr'; // 'same_fhr' or 'valid_time'

        // Model parameter helper — appends &model= to API calls
        function modelParam() { return `&model=${currentModel}`; }

        // Load available models from server and populate dropdown
        async function loadModels() {
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                const select = document.getElementById('model-select');
                select.innerHTML = '';
                (data.models || []).forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.name.toUpperCase();
                    select.appendChild(opt);
                    if (m.excluded_styles) {
                        modelExcludedStyles[m.id] = new Set(m.excluded_styles);
                    }
                });
                if (select.options.length > 0) {
                    currentModel = select.value;
                }
                select.onchange = async () => {
                    currentModel = select.value;
                    updateStyleDropdownForModel();
                    await loadCycles();
                    generateCrossSection();
                };
            } catch (e) {
                console.error('Failed to load models:', e);
                // Fallback: just show HRRR
                const select = document.getElementById('model-select');
                select.innerHTML = '<option value="hrrr">HRRR</option>';
            }
        }

        // Hide styles that aren't available for the current model
        function updateStyleDropdownForModel() {
            const excluded = modelExcludedStyles[currentModel] || new Set();
            const select = document.getElementById('style-select');
            const currentVal = select.value;
            Array.from(select.options).forEach(opt => {
                opt.style.display = excluded.has(opt.value) ? 'none' : '';
            });
            // If current selection is excluded, switch to first visible
            if (excluded.has(currentVal)) {
                const first = Array.from(select.options).find(o => !excluded.has(o.value));
                if (first) select.value = first.value;
            }
            updateTempCmapVisibility();
            updateAnomalyVisibility();
        }

        // Admin key management
        function getAdminKey() { return localStorage.getItem('wxsection_admin_key') || ''; }
        function adminParam() { const k = getAdminKey(); return k ? `&key=${encodeURIComponent(k)}` : ''; }
        async function checkAdminKey() {
            const k = getAdminKey();
            if (!k) { setAdminState(false, []); return; }
            try {
                const res = await fetch(`/api/check_key?key=${encodeURIComponent(k)}${modelParam()}`);
                const data = await res.json();
                setAdminState(data.valid, data.protected || []);
            } catch { setAdminState(false, []); }
        }
        function setAdminState(valid, prot) {
            isAdmin = valid;
            protectedCycles = prot;
            const btn = document.getElementById('admin-key-btn');
            btn.textContent = valid ? '🔓' : '🔒';
            btn.title = valid ? 'Admin mode active (click to change key)' : 'Enter admin key for archive access';
            document.getElementById('load-all-btn').style.display = '';  // Always visible (mmap makes loading cheap)
        }
        document.getElementById('admin-key-btn').onclick = async function() {
            const key = prompt(isAdmin ? 'Admin key (current key active, clear to revoke):' : 'Enter admin key:', getAdminKey());
            if (key === null) return;
            if (key === '') { localStorage.removeItem('wxsection_admin_key'); setAdminState(false, []); showToast('Key removed', 'success', 2000); return; }
            localStorage.setItem('wxsection_admin_key', key);
            await checkAdminKey();
            showToast(isAdmin ? 'Admin access granted' : 'Invalid key', isAdmin ? 'success' : 'error', 2000);
        };
        checkAdminKey();

        // Load All button — loads every FHR for current cycle
        document.getElementById('load-all-btn').onclick = async () => {
            if (!currentCycle) return;
            const btn = document.getElementById('load-all-btn');
            btn.disabled = true;
            btn.textContent = 'Loading...';
            const toast = showToast(`Loading all FHRs for ${currentCycle}...`);
            try {
                const res = await fetch(`/api/load_cycle?cycle=${currentCycle}${modelParam()}${adminParam()}`, {method: 'POST'});
                const data = await res.json();
                toast.remove();
                if (data.success) {
                    showToast(`Loaded ${data.loaded_fhrs} FHRs (${Math.round(data.memory_mb || 0)} MB)`, 'success');
                    await refreshLoadedStatus();
                    updateChipStates();
                } else {
                    showToast(data.error || 'Load failed', 'error');
                }
            } catch (err) {
                toast.remove();
                showToast('Load all failed: ' + err.message, 'error');
            }
            btn.disabled = false;
            btn.textContent = 'Load All';
        };

        // Initialize map
        const map = L.map('map', {
            center: [39, -98],
            zoom: 5,
            minZoom: 4,
            maxZoom: 10
        });
        L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap, &copy; OpenTopoMap'
        }).addTo(map);

        // Inset map removed - cross-section path shown in matplotlib plot inset

        // =========================================================================
        // Toast Notification System
        // =========================================================================
        function showToast(message, type = 'loading', duration = null) {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            const icon = type === 'loading' ? '⏳' : (type === 'success' ? '✓' : '✗');
            toast.innerHTML = `<span>${icon} ${message}</span>`;
            container.appendChild(toast);

            if (duration || type !== 'loading') {
                setTimeout(() => toast.remove(), duration || 3000);
            }
            return toast;
        }

        function updateMemoryDisplay(memMb) {
            const maxMem = 117000;  // 117GB memory cap
            const pct = Math.min(100, (memMb / maxMem) * 100);
            document.getElementById('mem-text').textContent = `${Math.round(memMb)} MB`;
            document.getElementById('mem-fill').style.width = `${pct}%`;
        }

        // =========================================================================
        // Style Selector
        // =========================================================================
        const styleSelect = document.getElementById('style-select');
        styles.forEach(([val, label]) => {
            const opt = document.createElement('option');
            opt.value = val;
            opt.textContent = label;
            styleSelect.appendChild(opt);
        });
        const tempCmapSelect = document.getElementById('temp-cmap-select');
        function updateTempCmapVisibility() {
            tempCmapSelect.style.display = styleSelect.value === 'temp' ? '' : 'none';
        }
        styleSelect.onchange = () => { updateTempCmapVisibility(); updateAnomalyVisibility(); generateCrossSection(); };
        tempCmapSelect.onchange = generateCrossSection;

        // =========================================================================
        // Anomaly Mode — disabled (kept for future use, always false)
        // =========================================================================
        let anomalyMode = false;
        let anomalyStyles = new Set();
        let climatologyAvailable = false;
        const anomalyGroup = document.getElementById('anomaly-group');
        const anomalyOffBtn = document.getElementById('anomaly-off');
        const anomalyOnBtn = document.getElementById('anomaly-on');

        // Handlers kept but anomaly group is permanently hidden
        anomalyOffBtn.onclick = () => {};
        anomalyOnBtn.onclick = () => {};

        function updateAnomalyVisibility() {
            // Anomaly mode disabled — always hidden
            anomalyGroup.style.display = 'none';
        }

        // Climatology fetch skipped — anomaly mode disabled

        // =========================================================================
        // Y-Axis Toggle (Pressure / Height)
        // =========================================================================
        let currentYAxis = 'pressure';
        const yaxisPressureBtn = document.getElementById('yaxis-pressure');
        const yaxisHeightBtn = document.getElementById('yaxis-height');

        yaxisPressureBtn.onclick = () => {
            if (currentYAxis !== 'pressure') {
                currentYAxis = 'pressure';
                yaxisPressureBtn.classList.add('active');
                yaxisHeightBtn.classList.remove('active');
                generateCrossSection();
            }
        };
        yaxisHeightBtn.onclick = () => {
            if (currentYAxis !== 'height') {
                currentYAxis = 'height';
                yaxisHeightBtn.classList.add('active');
                yaxisPressureBtn.classList.remove('active');
                generateCrossSection();
            }
        };

        // =========================================================================
        // Vertical Scale Selector
        // =========================================================================
        const vscaleSelect = document.getElementById('vscale-select');
        vscaleSelect.onchange = generateCrossSection;

        // =========================================================================
        // Y-Top (Vertical Range) Selector
        // =========================================================================
        const ytopSelect = document.getElementById('ytop-select');
        ytopSelect.onchange = generateCrossSection;

        // =========================================================================
        // Units (km/mi) Selector
        // =========================================================================
        const unitsSelect = document.getElementById('units-select');
        unitsSelect.onchange = generateCrossSection;

        // =========================================================================
        // Community Favorites
        // =========================================================================
        const favoritesSelect = document.getElementById('favorites-select');
        const saveFavoriteBtn = document.getElementById('save-favorite-btn');

        async function loadFavorites() {
            try {
                const res = await fetch('/api/favorites');
                const favorites = await res.json();
                favoritesSelect.innerHTML = '<option value="">⭐ Favorites (' + favorites.length + ')</option>';
                favorites.forEach(fav => {
                    const opt = document.createElement('option');
                    opt.value = JSON.stringify(fav);
                    opt.textContent = fav.name + (fav.label ? ' - ' + fav.label.substring(0, 30) : '');
                    opt.title = fav.label || fav.name;
                    favoritesSelect.appendChild(opt);
                });
            } catch (e) {
                console.error('Failed to load favorites:', e);
            }
        }

        favoritesSelect.onchange = function() {
            if (!this.value) return;
            try {
                const fav = JSON.parse(this.value);
                const cfg = fav.config;
                // Apply the favorite config
                if (cfg.start_lat && cfg.start_lon && cfg.end_lat && cfg.end_lon) {
                    // Clear existing markers/line first
                    if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
                    if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
                    if (line) { map.removeLayer(line); line = null; }

                    // Create start marker
                    startMarker = L.marker([cfg.start_lat, cfg.start_lon], {
                        draggable: true,
                        icon: L.divIcon({
                            className: 'marker-start',
                            html: '<div style="width:16px;height:16px;background:#38bdf8;border-radius:50%;border:2px solid white;"></div>',
                            iconSize: [16, 16],
                            iconAnchor: [8, 8]
                        })
                    }).addTo(map);
                    startMarker.on('drag', updateLine);
                    startMarker.on('dragend', () => { invalidatePrerender(); generateCrossSection(); });

                    // Create end marker
                    endMarker = L.marker([cfg.end_lat, cfg.end_lon], {
                        draggable: true,
                        icon: L.divIcon({
                            className: 'marker-end',
                            html: '<div style="width:16px;height:16px;background:#f87171;border-radius:50%;border:2px solid white;"></div>',
                            iconSize: [16, 16],
                            iconAnchor: [8, 8]
                        })
                    }).addTo(map);
                    endMarker.on('drag', updateLine);
                    endMarker.on('dragend', () => { invalidatePrerender(); generateCrossSection(); });

                    // Create line
                    line = L.polyline([[cfg.start_lat, cfg.start_lon], [cfg.end_lat, cfg.end_lon]], {
                        color: '#f59e0b', weight: 3, opacity: 0.9
                    }).addTo(map);

                    map.fitBounds([[cfg.start_lat, cfg.start_lon], [cfg.end_lat, cfg.end_lon]], {padding: [50, 50]});
                }
                if (cfg.style) document.getElementById('style-select').value = cfg.style;
                if (cfg.y_axis) {
                    currentYAxis = cfg.y_axis;
                    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                    document.getElementById('yaxis-' + cfg.y_axis).classList.add('active');
                }
                if (cfg.vscale) document.getElementById('vscale-select').value = cfg.vscale;
                if (cfg.y_top) document.getElementById('ytop-select').value = cfg.y_top;
                this.value = '';  // Reset dropdown
                generateCrossSection();
                showToast('Loaded: ' + fav.name);
            } catch (e) {
                console.error('Failed to apply favorite:', e);
            }
        };

        saveFavoriteBtn.onclick = async function() {
            if (!startMarker._map || !endMarker._map) {
                showToast('Draw a cross-section first!', true);
                return;
            }
            const name = prompt('Name this favorite (e.g., "LA Basin East-West"):');
            if (!name) return;
            const label = prompt('Optional description (leave blank for none):') || '';

            const start = startMarker.getLatLng();
            const end = endMarker.getLatLng();
            const config = {
                start_lat: start.lat,
                start_lon: start.lng,
                end_lat: end.lat,
                end_lon: end.lng,
                style: document.getElementById('style-select').value,
                y_axis: currentYAxis,
                vscale: document.getElementById('vscale-select').value,
                y_top: document.getElementById('ytop-select').value
            };

            try {
                const res = await fetch('/api/favorite', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name, label, config})
                });
                if (res.ok) {
                    showToast('Saved: ' + name);
                    loadFavorites();
                } else {
                    showToast('Failed to save', true);
                }
            } catch (e) {
                showToast('Error saving favorite', true);
            }
        };

        // Load favorites on startup
        loadFavorites();

        // =========================================================================
        // Request Custom Date/Cycle — Modal Dialog
        // =========================================================================
        document.getElementById('request-cycle-btn').onclick = function() {
            if (!isAdmin) {
                showToast('Admin key required', 'error');
                return;
            }
            document.getElementById('run-request-modal').classList.add('visible');
            // Set date default to today
            const today = new Date().toISOString().slice(0, 10);
            document.getElementById('req-date').value = today;
            updateMaxFhr();
        };
        document.getElementById('req-cancel').onclick = function() {
            document.getElementById('run-request-modal').classList.remove('visible');
        };
        // Close on backdrop click
        document.getElementById('run-request-modal').addEventListener('click', function(e) {
            if (e.target === this) this.classList.remove('visible');
        });

        // Update max FHR when hour changes (HRRR synoptic = 48)
        function updateMaxFhr() {
            const hour = parseInt(document.getElementById('req-hour').value) || 0;
            const isSynoptic = [0, 6, 12, 18].includes(hour);
            const model = currentModel;
            let maxFhr = 18;
            if (model === 'hrrr' && isSynoptic) maxFhr = 48;
            else if (model === 'gfs') maxFhr = 384;
            const endInput = document.getElementById('req-fhr-end');
            endInput.max = maxFhr;
            if (parseInt(endInput.value) > maxFhr) endInput.value = maxFhr;
            document.getElementById('req-fhr-max-hint').textContent = `(max ${maxFhr})`;
        }
        document.getElementById('req-hour').addEventListener('change', updateMaxFhr);

        // Submit request
        document.getElementById('req-submit').onclick = async function() {
            const dateStr = document.getElementById('req-date').value.replace(/-/g, '');
            const hour = parseInt(document.getElementById('req-hour').value);
            const fhrStart = parseInt(document.getElementById('req-fhr-start').value);
            const fhrEnd = parseInt(document.getElementById('req-fhr-end').value);

            if (!dateStr || dateStr.length !== 8) { showToast('Invalid date', 'error'); return; }
            if (isNaN(hour)) { showToast('Select an init hour', 'error'); return; }
            if (fhrStart > fhrEnd) { showToast('Start FHR must be <= End FHR', 'error'); return; }

            document.getElementById('run-request-modal').classList.remove('visible');
            const label = `${dateStr}/${String(hour).padStart(2,'0')}z F${String(fhrStart).padStart(2,'0')}-F${String(fhrEnd).padStart(2,'0')}`;
            showToast(`Requesting ${label}...`, 'success', 3000);

            try {
                const res = await fetch(
                    `/api/request_cycle?date=${dateStr}&hour=${hour}&fhr_start=${fhrStart}&fhr_end=${fhrEnd}${modelParam()}${adminParam()}`,
                    {method: 'POST'}
                );
                const data = await res.json();
                if (!data.success) {
                    showToast(data.error || 'Request failed', 'error');
                }
                // Progress is shown in the progress panel now — no need for separate polling
            } catch (e) {
                showToast('Request failed', 'error');
            }
        };

        // =========================================================================
        // Cycle (Model Run) Selector
        // =========================================================================
        const cycleSelect = document.getElementById('cycle-select');

        function buildCycleDropdown(cycleList, preserveSelection) {
            const savedCycle = preserveSelection ? currentCycle : null;
            cycleSelect.innerHTML = '';

            if (cycleList.length === 0) {
                const opt = document.createElement('option');
                opt.textContent = 'No data available';
                cycleSelect.appendChild(opt);
                return;
            }

            // Group by date
            const groups = {};
            cycleList.forEach(c => {
                const d = c.date || c.key.split('_')[0];
                if (!groups[d]) groups[d] = [];
                groups[d].push(c);
            });

            Object.keys(groups).sort().reverse().forEach(date => {
                const formatted = date.slice(0,4)+'-'+date.slice(4,6)+'-'+date.slice(6,8);
                const grp = document.createElement('optgroup');
                grp.label = formatted;
                groups[date].forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c.key;
                    const status = c.loaded ? '●' : '○';
                    opt.textContent = `${status} ${c.display} (${c.fhr_count} FHRs)`;
                    opt.dataset.fhrs = JSON.stringify(c.fhrs);
                    opt.dataset.loaded = c.loaded ? 'true' : 'false';
                    grp.appendChild(opt);
                });
                cycleSelect.appendChild(grp);
            });

            // Restore selection if it still exists
            if (savedCycle) {
                const exists = Array.from(cycleSelect.options).some(o => o.value === savedCycle);
                if (exists) {
                    cycleSelect.value = savedCycle;
                    return;
                }
            }

            // Otherwise select first
            if (cycleList.length > 0) {
                cycleSelect.value = cycleList[0].key;
                currentCycle = cycleList[0].key;
            }
        }

        async function loadCycles() {
            try {
                const res = await fetch(`/api/cycles?model=${currentModel}`);
                const data = await res.json();
                cycles = data.cycles || [];

                buildCycleDropdown(cycles, false);

                if (cycles.length === 0) return;

                currentCycle = cycles[0].key;

                // Check what's already loaded, then render chips
                await refreshLoadedStatus();

                // If first cycle has loaded FHRs, auto-select first one
                if (selectedFhrs.length > 0) {
                    activeFhr = selectedFhrs[0];
                    document.getElementById('active-fhr').textContent = `F${String(activeFhr).padStart(2,'0')}`;
                }
                renderFhrChips(cycles[0].fhrs);
            } catch (err) {
                console.error('Failed to load cycles:', err);
            }
        }

        // Auto-refresh cycles every 60s to pick up newly downloaded forecast hours
        async function refreshCycleList() {
            try {
                const res = await fetch(`/api/cycles?model=${currentModel}`);
                const data = await res.json();
                const newCycles = data.cycles || [];
                if (!newCycles.length) return;

                // Check if anything changed at all
                const oldKeys = cycles.map(c => c.key + ':' + c.fhr_count).join(',');
                const newKeys = newCycles.map(c => c.key + ':' + c.fhr_count).join(',');
                if (oldKeys === newKeys) return;  // Nothing changed

                // Update FHR chips if current cycle got new forecast hours
                const currentInfo = newCycles.find(c => c.key === currentCycle);
                const oldInfo = cycles.find(c => c.key === currentCycle);
                if (currentInfo && oldInfo) {
                    const newFhrs = JSON.stringify(currentInfo.fhrs);
                    const oldFhrs = JSON.stringify(oldInfo.fhrs);
                    if (newFhrs !== oldFhrs) {
                        renderFhrChips(currentInfo.fhrs);
                    }
                }

                cycles = newCycles;
                buildCycleDropdown(cycles, true);  // Always preserve selection
                if (compareActive) populateCompareCycleDropdown();
            } catch (e) {
                // Silent fail for background refresh
            }
        }
        setInterval(() => { refreshCycleList(); refreshLoadedStatus(); }, 60000);

        // === Progress Panel ===
        const OP_ICONS = {
            preload: '\\u25B6',  // play triangle
            load: '\\u2191',     // up arrow
            autoload: '\\u25B6', // play triangle (same as preload)
            prerender: '\\u25CF',// filled circle
            download: '\\u2193', // down arrow
            autoupdate: '\\u21BB', // clockwise arrow ↻
        };

        function fmtTime(sec) {
            if (sec >= 3600) return `${Math.floor(sec/3600)}h ${Math.floor((sec%3600)/60)}m`;
            if (sec >= 60) return `${Math.floor(sec/60)}m ${sec%60}s`;
            return `${sec}s`;
        }

        // Collapse/expand toggle
        document.getElementById('progress-header').addEventListener('click', () => {
            document.getElementById('progress-panel').classList.toggle('collapsed');
        });

        async function pollProgress() {
            try {
                const res = await fetch('/api/progress');
                const data = await res.json();
                const panel = document.getElementById('progress-panel');
                const container = document.getElementById('progress-items');
                const footer = document.getElementById('progress-footer');
                const badge = document.getElementById('progress-badge');
                const entries = Object.entries(data);

                if (entries.length === 0) {
                    panel.classList.remove('visible');
                    return;
                }

                panel.classList.add('visible');
                container.innerHTML = '';

                let activeCount = 0;
                let allDone = true;

                for (const [opId, info] of entries) {
                    if (!info.done) { activeCount++; allDone = false; }

                    const item = document.createElement('div');
                    item.className = 'progress-item' + (info.done ? ' done' : '');
                    item.setAttribute('data-op', info.op || '');

                    const icon = OP_ICONS[info.op] || '\\u2022';  // bullet default
                    const timeStr = fmtTime(info.elapsed);

                    // ETA string
                    let etaStr = '';
                    if (info.eta && !info.done) {
                        etaStr = `<span class="eta">${fmtTime(info.eta)} left</span>`;
                    } else if (info.done) {
                        etaStr = `<span class="eta" style="color:var(--success)">done</span>`;
                    }

                    // Rate string
                    let rateStr = '';
                    if (info.rate && !info.done) {
                        rateStr = ` · ${info.rate.toFixed(1)}/s`;
                    }

                    // If detail is "Starting..." and elapsed > 10s, show converting hint
                    let detailText = info.detail;
                    if (detailText === 'Starting...' && info.elapsed > 10 && !info.done) {
                        detailText = 'Converting GRIB files to cache...';
                    }

                    // Cancel button for admins on active pre-render and download jobs
                    let cancelBtn = '';
                    if (isAdmin && !info.done && info.detail !== 'Cancelling...' && (info.op === 'prerender' || info.op === 'download')) {
                        cancelBtn = `<button class="cancel-op-btn" data-op="${opId}" title="Cancel">\u2715</button>`;
                    }

                    item.innerHTML = `
                        <div class="progress-item-header">
                            <span class="progress-label"><span class="op-icon">${icon}</span>${info.label}</span>
                            <span class="progress-stats">${info.step}/${info.total}${rateStr} · ${timeStr}${cancelBtn}</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width:${info.pct}%"></div>
                        </div>
                        <div class="progress-detail"><span>${detailText}</span>${etaStr}</div>
                    `;
                    container.appendChild(item);
                }

                // Badge
                badge.textContent = activeCount > 0 ? activeCount : '\\u2713';
                badge.className = 'progress-badge' + (allDone ? ' done-badge' : '');

                // Footer summary
                try {
                    const statusRes = await fetch(`/api/status?model=${currentModel}`);
                    const status = await statusRes.json();
                    const loadedCount = (status.loaded || []).length;
                    const memMb = Math.round(status.memory_mb || 0);
                    footer.innerHTML = `<span>${loadedCount} FHRs loaded</span><span>${memMb} MB</span>`;
                } catch(e) {
                    footer.innerHTML = '';
                }

                // Also update memory display from any active load
                refreshLoadedStatus();
            } catch (e) {
                // Silent fail
            }
        }
        setInterval(pollProgress, 1500);
        pollProgress();

        // Cancel button handler (delegated)
        document.getElementById('progress-items').addEventListener('click', async (e) => {
            const btn = e.target.closest('.cancel-op-btn');
            if (!btn) return;
            const opId = btn.dataset.op;
            btn.disabled = true;
            btn.textContent = '...';
            try {
                await fetch(`/api/cancel?op_id=${encodeURIComponent(opId)}&key=${encodeURIComponent(getAdminKey())}`, {method: 'POST'});
            } catch(err) {
                console.error('Cancel failed:', err);
            }
        });

        cycleSelect.onchange = async () => {
            const selected = cycleSelect.options[cycleSelect.selectedIndex];
            currentCycle = selected.value;
            const fhrs = JSON.parse(selected.dataset.fhrs || '[]');
            const isLoaded = selected.dataset.loaded === 'true';

            if (!isLoaded) {
                // Need to load this cycle first
                const toast = showToast(`Loading cycle (this may take a minute)...`);
                try {
                    const res = await fetch(`/api/load_cycle?cycle=${currentCycle}${modelParam()}${adminParam()}`, {method: 'POST'});
                    const data = await res.json();
                    toast.remove();

                    if (data.success) {
                        showToast(`Loaded ${data.loaded_fhrs} forecast hours`, 'success');
                        selected.textContent = selected.textContent.replace(' ⏳', '');
                        selected.dataset.loaded = 'true';
                        updateMemoryDisplay(data.memory_mb || 0);

                        // Refresh cycles list to update loaded status
                        const cyclesRes = await fetch(`/api/cycles?model=${currentModel}`);
                        const cyclesData = await cyclesRes.json();
                        cycles = cyclesData.cycles || [];
                    } else {
                        showToast(data.error || 'Failed to load cycle', 'error');
                        return;
                    }
                } catch (err) {
                    toast.remove();
                    showToast('Failed to load cycle', 'error');
                    return;
                }
            }

            // Update loaded state and render chips
            await refreshLoadedStatus();
            renderFhrChips(fhrs);

            // Auto-select first FHR
            if (selectedFhrs.length > 0) {
                activeFhr = selectedFhrs[0];
                document.getElementById('active-fhr').textContent = `F${String(activeFhr).padStart(2,'0')}`;
                updateChipStates();
                generateCrossSection();
            }
        };

        // =========================================================================
        // Forecast Hour Chips (Redesigned: clear states, no accidental unloads)
        //
        // Visual states:
        //   - default (grey)  = downloaded on disk, not loaded to RAM
        //   - .loaded (green) = loaded in RAM, click for instant view
        //   - .active (blue)  = currently viewing this FHR
        //   - .loading (yellow pulse) = loading in progress
        //   - .unavailable (faded) = not downloaded yet
        //
        // Click behavior:
        //   - Click loaded/active chip = instant view switch (no load time)
        //   - Click unloaded chip = load to RAM (~15s), then view
        //   - Shift+click loaded chip = unload from RAM (deliberate only)
        // =========================================================================
        function renderFhrChips(availableFhrs) {
            const container = document.getElementById('fhr-chips');
            container.innerHTML = '';

            // Determine expected FHRs from current cycle metadata
            const cycleInfo = cycles.find(c => c.key === currentCycle);
            const maxFhr = (cycleInfo && cycleInfo.max_fhr) || 18;
            const isSynoptic = cycleInfo && cycleInfo.is_synoptic;
            const expectedFhrs = (cycleInfo && cycleInfo.expected_fhrs) || null;

            if (expectedFhrs) {
                // Use the model's actual FHR list (handles GFS every-6h, etc.)
                let addedDivider = false;
                for (const fhr of expectedFhrs) {
                    // Add divider before extended range (HRRR synoptic only)
                    if (!addedDivider && isSynoptic && fhr > 18) {
                        const divider = document.createElement('span');
                        divider.className = 'chip-divider';
                        divider.textContent = '|';
                        container.appendChild(divider);
                        addedDivider = true;
                    }
                    const chip = createFhrChip(fhr, availableFhrs);
                    if (fhr > 18 && isSynoptic) chip.classList.add('extended');
                    container.appendChild(chip);
                }
            } else {
                // Fallback: F00-F18 + extended F19-maxFhr for synoptic
                for (let fhr = 0; fhr <= Math.min(maxFhr, 18); fhr++) {
                    container.appendChild(createFhrChip(fhr, availableFhrs));
                }
                if (isSynoptic && maxFhr > 18) {
                    const divider = document.createElement('span');
                    divider.className = 'chip-divider';
                    divider.textContent = '|';
                    container.appendChild(divider);
                    for (let fhr = 19; fhr <= maxFhr; fhr++) {
                        const chip = createFhrChip(fhr, availableFhrs);
                        chip.classList.add('extended');
                        container.appendChild(chip);
                    }
                }
            }

            updateSliderVisibility();
        }

        function createFhrChip(fhr, availableFhrs) {
            const chip = document.createElement('div');
            chip.className = 'chip';
            chip.textContent = `F${String(fhr).padStart(2, '0')}`;
            chip.dataset.fhr = fhr;

            if (!availableFhrs.includes(fhr)) {
                chip.classList.add('unavailable');
                chip.title = 'Not downloaded yet';
            } else {
                // Set visual state based on loaded/active
                if (fhr === activeFhr) {
                    chip.classList.add('active');
                    chip.title = 'Currently viewing (Shift+click to unload)';
                } else if (selectedFhrs.includes(fhr)) {
                    chip.classList.add('loaded');
                    chip.title = 'Loaded in RAM — click for instant view (Shift+click to unload)';
                } else {
                    chip.title = 'Click to load (~15s)';
                }
                chip.onclick = (e) => handleChipClick(fhr, chip, e);
            }
            return chip;
        }

        // Unified click handler for all chips
        async function handleChipClick(fhr, chipEl, event) {
            if (chipEl.classList.contains('loading') || chipEl.classList.contains('unavailable')) {
                return;
            }

            const isLoaded = selectedFhrs.includes(fhr);

            // --- Shift+click = UNLOAD (deliberate action only) ---
            if (event.shiftKey && isLoaded) {
                chipEl.classList.add('loading');
                chipEl.classList.remove('loaded', 'active');
                const toast = showToast(`Unloading F${String(fhr).padStart(2,'0')}...`);

                try {
                    const res = await fetch(`/api/unload?cycle=${currentCycle}&fhr=${fhr}${modelParam()}${adminParam()}`, {method: 'POST'});
                    const data = await res.json();

                    if (data.success) {
                        selectedFhrs = selectedFhrs.filter(f => f !== fhr);
                        toast.remove();
                        showToast(`Unloaded F${String(fhr).padStart(2,'0')}`, 'success');
                        updateMemoryDisplay(data.memory_mb || 0);

                        if (activeFhr === fhr) {
                            activeFhr = selectedFhrs.length > 0 ? selectedFhrs[selectedFhrs.length - 1] : null;
                            if (activeFhr !== null) {
                                document.getElementById('active-fhr').textContent = `F${String(activeFhr).padStart(2,'0')}`;
                                generateCrossSection();
                            } else {
                                document.getElementById('xsect-container').innerHTML =
                                    '<div id="instructions">Select a forecast hour to view</div>';
                                document.getElementById('active-fhr').textContent = '';
                            }
                        }
                    } else {
                        toast.remove();
                        showToast(data.error || 'Unload failed', 'error');
                    }
                } catch (err) {
                    toast.remove();
                    showToast('Unload failed', 'error');
                }
                chipEl.classList.remove('loading');
                updateChipStates();
                return;
            }

            // --- Normal click on loaded chip = INSTANT VIEW SWITCH ---
            if (isLoaded) {
                activeFhr = fhr;
                document.getElementById('active-fhr').textContent = `F${String(fhr).padStart(2,'0')}`;
                updateChipStates();
                generateCrossSection();
                return;
            }

            // --- Normal click on unloaded chip = LOAD then VIEW ---
            chipEl.classList.add('loading');
            const toast = showToast(`Loading F${String(fhr).padStart(2,'0')}... (~15s)`);

            try {
                const loadStart = Date.now();
                const res = await fetch(`/api/load?cycle=${currentCycle}&fhr=${fhr}${modelParam()}${adminParam()}`, {method: 'POST'});
                const data = await res.json();
                const loadSec = ((Date.now() - loadStart) / 1000).toFixed(1);

                if (data.success) {
                    toast.remove();
                    const serverTime = data.load_time ? `${data.load_time}s` : `${loadSec}s`;
                    showToast(`Loaded F${String(fhr).padStart(2,'0')} in ${serverTime} (${Math.round(data.memory_mb || 0)} MB)`, 'success');

                    await refreshLoadedStatus();

                    activeFhr = fhr;
                    document.getElementById('active-fhr').textContent = `F${String(fhr).padStart(2,'0')}`;
                    generateCrossSection();
                } else {
                    toast.remove();
                    showToast(data.error || 'Load failed', 'error');
                }
            } catch (err) {
                toast.remove();
                showToast('Load failed', 'error');
            }
            chipEl.classList.remove('loading');
            updateChipStates();
        }

        // Update all chip visual states to match current data
        function updateChipStates() {
            document.querySelectorAll('#fhr-chips .chip').forEach(chip => {
                const fhr = parseInt(chip.dataset.fhr);
                if (chip.classList.contains('unavailable') || chip.classList.contains('loading')) return;

                chip.classList.remove('loaded', 'active');
                if (fhr === activeFhr) {
                    chip.classList.add('active');
                    chip.title = 'Currently viewing (Shift+click to unload)';
                } else if (selectedFhrs.includes(fhr)) {
                    chip.classList.add('loaded');
                    chip.title = 'Loaded in RAM — click for instant view (Shift+click to unload)';
                } else {
                    chip.title = 'Click to load (~15s)';
                }
            });
            updateSliderVisibility();
        }

        async function refreshLoadedStatus() {
            try {
                const res = await fetch(`/api/status?model=${currentModel}`);
                const data = await res.json();

                // Update selected FHRs based on what's actually loaded
                selectedFhrs = [];
                (data.loaded || []).forEach(item => {
                    if (item[0] === currentCycle) {
                        selectedFhrs.push(item[1]);
                    }
                });

                // Update chip UI with new state system
                updateChipStates();

                updateMemoryDisplay(data.memory_mb || 0);
            } catch (err) {
                console.error('Failed to refresh status:', err);
            }
        }

        // =========================================================================
        // Time Slider + Auto-Play
        // =========================================================================

        function updateSliderVisibility() {
            const sliderRow = document.getElementById('slider-row');
            if (selectedFhrs.length >= 2) {
                sliderRow.style.display = '';
                updateSliderRange();
            } else {
                sliderRow.style.display = 'none';
                stopPlayback();
            }
        }

        function updateSliderRange() {
            const slider = document.getElementById('fhr-slider');
            const sorted = [...selectedFhrs].sort((a, b) => a - b);
            slider.min = 0;
            slider.max = sorted.length - 1;
            const idx = sorted.indexOf(activeFhr);
            slider.value = idx >= 0 ? idx : 0;
            slider.dataset.fhrMap = JSON.stringify(sorted);
            document.getElementById('slider-label').textContent = activeFhr != null ? `F${String(activeFhr).padStart(2, '0')}` : '';
        }

        document.getElementById('fhr-slider').addEventListener('input', function() {
            const fhrMap = JSON.parse(this.dataset.fhrMap || '[]');
            const fhr = fhrMap[parseInt(this.value)];
            if (fhr === undefined) return;

            document.getElementById('slider-label').textContent = `F${String(fhr).padStart(2, '0')}`;
            activeFhr = fhr;
            updateChipStates();

            // Use prerendered frame if available
            if (prerenderedFrames[fhr]) {
                const container = document.getElementById('xsect-container');
                let img = document.getElementById('xsect-img');
                if (!img) {
                    img = document.createElement('img');
                    img.id = 'xsect-img';
                    img.style.maxWidth = '100%';
                    container.innerHTML = '';
                    container.appendChild(img);
                }
                img.src = prerenderedFrames[fhr];
                document.getElementById('active-fhr').textContent = `F${String(fhr).padStart(2, '0')}`;
                if (compareActive) { updateCompareLabels(); generateComparisonSection(); }
            } else {
                generateCrossSection();
            }
        });

        document.getElementById('play-btn').addEventListener('click', () => {
            if (isPlaying) {
                stopPlayback();
            } else {
                startPlayback();
            }
        });

        function stepFrame(delta) {
            stopPlayback();
            const slider = document.getElementById('fhr-slider');
            let val = parseInt(slider.value) + delta;
            if (val < 0) val = parseInt(slider.max);
            if (val > parseInt(slider.max)) val = 0;
            slider.value = val;
            slider.dispatchEvent(new Event('input'));
        }

        document.getElementById('prev-btn').addEventListener('click', () => stepFrame(-1));
        document.getElementById('next-btn').addEventListener('click', () => stepFrame(1));

        function startPlayback() {
            isPlaying = true;
            document.getElementById('play-btn').innerHTML = '&#9646;&#9646;';
            const speed = parseInt(document.getElementById('play-speed').value);
            const slider = document.getElementById('fhr-slider');

            playInterval = setInterval(() => {
                let val = parseInt(slider.value) + 1;
                if (val > parseInt(slider.max)) val = 0;
                slider.value = val;
                slider.dispatchEvent(new Event('input'));
            }, speed);
        }

        function stopPlayback() {
            isPlaying = false;
            document.getElementById('play-btn').innerHTML = '&#9654;';
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
        }

        function invalidatePrerender() {
            Object.values(prerenderedFrames).forEach(url => {
                if (url && url.startsWith('blob:')) URL.revokeObjectURL(url);
            });
            prerenderedFrames = {};
        }

        document.getElementById('prerender-btn').addEventListener('click', async () => {
            if (!startMarker || !endMarker || !currentCycle) return;

            const btn = document.getElementById('prerender-btn');
            btn.disabled = true;
            btn.textContent = 'Rendering...';

            const start = startMarker.getLatLng();
            const end = endMarker.getLatLng();
            const sorted = [...selectedFhrs].sort((a, b) => a - b);

            const body = {
                frames: sorted.map(fhr => ({cycle: currentCycle, fhr})),
                start: [start.lat, start.lng],
                end: [end.lat, end.lng],
                style: document.getElementById('style-select').value,
                y_axis: document.querySelector('#y-axis-toggle .toggle-btn.active')?.dataset?.value || 'pressure',
                vscale: parseFloat(document.getElementById('vscale-select').value),
                y_top: parseInt(document.getElementById('ytop-select').value),
                units: document.getElementById('units-select').value,
                temp_cmap: document.getElementById('temp-cmap-select')?.value || 'standard',
                anomaly: document.querySelector('#anomaly-toggle .toggle-btn.active')?.dataset?.value === 'anomaly',
                model: currentModel,
            };

            try {
                const res = await fetch('/api/prerender', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body),
                });
                const data = await res.json();
                const sessionId = data.session_id;

                // Poll progress until done
                const pollId = setInterval(async () => {
                    try {
                        const pRes = await fetch('/api/progress');
                        const progress = await pRes.json();
                        const session = progress[sessionId];

                        if (session) {
                            btn.textContent = `${session.pct}%`;
                        }

                        if (!session || session.done) {
                            clearInterval(pollId);
                            btn.disabled = false;
                            btn.textContent = 'Pre-render';

                            // Fetch all frames as blob URLs
                            const style = body.style;
                            const baseParams = `start_lat=${body.start[0]}&start_lon=${body.start[1]}&end_lat=${body.end[0]}&end_lon=${body.end[1]}&style=${style}&y_axis=${body.y_axis}&vscale=${body.vscale}&y_top=${body.y_top}&units=${body.units}&temp_cmap=${body.temp_cmap}&anomaly=${body.anomaly ? '1' : '0'}&model=${currentModel}`;

                            for (const fhr of sorted) {
                                try {
                                    const fRes = await fetch(`/api/frame?cycle=${currentCycle}&fhr=${fhr}&${baseParams}`);
                                    if (fRes.ok) {
                                        const blob = await fRes.blob();
                                        prerenderedFrames[fhr] = URL.createObjectURL(blob);
                                    }
                                } catch (e) { /* skip failed frames */ }
                            }
                            showToast(`${sorted.length} frames pre-rendered`, 'success');
                        }
                    } catch (e) {
                        clearInterval(pollId);
                        btn.disabled = false;
                        btn.textContent = 'Pre-render';
                    }
                }, 800);
            } catch (err) {
                btn.disabled = false;
                btn.textContent = 'Pre-render';
                showToast('Pre-render failed', 'error');
            }
        });

        // Invalidate prerendered frames when render params change
        ['style-select', 'vscale-select', 'ytop-select', 'units-select', 'temp-cmap-select'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', invalidatePrerender);
        });

        // =========================================================================
        // Cycle Comparison Mode
        // =========================================================================

        function toggleCompareMode() {
            compareActive = !compareActive;
            const btn = document.getElementById('compare-btn');
            const controls = document.getElementById('compare-controls');
            const panels = document.getElementById('xsect-panels');
            const panelCompare = document.getElementById('panel-compare');

            if (compareActive) {
                btn.style.background = 'var(--accent)';
                btn.style.color = '#000';
                controls.classList.add('visible');
                panels.classList.add('compare-active');
                panelCompare.style.display = '';
                populateCompareCycleDropdown();
                updateCompareLabels();
            } else {
                btn.style.background = '';
                btn.style.color = '';
                controls.classList.remove('visible');
                panels.classList.remove('compare-active');
                panelCompare.style.display = 'none';
                compareCycle = null;
                document.getElementById('xsect-container-compare').innerHTML =
                    '<div style="color:var(--muted);">Select a comparison cycle</div>';
            }
        }

        document.getElementById('compare-btn').addEventListener('click', toggleCompareMode);

        function populateCompareCycleDropdown() {
            const sel = document.getElementById('compare-cycle-select');
            sel.innerHTML = '<option value="">-- Select cycle --</option>';
            cycles.forEach(c => {
                if (c.key === currentCycle) return;
                const opt = document.createElement('option');
                opt.value = c.key;
                opt.textContent = c.label || c.key;
                sel.appendChild(opt);
            });
            if (compareCycle) sel.value = compareCycle;
        }

        document.getElementById('compare-cycle-select').addEventListener('change', function() {
            compareCycle = this.value || null;
            updateCompareLabels();
            generateComparisonSection();
        });

        // Compare mode toggle (Same FHR vs Valid Time)
        document.querySelectorAll('#compare-mode-toggle .toggle-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('#compare-mode-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                compareMode = this.dataset.value;
                updateCompareLabels();
                generateComparisonSection();
            });
        });

        function getCompareFhr() {
            if (!compareCycle || activeFhr === null) return null;

            if (compareMode === 'same_fhr') {
                return activeFhr;
            }

            // Valid Time mode: find FHR in comparison cycle that matches same valid time
            const primaryInfo = cycles.find(c => c.key === currentCycle);
            const compareInfo = cycles.find(c => c.key === compareCycle);
            if (!primaryInfo || !compareInfo) return activeFhr;

            // Parse cycle init hours from keys like "20260205_18z"
            const parseCycleTime = (key) => {
                const m = key.match(/(\d{8})_(\d{2})z/);
                if (!m) return null;
                const yr = parseInt(m[1].substring(0, 4));
                const mo = parseInt(m[1].substring(4, 6)) - 1;
                const dy = parseInt(m[1].substring(6, 8));
                const hr = parseInt(m[2]);
                return new Date(Date.UTC(yr, mo, dy, hr));
            };

            const primaryInit = parseCycleTime(currentCycle);
            const compareInit = parseCycleTime(compareCycle);
            if (!primaryInit || !compareInit) return activeFhr;

            // Valid time = init + FHR hours
            const validTime = new Date(primaryInit.getTime() + activeFhr * 3600000);
            const neededFhr = Math.round((validTime - compareInit) / 3600000);

            if (neededFhr < 0 || neededFhr > (compareInfo.max_fhr || 48)) return null;
            return neededFhr;
        }

        function updateCompareLabels() {
            const primaryLabel = document.getElementById('panel-primary-label');
            const compareLabel = document.getElementById('panel-compare-label');
            const fhrLabel = document.getElementById('compare-fhr-label');

            if (!compareActive) return;

            const primaryInfo = cycles.find(c => c.key === currentCycle);
            primaryLabel.textContent = (primaryInfo ? primaryInfo.label || currentCycle : currentCycle || '') +
                (activeFhr !== null ? ` F${String(activeFhr).padStart(2, '0')}` : '');

            if (compareCycle) {
                const cFhr = getCompareFhr();
                const compareInfo = cycles.find(c => c.key === compareCycle);
                const cLabel = compareInfo ? compareInfo.label || compareCycle : compareCycle;
                compareLabel.textContent = cLabel + (cFhr !== null ? ` F${String(cFhr).padStart(2, '0')}` : '');

                if (compareMode === 'valid_time' && cFhr !== null && activeFhr !== null) {
                    const primaryInit = parseCycleKey(currentCycle);
                    if (primaryInit) {
                        const vt = new Date(primaryInit.getTime() + activeFhr * 3600000);
                        fhrLabel.textContent = `Valid: ${vt.getUTCHours().toString().padStart(2,'0')}z`;
                    }
                } else {
                    fhrLabel.textContent = '';
                }
            } else {
                compareLabel.textContent = 'No cycle selected';
                fhrLabel.textContent = '';
            }
        }

        function parseCycleKey(key) {
            const m = key.match(/(\d{8})_(\d{2})z/);
            if (!m) return null;
            const yr = parseInt(m[1].substring(0, 4));
            const mo = parseInt(m[1].substring(4, 6)) - 1;
            const dy = parseInt(m[1].substring(6, 8));
            const hr = parseInt(m[2]);
            return new Date(Date.UTC(yr, mo, dy, hr));
        }

        async function generateComparisonSection() {
            if (!compareActive || !compareCycle || !startMarker || !endMarker) return;

            const container = document.getElementById('xsect-container-compare');
            const cFhr = getCompareFhr();

            if (cFhr === null) {
                container.innerHTML = '<div style="color:var(--muted);">FHR not available in comparison cycle</div>';
                return;
            }

            container.innerHTML = '<div class="loading-text">Generating...</div>';

            const start = startMarker.getLatLng();
            const end = endMarker.getLatLng();
            const style = document.getElementById('style-select').value;
            const vscale = document.getElementById('vscale-select').value;
            const ytop = document.getElementById('ytop-select').value;
            const units = document.getElementById('units-select').value;
            const tempCmap = document.getElementById('temp-cmap-select').value;

            // Use /api/frame for comparison (benefits from prerender cache)
            const url = `/api/frame?start_lat=${start.lat}&start_lon=${start.lng}` +
                `&end_lat=${end.lat}&end_lon=${end.lng}&cycle=${compareCycle}&fhr=${cFhr}&style=${style}` +
                `&y_axis=${currentYAxis}&vscale=${vscale}&y_top=${ytop}&units=${units}&temp_cmap=${tempCmap}` +
                `&anomaly=${anomalyMode ? 1 : 0}&model=${currentModel}`;

            try {
                const res = await fetch(url);
                if (!res.ok) throw new Error('Failed to generate comparison');
                const blob = await res.blob();
                const img = document.createElement('img');
                img.src = URL.createObjectURL(blob);
                container.innerHTML = '';
                container.appendChild(img);
            } catch (err) {
                container.innerHTML = `<div style="color:#f87171">${err.message}</div>`;
            }
        }

        // =========================================================================
        // Map Interaction
        // =========================================================================
        map.on('click', e => {
            const {lat, lng} = e.latlng;

            if (!startMarker) {
                startMarker = L.marker([lat, lng], {
                    draggable: true,
                    icon: L.divIcon({
                        className: 'marker-start',
                        html: '<div style="width:16px;height:16px;background:#38bdf8;border-radius:50%;border:2px solid white;"></div>',
                        iconSize: [16, 16],
                        iconAnchor: [8, 8]
                    })
                }).addTo(map);
                startMarker.on('drag', updateLine);
                startMarker.on('dragend', () => { invalidatePrerender(); generateCrossSection(); });
            } else if (!endMarker) {
                endMarker = L.marker([lat, lng], {
                    draggable: true,
                    icon: L.divIcon({
                        className: 'marker-end',
                        html: '<div style="width:16px;height:16px;background:#f87171;border-radius:50%;border:2px solid white;"></div>',
                        iconSize: [16, 16],
                        iconAnchor: [8, 8]
                    })
                }).addTo(map);
                endMarker.on('drag', updateLine);
                endMarker.on('dragend', () => { invalidatePrerender(); generateCrossSection(); });

                line = L.polyline([startMarker.getLatLng(), endMarker.getLatLng()], {
                    color: '#fbbf24', weight: 3, dashArray: '10, 5'
                }).addTo(map);

                generateCrossSection();
            }
        });

        function updateLine() {
            if (startMarker && endMarker && line) {
                line.setLatLngs([startMarker.getLatLng(), endMarker.getLatLng()]);
            }
        }

        // =========================================================================
        // Cross-Section Generation
        // =========================================================================
        async function generateCrossSection() {
            if (!startMarker || !endMarker) return;
            if (activeFhr === null) {
                document.getElementById('xsect-container').innerHTML =
                    '<div id="instructions">Select a forecast hour chip to load data first</div>';
                return;
            }

            const container = document.getElementById('xsect-container');
            container.innerHTML = '<div class="loading-text">Generating cross-section...</div>';

            const start = startMarker.getLatLng();
            const end = endMarker.getLatLng();
            const style = document.getElementById('style-select').value;
            const vscale = document.getElementById('vscale-select').value;
            const ytop = document.getElementById('ytop-select').value;

            const units = document.getElementById('units-select').value;

            const tempCmap = document.getElementById('temp-cmap-select').value;
            const url = `/api/xsect?start_lat=${start.lat}&start_lon=${start.lng}` +
                `&end_lat=${end.lat}&end_lon=${end.lng}&cycle=${currentCycle}&fhr=${activeFhr}&style=${style}` +
                `&y_axis=${currentYAxis}&vscale=${vscale}&y_top=${ytop}&units=${units}&temp_cmap=${tempCmap}` +
                `&anomaly=${anomalyMode ? 1 : 0}${modelParam()}`;

            try {
                const res = await fetch(url);
                if (!res.ok) throw new Error('Failed to generate');
                const blob = await res.blob();
                const img = document.createElement('img');
                img.id = 'xsect-img';
                img.src = URL.createObjectURL(blob);
                container.innerHTML = '';
                container.appendChild(img);
            } catch (err) {
                container.innerHTML = `<div style="color:#f87171">${err.message}</div>`;
            }

            // Update comparison panel if active
            if (compareActive) {
                updateCompareLabels();
                generateComparisonSection();
            }
        }

        // More toggle — show/hide secondary controls
        document.getElementById('more-toggle').onclick = function() {
            const sec = document.getElementById('secondary-controls');
            const vis = sec.classList.toggle('visible');
            this.innerHTML = vis ? 'Less &#9650;' : 'More &#9660;';
        };

        // Clear button
        document.getElementById('clear-btn').onclick = () => {
            if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
            if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
            if (line) { map.removeLayer(line); line = null; }
            document.getElementById('xsect-container').innerHTML =
                '<div id="instructions">Click two points on the map to draw a cross-section line</div>';
            if (compareActive) {
                document.getElementById('xsect-container-compare').innerHTML =
                    '<div style="color:var(--muted);">Draw a line to compare</div>';
            }
        };

        // GIF button
        document.getElementById('gif-btn').onclick = async () => {
            if (!startMarker || !endMarker || !currentCycle) return;
            const btn = document.getElementById('gif-btn');
            btn.disabled = true;
            btn.textContent = 'GIF...';
            const start = startMarker.getLatLng();
            const end = endMarker.getLatLng();
            const style = document.getElementById('style-select').value;
            const vscale = document.getElementById('vscale-select').value;
            const ytop = document.getElementById('ytop-select').value;
            const units = document.getElementById('units-select').value;
            const speed = document.getElementById('gif-speed').value;
            const url = `/api/xsect_gif?start_lat=${start.lat}&start_lon=${start.lng}` +
                `&end_lat=${end.lat}&end_lon=${end.lng}&cycle=${currentCycle}&style=${style}` +
                `&y_axis=${currentYAxis}&vscale=${vscale}&y_top=${ytop}&units=${units}&speed=${speed}` +
                `&temp_cmap=${document.getElementById('temp-cmap-select').value}` +
                `&anomaly=${anomalyMode ? 1 : 0}${modelParam()}` + adminParam();
            try {
                const res = await fetch(url);
                if (!res.ok) {
                    const err = await res.json();
                    alert(err.error || 'GIF generation failed');
                    return;
                }
                const blob = await res.blob();
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = `xsect_${currentCycle}_${style}.gif`;
                a.click();
                URL.revokeObjectURL(a.href);
            } catch (err) {
                alert('GIF generation failed: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'GIF';
            }
        };

        // Swap start/end button
        document.getElementById('swap-btn').onclick = () => {
            if (!startMarker || !endMarker) return;

            // Get current positions
            const startPos = startMarker.getLatLng();
            const endPos = endMarker.getLatLng();

            // Swap positions
            startMarker.setLatLng(endPos);
            endMarker.setLatLng(startPos);

            // Update line
            updateLine();

            // Regenerate cross-section
            generateCrossSection();
        };

        // =========================================================================
        // Auto-refresh for new cycles
        // =========================================================================
        setInterval(async () => {
            const oldCount = cycles.length;
            await loadCycles();
            if (cycles.length > oldCount) {
                showToast('New model run available!', 'success');
            }
        }, 5 * 60 * 1000);  // Every 5 minutes

        // =========================================================================
        // Explainer Modal & Voting
        // =========================================================================
        const styleExplanations = {
            wind_speed: {
                name: 'Wind Speed',
                desc: 'Shows horizontal wind speed in knots. Useful for identifying jet streams, low-level jets, and wind maxima. Wind barbs show true eastward wind component.',
                tech: 'wind_speed = sqrt(u² + v²) × 1.944 kt/m/s'
            },
            temp: {
                name: 'Temperature',
                desc: 'Temperature in Celsius with isotherms. Identifies inversions, frontal zones, and the freezing level. Cyan lines show key isotherms (-10°C, -20°C, -30°C).',
                tech: 'temp_c = T - 273.15'
            },
            theta_e: {
                name: 'Equivalent Potential Temperature (θe)',
                desc: 'Conservative tracer for moist air parcels. Useful for identifying warm/cold advection, atmospheric rivers, and instability. Higher values = warmer, moister air.',
                tech: 'θe = θ × exp(Lv × r / (cp × T))'
            },
            rh: {
                name: 'Relative Humidity',
                desc: 'Percentage saturation of air. Brown = dry air (dry slots, subsidence), Green = moist air. Useful for identifying moisture plumes and dry intrusions.',
                tech: 'RH directly from model output (%)'
            },
            q: {
                name: 'Specific Humidity',
                desc: 'Absolute moisture content in g/kg. Unlike RH, this is not temperature-dependent. Useful for tracking moisture transport and atmospheric rivers.',
                tech: 'q in g/kg, with RH contours at 70%, 80%, 90%'
            },
            omega: {
                name: 'Vertical Velocity (ω)',
                desc: 'Vertical motion in pressure coordinates. Blue = rising air (convection, frontal lift), Red = sinking air (subsidence). Key for precipitation and cloud formation.',
                tech: 'ω in Pa/s, converted to hPa/hr. Negative = rising.'
            },
            vorticity: {
                name: 'Absolute Vorticity',
                desc: 'Spin of the atmosphere. Red = cyclonic (counterclockwise NH), Blue = anticyclonic. Vorticity maxima often associated with troughs and storm development.',
                tech: 'ζ_abs = ζ_rel + f, units: 10⁻⁵ s⁻¹'
            },
            shear: {
                name: 'Wind Shear',
                desc: 'Rate of change of wind with height. High shear indicates jet cores and potential turbulence zones. Important for aviation and severe weather.',
                tech: 'shear = |dV/dz|, units: 10⁻³ s⁻¹'
            },
            lapse_rate: {
                name: 'Temperature Lapse Rate',
                desc: 'Rate of temperature decrease with height. Values near 9.8°C/km (dry adiabatic) indicate instability. Values < 6°C/km indicate stability. Reference lines show dry and moist adiabatic rates.',
                tech: 'Γ = -dT/dz, units: °C/km'
            },
            cloud: {
                name: 'Cloud Water',
                desc: 'Cloud liquid water content. Shows cloud layer locations and thickness. Higher values indicate denser clouds with more precipitation potential.',
                tech: 'Cloud water mixing ratio (g/kg)'
            },
            cloud_total: {
                name: 'Total Condensate',
                desc: 'Sum of all hydrometeors: cloud water, rain, snow, ice, graupel. Gives complete picture of where precipitation and clouds exist in the atmosphere.',
                tech: 'Total = cloud + rain + snow + ice + graupel (g/kg)'
            },
            wetbulb: {
                name: 'Wet-Bulb Temperature',
                desc: 'Temperature if air were cooled to saturation by evaporation. Critical for precipitation type: above 0°C = rain, below 0°C = snow. Lime line shows wet-bulb 0°C.',
                tech: 'Tw computed via iterative psychrometric formula'
            },
            icing: {
                name: 'Icing Potential',
                desc: 'Supercooled liquid water proxy for aircraft icing. Shows where liquid water exists at subfreezing temperatures (0°C to -20°C). Purple = higher icing risk.',
                tech: 'Icing = cloud_water where -20°C < T < 0°C'
            },
            frontogenesis: {
                name: 'Frontogenesis (Winter Bander)',
                desc: 'Petterssen kinematic frontogenesis - the key diagnostic for mesoscale snow bands. Red = frontogenesis (temperature gradient intensifying, banding likely). Blue = frontolysis.',
                tech: 'F = -|∇θ|⁻¹ × (deformation terms), K/100km/3hr, σ=1.5 smoothing'
            }
        };

        let currentVotes = {};

        async function loadVotes() {
            try {
                const res = await fetch('/api/votes');
                currentVotes = await res.json();
            } catch (e) {
                currentVotes = {};
            }
        }

        async function submitVote(style, vote) {
            try {
                const res = await fetch('/api/vote', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({style, vote})
                });
                const data = await res.json();
                currentVotes[style] = data;
                renderExplainerModal();
            } catch (e) {
                showToast('Failed to submit vote', 'error');
            }
        }

        function renderExplainerModal() {
            const body = document.getElementById('modal-body');
            body.innerHTML = Object.entries(styleExplanations).map(([key, info]) => {
                const votes = currentVotes[key] || {up: 0, down: 0};
                const net = votes.up - votes.down;
                const netColor = net > 0 ? '#22c55e' : (net < 0 ? '#ef4444' : 'var(--muted)');
                return `
                    <div class="param-card">
                        <div class="param-header">
                            <span class="param-name">${info.name}</span>
                            <div class="vote-buttons">
                                <button class="vote-btn" onclick="submitVote('${key}', 'up')" title="Good implementation">
                                    👍 <span class="vote-count">${votes.up}</span>
                                </button>
                                <button class="vote-btn" onclick="submitVote('${key}', 'down')" title="Needs improvement">
                                    👎 <span class="vote-count">${votes.down}</span>
                                </button>
                                <span class="vote-count" style="color:${netColor};margin-left:4px">${net >= 0 ? '+' : ''}${net}</span>
                            </div>
                        </div>
                        <div class="param-desc">${info.desc}</div>
                        <div class="param-tech">${info.tech}</div>
                    </div>
                `;
            }).join('');
        }

        document.getElementById('help-btn').onclick = async () => {
            await loadVotes();
            renderExplainerModal();
            document.getElementById('explainer-modal').classList.add('visible');
        };

        document.getElementById('modal-close').onclick = () => {
            document.getElementById('explainer-modal').classList.remove('visible');
        };

        document.getElementById('explainer-modal').onclick = (e) => {
            if (e.target.id === 'explainer-modal') {
                document.getElementById('explainer-modal').classList.remove('visible');
            }
        };

        // =========================================================================
        // Feature Requests
        // =========================================================================
        let requests = [];

        async function loadRequests() {
            try {
                const res = await fetch('/api/requests');
                requests = await res.json();
            } catch (e) {
                requests = [];
            }
        }

        function renderRequests() {
            const list = document.getElementById('request-list');
            if (requests.length === 0) {
                list.innerHTML = '<div style="color:var(--muted);text-align:center;padding:20px;">No requests yet. Be the first!</div>';
                return;
            }
            list.innerHTML = '<h3 style="margin:0 0 12px 0;font-size:14px;color:var(--muted);">Recent Requests</h3>' +
                requests.slice().reverse().slice(0, 20).map(r => `
                    <div class="request-item">
                        <div class="request-item-header">
                            <span>${r.name || 'Anonymous'}</span>
                            <span>${new Date(r.timestamp).toLocaleDateString()}</span>
                        </div>
                        <div class="request-item-text">${escapeHtml(r.text)}</div>
                    </div>
                `).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Feedback UI removed — check data/requests.json directly

        // =========================================================================
        // Initialize
        // =========================================================================
        // =========================================================================
        // RAM Status Modal
        // =========================================================================
        const ramModal = document.getElementById('ram-modal');
        const ramModalBody = document.getElementById('ram-modal-body');

        document.getElementById('memory-status').onclick = async () => {
            try {
                // Fetch status for all registered models
                const modelsRes = await fetch('/api/models');
                const modelsData = await modelsRes.json();
                const modelIds = (modelsData.models || []).map(m => m.id);

                const allResults = await Promise.all(
                    modelIds.map(async m => {
                        const r = await fetch(`/api/status?model=${m}`);
                        const d = await r.json();
                        return { model: m, loaded: d.loaded || [], memory_mb: d.memory_mb || 0 };
                    })
                );

                const totalLoaded = allResults.reduce((s, r) => s + r.loaded.length, 0);
                const totalMb = allResults.reduce((s, r) => s + r.memory_mb, 0);

                if (totalLoaded === 0) {
                    ramModalBody.innerHTML = '<p style="color:var(--muted);text-align:center;padding:20px;">Nothing loaded in RAM</p>';
                } else {
                    let html = '';
                    for (const { model, loaded, memory_mb } of allResults) {
                        if (loaded.length === 0) continue;

                        // Group by cycle
                        const groups = {};
                        loaded.forEach(([cycle, fhr]) => {
                            if (!groups[cycle]) groups[cycle] = [];
                            groups[cycle].push(fhr);
                        });

                        const perFhr = loaded.length > 0 ? memory_mb / loaded.length : 0;
                        const modelMbStr = memory_mb >= 1000 ? (memory_mb/1000).toFixed(1) + ' GB' : Math.round(memory_mb) + ' MB';

                        html += `<div style="margin-bottom:8px;"><strong style="color:var(--accent)">${model.toUpperCase()}</strong> <span style="color:var(--muted);font-size:11px;">${loaded.length} FHRs \u00B7 ${modelMbStr}</span></div>`;
                        html += '<table><tr><th>Cycle</th><th>Forecast Hours</th><th>~RAM</th></tr>';

                        Object.keys(groups).sort().reverse().forEach(cycle => {
                            const fhrs = groups[cycle].sort((a,b) => a - b);
                            const cycleMb = fhrs.length * perFhr;
                            const fhrStr = fhrs.map(f => 'F' + String(f).padStart(2,'0')).join(', ');
                            html += `<tr>
                                <td class="cycle-group">${cycle}</td>
                                <td>${fhrStr}</td>
                                <td>${cycleMb >= 1000 ? (cycleMb/1000).toFixed(1) + ' GB' : Math.round(cycleMb) + ' MB'}</td>
                            </tr>`;
                        });
                        html += '</table>';
                    }

                    const totalStr = totalMb >= 1000 ? (totalMb/1000).toFixed(1) + ' GB' : Math.round(totalMb) + ' MB';
                    html += `<div class="summary">
                        <strong>${totalLoaded}</strong> forecast hours loaded &bull;
                        <strong>${totalStr}</strong> total RAM &bull;
                        <strong>117 GB</strong> cap
                    </div>`;
                    ramModalBody.innerHTML = html;
                }

                ramModal.classList.add('visible');
            } catch (e) {
                showToast('Failed to fetch RAM status', 'error');
            }
        };

        ramModal.onclick = (e) => {
            if (e.target === ramModal) ramModal.classList.remove('visible');
        };
        document.getElementById('ram-modal-close').onclick = () => ramModal.classList.remove('visible');

        loadModels().then(() => loadCycles());
    </script>
</body>
</html>'''

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/api/models')
def api_models():
    """List enabled models."""
    return jsonify({'models': model_registry.list_models()})

@app.route('/api/cycles')
def api_cycles():
    """Return available cycles for the dropdown. Supports ?model=hrrr."""
    mgr = get_manager_from_request() or data_manager
    return jsonify({
        'cycles': mgr.get_cycles_for_ui(),
        'model': mgr.model_name,
    })

@app.route('/api/check_key')
def api_check_key():
    """Check if the provided admin key is valid."""
    mgr = get_manager_from_request() or data_manager
    return jsonify({'valid': check_admin_key(), 'protected': list(mgr.get_protected_cycles())})

@app.route('/api/climatology_status')
def api_climatology_status():
    """Return climatology availability for anomaly mode."""
    if not CLIMATOLOGY_DIR.exists():
        return jsonify({'available': False})
    # Scan for available climo files
    months = {}
    for npz in CLIMATOLOGY_DIR.glob('climo_*.npz'):
        parts = npz.stem.split('_')  # climo_01_00z_F06
        if len(parts) == 4:
            month = int(parts[1])
            init = parts[2]  # "00z"
            if month not in months:
                months[month] = set()
            months[month].add(init)
    # Convert sets to sorted lists
    months = {m: sorted(inits) for m, inits in sorted(months.items())}
    return jsonify({
        'available': len(months) > 0,
        'months': months,
        'anomaly_styles': sorted(ANOMALY_STYLES),
    })

@app.route('/api/status')
def api_status():
    """Return current memory/loading status. Supports ?model=."""
    mgr = get_manager_from_request() or data_manager
    return jsonify(mgr.get_loaded_status())

AUTO_UPDATE_STATUS_FILE = '/tmp/auto_update_status.json'

def _read_auto_update_status():
    """Read auto-update status file written by auto_update.py. Returns dict or None."""
    try:
        import os
        stat = os.stat(AUTO_UPDATE_STATUS_FILE)
        # Skip if stale (>5 min old)
        if time.time() - stat.st_mtime > 300:
            return None
        with open(AUTO_UPDATE_STATUS_FILE, 'r') as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


@app.route('/api/progress')
def api_progress():
    """Return all active progress operations."""
    progress_cleanup()
    now = time.time()
    result = {}
    for op_id, info in PROGRESS.items():
        elapsed = now - info['started']
        step = info['step']
        total = info['total']
        pct = round(100 * step / max(total, 1))

        # Compute rate and ETA from history
        rate = None  # items/sec
        eta = None   # seconds remaining
        hist = info.get('rate_history', [])
        last_step_at = info.get('last_step_at', info['started'])
        stalled = (now - last_step_at) > 10  # no progress for 10s
        if step > 0 and not info['done']:
            if stalled:
                # Use overall rate when stalled (more accurate long-term estimate)
                rate = step / elapsed if elapsed > 0 else None
            elif len(hist) >= 2:
                t0, s0 = hist[0]
                t1, s1 = hist[-1]
                dt = t1 - t0
                ds = s1 - s0
                if dt > 0 and ds > 0:
                    rate = ds / dt
            if rate and rate > 0:
                remaining = total - step
                eta = round(remaining / rate)

        entry = {
            'label': info['label'],
            'op': info['op'],
            'step': step,
            'total': total,
            'detail': info['detail'],
            'pct': pct,
            'elapsed': round(elapsed),
            'done': info['done'],
        }
        if rate is not None:
            entry['rate'] = round(rate, 2)
        if eta is not None:
            entry['eta'] = eta
        result[op_id] = entry

    # Inject auto-update progress from status file (written by auto_update.py)
    au = _read_auto_update_status()
    if au and au.get('models'):
        au_started = au.get('started', au.get('ts', now))
        au_elapsed = round(now - au_started)
        for model, ms in au['models'].items():
            op_id = f"autoupdate:{model}"
            step = ms.get('done', 0)
            total = ms.get('total', 0)
            if total <= 0:
                continue
            # Skip models that are done (no in-flight, all complete)
            if step >= total and not ms.get('in_flight'):
                continue
            pct = round(100 * step / max(total, 1))
            flying = ms.get('in_flight', [])
            last_ok = ms.get('last_ok')
            last_fail = ms.get('last_fail')

            # Build detail string like: "F05 OK — downloading F06, F07, F08"
            parts = []
            if last_ok:
                parts.append(f"{last_ok} OK")
            if last_fail:
                parts.append(f"{last_fail} FAIL")
            detail = ' · '.join(parts) if parts else 'Starting...'
            if flying:
                detail += f" \u2014 downloading {', '.join(flying)}"

            # Compute rate/ETA from elapsed time
            au_rate = None
            au_eta = None
            if step > 0 and au_elapsed > 0:
                au_rate = step / au_elapsed
                remaining = total - step
                if au_rate > 0:
                    au_eta = round(remaining / au_rate)

            entry = {
                'label': f"Auto-update {model.upper()} {ms.get('cycle', '')}",
                'op': 'autoupdate',
                'step': step,
                'total': total,
                'detail': detail,
                'pct': pct,
                'elapsed': au_elapsed,
                'done': False,
            }
            if au_rate is not None:
                entry['rate'] = round(au_rate, 3)
            if au_eta is not None:
                entry['eta'] = au_eta
            result[op_id] = entry

    return jsonify(result)

@app.route('/api/cancel', methods=['POST'])
def api_cancel():
    """Cancel an active progress operation. Requires admin key."""
    if not check_admin_key():
        return jsonify({'error': 'Admin key required'}), 403
    op_id = request.args.get('op_id', '')
    if not op_id:
        return jsonify({'error': 'op_id required'}), 400
    if op_id not in PROGRESS:
        return jsonify({'error': 'Operation not found'}), 404
    if PROGRESS[op_id].get('done'):
        return jsonify({'error': 'Operation already finished'}), 400
    cancel_request(op_id)
    logger.info(f"Cancel requested for {op_id}")
    return jsonify({'ok': True, 'op_id': op_id})

@app.route('/api/load', methods=['POST'])
@rate_limit
def api_load():
    """Load a forecast hour into memory. Archive cycles require admin key."""
    cycle_key = request.args.get('cycle')
    fhr = request.args.get('fhr')

    if not cycle_key or fhr is None:
        return jsonify({'success': False, 'error': 'Missing cycle or fhr parameter'}), 400

    try:
        fhr = int(fhr)
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid fhr'}), 400

    mgr = get_manager_from_request() or data_manager
    result = mgr.load_forecast_hour(cycle_key, fhr)
    return jsonify(result)

@app.route('/api/load_cycle', methods=['POST'])
@rate_limit
def api_load_cycle():
    """Load an entire cycle (all FHRs) into memory. Archive cycles require admin key."""
    cycle_key = request.args.get('cycle')

    if not cycle_key:
        return jsonify({'success': False, 'error': 'Missing cycle parameter'}), 400

    mgr = get_manager_from_request() or data_manager
    result = mgr.load_cycle(cycle_key)
    touch_cycle_access(cycle_key)
    return jsonify(result)

@app.route('/api/unload', methods=['POST'])
@rate_limit
def api_unload():
    """Unload a forecast hour from memory."""
    cycle_key = request.args.get('cycle')
    fhr = request.args.get('fhr')

    if not cycle_key or fhr is None:
        return jsonify({'success': False, 'error': 'Missing cycle or fhr parameter'}), 400

    try:
        fhr = int(fhr)
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid fhr'}), 400

    mgr = get_manager_from_request() or data_manager
    result = mgr.unload_forecast_hour(cycle_key, fhr, is_admin=check_admin_key())
    return jsonify(result)

@app.route('/api/xsect')
@rate_limit
def api_xsect():
    """Generate a cross-section image."""
    try:
        start = (float(request.args['start_lat']), float(request.args['start_lon']))
        end = (float(request.args['end_lat']), float(request.args['end_lon']))
        cycle_key = request.args.get('cycle')
        fhr = int(request.args.get('fhr', 0))
        style = request.args.get('style', 'wind_speed')
        y_axis = request.args.get('y_axis', 'pressure')  # 'pressure' or 'height'
        vscale = float(request.args.get('vscale', 1.0))  # vertical exaggeration
        y_top = int(request.args.get('y_top', 100))  # top of plot in hPa
        dist_units = request.args.get('units', 'km')  # 'km' or 'mi'
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    if not cycle_key:
        return jsonify({'error': 'Missing cycle parameter'}), 400

    # Validate parameters
    if y_axis not in ('pressure', 'height'):
        y_axis = 'pressure'
    vscale = max(0.5, min(3.0, vscale))  # Clamp between 0.5x and 3x
    if y_top not in (100, 200, 300, 500, 700):
        y_top = 100  # Default to full atmosphere

    if dist_units not in ('km', 'mi'):
        dist_units = 'km'
    temp_cmap_param = request.args.get('temp_cmap', 'standard')
    if temp_cmap_param not in ('standard', 'green_purple', 'white_zero', 'nws_ndfd'):
        temp_cmap_param = 'standard'
    anomaly_param = request.args.get('anomaly', '0') == '1'
    acquired = RENDER_SEMAPHORE.acquire(timeout=10)
    if not acquired:
        return jsonify({'error': 'Server busy, try again in a moment'}), 503
    mgr = get_manager_from_request() or data_manager
    try:
        buf = mgr.generate_cross_section(start, end, cycle_key, fhr, style, y_axis, vscale, y_top, units=dist_units, temp_cmap=temp_cmap_param, anomaly=anomaly_param)
    finally:
        RENDER_SEMAPHORE.release()
    if buf is None:
        return jsonify({'error': 'Failed to generate cross-section. Data may not be loaded.'}), 500

    touch_cycle_access(cycle_key)
    return send_file(buf, mimetype='image/png')

@app.route('/api/xsect_gif')
@rate_limit
def api_xsect_gif():
    """Generate an animated GIF of all loaded FHRs for a cycle."""
    try:
        start = (float(request.args['start_lat']), float(request.args['start_lon']))
        end = (float(request.args['end_lat']), float(request.args['end_lon']))
        cycle_key = request.args.get('cycle')
        style = request.args.get('style', 'wind_speed')
        y_axis = request.args.get('y_axis', 'pressure')
        vscale = float(request.args.get('vscale', 1.0))
        y_top = int(request.args.get('y_top', 100))
        dist_units = request.args.get('units', 'km')
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    if not cycle_key:
        return jsonify({'error': 'Missing cycle parameter'}), 400
    if y_axis not in ('pressure', 'height'):
        y_axis = 'pressure'
    vscale = max(0.5, min(3.0, vscale))
    if y_top not in (100, 200, 300, 500, 700):
        y_top = 100
    if dist_units not in ('km', 'mi'):
        dist_units = 'km'
    gif_temp_cmap = request.args.get('temp_cmap', 'standard')
    if gif_temp_cmap not in ('standard', 'green_purple', 'white_zero', 'nws_ndfd'):
        gif_temp_cmap = 'standard'
    gif_anomaly = request.args.get('anomaly', '0') == '1'

    mgr = get_manager_from_request() or data_manager

    # All loaded FHRs available for GIF (mmap makes loading all FHRs cheap)
    loaded_fhrs = sorted(fhr for ck, fhr in mgr.loaded_items
                         if ck == cycle_key)
    if len(loaded_fhrs) < 2:
        return jsonify({'error': f'Need at least 2 loaded FHRs for GIF (have {len(loaded_fhrs)})'}), 400

    # Lock terrain to first FHR so elevation doesn't jitter between frames
    terrain_data = mgr.get_terrain_data(start, end, cycle_key, loaded_fhrs[0], style)

    # GIF holds the semaphore for its entire render sequence (up to 19 frames)
    sem_timeout = 90
    acquired = RENDER_SEMAPHORE.acquire(timeout=sem_timeout)
    if not acquired:
        return jsonify({'error': 'Server busy, try again in a moment'}), 503
    try:
        frames = []
        for fhr in loaded_fhrs:
            buf = mgr.generate_cross_section(start, end, cycle_key, fhr, style, y_axis, vscale, y_top, units=dist_units, terrain_data=terrain_data, temp_cmap=gif_temp_cmap, anomaly=gif_anomaly)
            if buf is not None:
                frames.append(imageio.imread(buf))
    finally:
        RENDER_SEMAPHORE.release()

    if len(frames) < 2:
        return jsonify({'error': 'Failed to generate enough frames'}), 500

    # Speed: 1x = 250ms (fast), 0.75x = 500ms, 0.5x = 1000ms, 0.25x = 2000ms
    SPEED_MS = {'1': 250, '0.75': 500, '0.5': 1000, '0.25': 2000}
    speed_key = request.args.get('speed', '0.5')
    frame_ms = SPEED_MS.get(speed_key, 1000)

    # Use Pillow with disposal=2 (replace each frame) to prevent flickering on Discord
    gif_buf = io.BytesIO()
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_buf, format='GIF', save_all=True,
        append_images=pil_frames[1:],
        duration=frame_ms, loop=0, disposal=2
    )
    gif_buf.seek(0)

    touch_cycle_access(cycle_key)
    return send_file(gif_buf, mimetype='image/gif', download_name=f'xsect_{cycle_key}_{style}.gif')

# =============================================================================
# FRAME PRERENDER + CACHED FRAME API
# =============================================================================

@app.route('/api/prerender', methods=['POST'])
@rate_limit
def api_prerender():
    """Batch prerender frames for slider/comparison. Returns session_id for progress polling.

    POST JSON: {frames: [{cycle, fhr}, ...], start: [lat, lon], end: [lat, lon],
                style, y_axis, vscale, y_top, units, temp_cmap, anomaly, model}
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400

    try:
        frames = data['frames']  # [{cycle: str, fhr: int}, ...]
        start = tuple(data['start'])
        end = tuple(data['end'])
    except (KeyError, TypeError) as e:
        return jsonify({'error': f'Missing required field: {e}'}), 400

    style = data.get('style', 'temperature')
    y_axis = data.get('y_axis', 'pressure')
    vscale = float(data.get('vscale', 1.0))
    y_top = int(data.get('y_top', 100))
    units = data.get('units', 'km')
    temp_cmap = data.get('temp_cmap', 'standard')
    anomaly = bool(data.get('anomaly', False))
    model = data.get('model', 'hrrr')

    session_id = f"prerender:{int(time.time() * 1000)}"

    def _render_batch():
        mgr = model_registry.get(model)
        if not mgr:
            progress_update(session_id, 0, 1, "Unknown model", label="Pre-render failed")
            progress_done(session_id)
            return

        total = len(frames)
        progress_update(session_id, 0, total, "Starting...", label=f"Pre-rendering {total} frames")

        # Lock terrain to first frame for consistency
        first = frames[0]
        try:
            terrain_data = mgr.get_terrain_data(start, end, first['cycle'], first['fhr'], style)
        except Exception:
            terrain_data = None

        # Ensure all data is loaded first (sequential, fast from mmap cache)
        render_frames = []
        rendered = [0]  # mutable for closure
        for frame in frames:
            ck = frame['cycle']
            fhr = int(frame['fhr'])
            cache_key = frame_cache_key(model, ck, fhr, style, start, end, y_axis, vscale, y_top, units, temp_cmap, anomaly)

            if frame_cache_get(cache_key) is not None:
                rendered[0] += 1
                progress_update(session_id, rendered[0], total, f"F{fhr:02d} (cached)")
                continue

            try:
                mgr.ensure_loaded(ck, fhr)
                render_frames.append((ck, fhr, cache_key))
            except Exception:
                rendered[0] += 1
                progress_update(session_id, rendered[0], total, f"F{fhr:02d} load failed")

        if not render_frames:
            progress_done(session_id)
            return

        # Parallel render with ThreadPool — numpy/Agg release GIL for C-heavy work
        def render_one(item):
            ck, fhr, cache_key = item
            if is_cancelled(session_id):
                return fhr, False
            acquired = RENDER_SEMAPHORE.acquire(timeout=30)
            if not acquired:
                return fhr, False
            try:
                buf = mgr.generate_cross_section(
                    start, end, ck, fhr, style, y_axis, vscale, y_top,
                    units=units, terrain_data=terrain_data,
                    temp_cmap=temp_cmap, anomaly=anomaly
                )
                if buf:
                    frame_cache_put(cache_key, buf.getvalue())
                return fhr, True
            except Exception:
                return fhr, False
            finally:
                RENDER_SEMAPHORE.release()

        with ThreadPoolExecutor(max_workers=PRERENDER_WORKERS) as pool:
            futures = {pool.submit(render_one, item): item for item in render_frames}
            for future in as_completed(futures):
                fhr, ok = future.result()
                rendered[0] += 1
                detail = f"F{fhr:02d} {'rendered' if ok else 'failed'}"
                progress_update(session_id, rendered[0], total, detail)
                if is_cancelled(session_id):
                    for f in futures:
                        f.cancel()
                    logger.info(f"Pre-render CANCELLED at {rendered[0]}/{total}")
                    PROGRESS[session_id]['detail'] = 'Cancelled'
                    break

        progress_done(session_id)
        CANCEL_FLAGS.pop(session_id, None)

    threading.Thread(target=_render_batch, daemon=True).start()

    return jsonify({
        'session_id': session_id,
        'total': len(frames),
        'status': 'rendering',
    })


@app.route('/api/frame')
@rate_limit
def api_frame():
    """Get a single cross-section frame. Checks prerender cache first, falls back to live render."""
    try:
        start = (float(request.args['start_lat']), float(request.args['start_lon']))
        end = (float(request.args['end_lat']), float(request.args['end_lon']))
        cycle_key = request.args.get('cycle')
        fhr = int(request.args.get('fhr', 0))
        style = request.args.get('style', 'wind_speed')
        y_axis = request.args.get('y_axis', 'pressure')
        vscale = float(request.args.get('vscale', 1.0))
        y_top = int(request.args.get('y_top', 100))
        dist_units = request.args.get('units', 'km')
        temp_cmap = request.args.get('temp_cmap', 'standard')
        anomaly = request.args.get('anomaly', '0') == '1'
        model = request.args.get('model', 'hrrr')
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    if not cycle_key:
        return jsonify({'error': 'Missing cycle parameter'}), 400

    # Check cache first
    cache_key = frame_cache_key(model, cycle_key, fhr, style, start, end, y_axis, vscale, y_top, dist_units, temp_cmap, anomaly)
    cached = frame_cache_get(cache_key)
    if cached:
        return send_file(io.BytesIO(cached), mimetype='image/png')

    # Fall back to live render (same as /api/xsect)
    vscale = max(0.5, min(3.0, vscale))
    if y_axis not in ('pressure', 'height'):
        y_axis = 'pressure'
    if y_top not in (100, 200, 300, 500, 700):
        y_top = 100
    if dist_units not in ('km', 'mi'):
        dist_units = 'km'
    if temp_cmap not in ('standard', 'green_purple', 'white_zero', 'nws_ndfd'):
        temp_cmap = 'standard'

    acquired = RENDER_SEMAPHORE.acquire(timeout=10)
    if not acquired:
        return jsonify({'error': 'Server busy, try again in a moment'}), 503
    mgr = model_registry.get(model) or data_manager
    try:
        buf = mgr.generate_cross_section(start, end, cycle_key, fhr, style, y_axis, vscale, y_top, units=dist_units, temp_cmap=temp_cmap, anomaly=anomaly)
    finally:
        RENDER_SEMAPHORE.release()
    if buf is None:
        return jsonify({'error': 'Failed to generate frame. Data may not be loaded.'}), 500

    # Cache the result for future requests
    frame_cache_put(cache_key, buf.getvalue())
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


# =============================================================================
# v1 API — agent-friendly endpoints with smart defaults
# =============================================================================

@app.route('/api/v1/cross-section')
@rate_limit
def api_v1_cross_section():
    """Generate a cross-section PNG. Defaults to latest cycle, F00, temperature."""
    try:
        for p in ('start_lat', 'start_lon', 'end_lat', 'end_lon'):
            if p not in request.args:
                raise KeyError(p)
        start = (float(request.args['start_lat']), float(request.args['start_lon']))
        end = (float(request.args['end_lat']), float(request.args['end_lon']))
    except KeyError as e:
        return jsonify({
            'error': f'Missing required parameter: {e.args[0]}',
            'usage': 'Required: start_lat, start_lon, end_lat, end_lon',
            'example': '/api/v1/cross-section?start_lat=39.74&start_lon=-104.99&end_lat=41.88&end_lon=-87.63',
        }), 400
    except ValueError:
        return jsonify({
            'error': 'Coordinates must be numeric (e.g. start_lat=39.74)',
        }), 400

    product = request.args.get('product', 'temperature')
    cycle_raw = request.args.get('cycle', 'latest')
    try:
        fhr = int(request.args.get('fhr', 0))
    except ValueError:
        fhr = 0
    y_axis = request.args.get('y_axis', 'pressure')
    if y_axis not in ('pressure', 'height'):
        y_axis = 'pressure'
    try:
        y_top = int(request.args.get('y_top', 100))
    except ValueError:
        y_top = 100
    if y_top not in (100, 200, 300, 500, 700):
        y_top = 100
    units = request.args.get('units', 'km')
    if units not in ('km', 'mi'):
        units = 'km'

    # Map product name to internal style
    style = PRODUCT_TO_STYLE.get(product)
    if style is None:
        return jsonify({
            'error': f'Unknown product: {product}',
            'available': [p['id'] for p in PRODUCTS_INFO],
        }), 400

    mgr = get_manager_from_request() or data_manager

    # Resolve cycle
    cycle_key = mgr.resolve_cycle(cycle_raw, fhr)
    if not cycle_key:
        return jsonify({'error': f'No data available with forecast hour F{fhr:02d}'}), 404

    # Auto-load if needed (mmap = ~14ms, GRIB = ~30s)
    if not mgr.ensure_loaded(cycle_key, fhr):
        return jsonify({'error': f'Failed to load {cycle_key} F{fhr:02d}'}), 500

    acquired = RENDER_SEMAPHORE.acquire(timeout=90)
    if not acquired:
        return jsonify({'error': 'Server busy rendering other requests, try again in a moment'}), 503
    try:
        buf = mgr.generate_cross_section(
            start, end, cycle_key, fhr, style, y_axis, 1.0, y_top, units=units)
    finally:
        RENDER_SEMAPHORE.release()

    if buf is None:
        return jsonify({'error': 'Render failed'}), 500

    touch_cycle_access(cycle_key)
    return send_file(buf, mimetype='image/png')


@app.route('/api/v1/products')
@rate_limit
def api_v1_products():
    """List available cross-section products. Filters by model (e.g. no smoke for GFS)."""
    model = request.args.get('model', 'hrrr').lower()
    excluded = MODEL_EXCLUDED_STYLES.get(model, set())
    if excluded:
        filtered = [p for p in PRODUCTS_INFO if PRODUCT_TO_STYLE.get(p['id']) not in excluded]
        return jsonify({'products': filtered, 'model': model})
    return jsonify({'products': PRODUCTS_INFO, 'model': model})


@app.route('/api/v1/cycles')
@rate_limit
def api_v1_cycles():
    """List available cycles and their forecast hours."""
    mgr = get_manager_from_request() or data_manager
    cycles_out = []
    for c in mgr.available_cycles:
        ck = c['cycle_key']
        cycles_out.append({
            'key': ck,
            'display': c['display'],
            'forecast_hours': c['available_fhrs'],
            'loaded': any(k == ck for k, _ in mgr.loaded_items),
        })
    latest = mgr.available_cycles[0]['cycle_key'] if mgr.available_cycles else None
    return jsonify({'cycles': cycles_out, 'latest': latest, 'model': mgr.model_name})


@app.route('/api/v1/status')
@rate_limit
def api_v1_status():
    """Server health and status."""
    mgr = get_manager_from_request() or data_manager
    mem_mb = mgr.xsect.get_memory_usage() if mgr.xsect else 0
    latest = mgr.available_cycles[0]['cycle_key'] if mgr.available_cycles else None
    return jsonify({
        'ok': True,
        'model': mgr.model_name,
        'loaded_count': len(mgr.loaded_items),
        'memory_mb': round(mem_mb, 0),
        'latest_cycle': latest,
    })


# Legacy endpoint for compatibility
@app.route('/api/info')
def api_info():
    """Legacy endpoint - returns available times."""
    mgr = get_manager_from_request() or data_manager
    times = mgr.get_available_times()
    return jsonify({
        'times': times,
        'hours': [t['fhr'] for t in times],
        'styles': XSECT_STYLES,
    })

@app.route('/api/votes')
def api_votes():
    """Get current vote counts for all styles."""
    return jsonify(load_votes())

@app.route('/api/vote', methods=['POST'])
def api_vote():
    """Submit a vote for a style."""
    try:
        data = request.get_json()
        style = data.get('style')
        vote = data.get('vote')  # 'up' or 'down'

        if not style or vote not in ('up', 'down'):
            return jsonify({'error': 'Invalid vote data'}), 400

        votes = load_votes()
        if style not in votes:
            votes[style] = {'up': 0, 'down': 0}

        votes[style][vote] += 1
        save_votes(votes)

        return jsonify(votes[style])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/requests')
def api_requests():
    """Get all feature requests."""
    return jsonify(load_requests())

@app.route('/api/request', methods=['POST'])
def api_request():
    """Submit a new feature request."""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()[:100]  # Limit name length
        text = data.get('text', '').strip()[:1000]  # Limit text length

        if not text:
            return jsonify({'error': 'Request text is required'}), 400

        save_request(name, text)
        logger.info(f"New feature request from {name or 'Anonymous'}: {text[:50]}...")

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/request_cycle', methods=['POST'])
@rate_limit
def api_request_cycle():
    """Download specific FHR range for a date/init cycle. Requires admin key."""
    if not check_admin_key():
        return jsonify({'error': 'Admin key required to download archive data'}), 403

    date_str = request.args.get('date', '')  # YYYYMMDD
    hour = int(request.args.get('hour', -1))
    fhr_start = int(request.args.get('fhr_start', 0))
    fhr_end = int(request.args.get('fhr_end', request.args.get('max_fhr', 18)))

    if not date_str:
        return jsonify({'error': 'date required (YYYYMMDD)'}), 400
    if hour < 0 or hour > 23:
        return jsonify({'error': 'hour required (0-23)'}), 400

    try:
        datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format, use YYYYMMDD'}), 400

    mgr = get_manager_from_request() or data_manager
    model_name = mgr.model_name
    cycle_key = f"{date_str}/{hour:02d}z"

    # Determine source label + source preference for archive requests
    from datetime import timezone
    date_dt = datetime.strptime(f"{date_str}{hour:02d}", '%Y%m%d%H').replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - date_dt).total_seconds() / 3600
    source_preference = None
    if model_name == 'hrrr':
        if age_hours > 48:
            source = "AWS archive"
            source_preference = ['aws', 'pando', 'nomads']
        else:
            source = "NOMADS"
    elif model_name == 'rrfs':
        source = "AWS"
        source_preference = ['aws']
    elif model_name == 'gfs':
        source = "NOMADS/NCEP"
        if age_hours > 48:
            source = "NCEP backup"
            source_preference = ['ftpprd', 'nomads']
    else:
        source = "primary sources"

    # Download in background with progress tracking — load each FHR as it arrives
    def download_cycle():
        from smart_hrrr.orchestrator import download_gribs_parallel
        from smart_hrrr.io import create_output_structure

        op_id = f"download:{model_name}:{cycle_key}"
        fhrs = list(range(fhr_start, fhr_end + 1))
        completed = [0]
        downloaded_fhrs = []
        in_flight = set()

        def on_fhr_start(fhr):
            in_flight.add(fhr)
            active = ', '.join(f'F{f:02d}' for f in sorted(in_flight))
            progress_update(op_id, completed[0], len(fhrs),
                            f"Downloading {active} from {source}...")

        def on_fhr_done(fhr, ok):
            in_flight.discard(fhr)
            completed[0] += 1
            if ok:
                downloaded_fhrs.append(fhr)
            active = ', '.join(f'F{f:02d}' for f in sorted(in_flight))
            detail = f"F{fhr:02d} {'OK' if ok else 'FAILED'}"
            if active:
                detail += f" — downloading {active}"
            progress_update(op_id, completed[0], len(fhrs), detail)

        try:
            create_output_structure(model_name, date_str, hour)
            fhr_label = f"F{fhr_start:02d}-F{fhr_end:02d}" if fhr_start != 0 or fhr_end != 18 else f"{len(fhrs)} FHRs"
            progress_update(op_id, 0, len(fhrs), f"Connecting to {source}...",
                            label=f"Downloading {model_name.upper()} {cycle_key} ({fhr_label})")
            results = download_gribs_parallel(
                model=model_name,
                date_str=date_str,
                cycle_hour=hour,
                forecast_hours=fhrs,
                max_threads=4,
                on_complete=on_fhr_done,
                on_start=on_fhr_start,
                should_cancel=lambda: is_cancelled(op_id),
                source_preference=source_preference,
            )
            if is_cancelled(op_id):
                logger.info(f"Download CANCELLED for {model_name} {cycle_key}")
                PROGRESS[op_id]['detail'] = 'Cancelled'
                progress_done(op_id)
                CANCEL_FLAGS.pop(op_id, None)
                return

            success = sum(1 for ok in results.values() if ok)
            logger.info(f"Cycle request {model_name} {cycle_key}: {success}/{len(fhrs)} forecast hours downloaded")

            # Immediately load the downloaded FHRs into memory
            if downloaded_fhrs:
                mgr.scan_available_cycles()
                ck = f"{date_str}_{hour:02d}z"
                ARCHIVE_CACHE_KEYS.add(ck)
                load_id = f"load:{model_name}:{cycle_key}"
                progress_update(load_id, 0, len(downloaded_fhrs), "Loading...",
                                label=f"Loading {model_name.upper()} {cycle_key}")
                for i, fhr in enumerate(sorted(downloaded_fhrs)):
                    try:
                        mgr.ensure_loaded(ck, fhr)
                        progress_update(load_id, i + 1, len(downloaded_fhrs), f"Loaded F{fhr:02d}")
                    except Exception as e:
                        logger.warning(f"Failed to load {ck} F{fhr:02d}: {e}")
                progress_done(load_id)
                logger.info(f"Loaded {len(downloaded_fhrs)} FHRs for {model_name} {cycle_key}")
        except Exception as e:
            logger.warning(f"Cycle request {model_name} {cycle_key} failed: {e}")
        finally:
            progress_done(op_id)

    t = threading.Thread(target=download_cycle, daemon=True)
    t.start()

    n_fhrs = fhr_end - fhr_start + 1
    est_minutes = max(2, n_fhrs * (0.3 if age_hours <= 48 else 0.8))
    return jsonify({
        'success': True,
        'message': f'Downloading {cycle_key} F{fhr_start:02d}-F{fhr_end:02d} from {source} (~{est_minutes:.0f} min)',
        'cycle_key': cycle_key,
        'source': source,
        'est_minutes': round(est_minutes),
        'fhr_start': fhr_start,
        'fhr_end': fhr_end,
    })

@app.route('/api/favorites')
def api_favorites():
    """Get all community favorites."""
    return jsonify(load_favorites())

@app.route('/api/favorite', methods=['POST'])
def api_favorite_save():
    """Save a new community favorite."""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()[:50]  # Limit name length
        label = data.get('label', '').strip()[:200]  # Limit label length
        config = data.get('config', {})

        if not name:
            return jsonify({'error': 'Name is required'}), 400

        fav_id = save_favorite(name, config, label)
        logger.info(f"New favorite saved: {name}")

        return jsonify({'success': True, 'id': fav_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/favorite/<fav_id>', methods=['DELETE'])
def api_favorite_delete(fav_id):
    """Delete a community favorite."""
    try:
        delete_favorite(fav_id)
        logger.info(f"Favorite deleted: {fav_id}")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# MAIN
# =============================================================================

def main():
    global data_manager

    parser = argparse.ArgumentParser(description='Cross-Section Dashboard (multi-model)')
    parser.add_argument('--auto-update', action='store_true', help='Download latest data before starting')
    parser.add_argument('--preload', type=int, default=12, help='Number of latest cycles to pre-load (mmap makes all cheap)')
    parser.add_argument('--max-hours', type=int, default=18, help='Max forecast hour to download')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--production', action='store_true', help='Enable rate limiting')
    parser.add_argument('--models', type=str, default='hrrr',
                        help='Comma-separated list of models to enable (e.g. hrrr,gfs)')
    parser.add_argument('--grib-workers', type=int, default=_env_int('XSECT_GRIB_WORKERS', 4),
                        help='Workers for uncached GRIB->mmap conversion (default: env XSECT_GRIB_WORKERS or 4)')
    parser.add_argument('--preload-workers', type=int, default=_env_int('XSECT_PRELOAD_WORKERS', 20),
                        help='Workers for cached mmap loads (default: env XSECT_PRELOAD_WORKERS or 20)')

    args = parser.parse_args()
    app.config['PRODUCTION'] = args.production

    # Register requested models
    enabled_models = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    for model_name in enabled_models:
        if model_name not in model_registry.managers:
            model_registry.register(model_name)

    # Apply runtime worker tuning to all managers
    grib_workers = max(1, args.grib_workers)
    preload_workers = max(1, args.preload_workers)
    for _, mgr in model_registry.managers.items():
        mgr.GRIB_WORKERS = grib_workers
        mgr.PRELOAD_WORKERS = preload_workers

    data_manager = model_registry.get('hrrr')  # Keep backward compat alias

    # Optionally download fresh data (HRRR only for now)
    if args.auto_update:
        from smart_hrrr.orchestrator import download_latest_cycle

        logger.info("Downloading latest HRRR data...")
        fhrs_to_download = [0, 6, 12, 18]
        fhrs_to_download = [f for f in fhrs_to_download if f <= args.max_hours]

        date_str, hour, results = download_latest_cycle(
            max_hours=max(fhrs_to_download),
            forecast_hours=fhrs_to_download
        )
        if not date_str:
            logger.error("Failed to download data")
            sys.exit(1)
        logger.info(f"Downloaded {sum(results.values())}/{len(fhrs_to_download)} forecast hours")

    # Scan for available cycles across all models
    for model_name, mgr in model_registry.managers.items():
        logger.info(f"Scanning {model_name.upper()} for available cycles...")
        cycles = mgr.scan_available_cycles()
        if cycles:
            logger.info(f"  {model_name.upper()}: {len(cycles)} cycles found")
            for c in cycles[:3]:  # Show first 3
                fhrs_str = ', '.join(f'F{f:02d}' for f in c['available_fhrs'])
                logger.info(f"    {c['display']}: [{fhrs_str}]")
            if len(cycles) > 3:
                logger.info(f"    ... and {len(cycles) - 3} more")
        else:
            logger.info(f"  {model_name.upper()}: No data found")

    # Pre-load latest cycles in background so Flask starts immediately
    if args.preload > 0:
        def _startup_preload():
            time.sleep(2)  # Let Flask bind first
            # HRRR always loads first — it's the primary product
            ordered = sorted(model_registry.managers.items(),
                             key=lambda x: (0 if x[0] == 'hrrr' else 1, x[0]))
            for model_name, mgr in ordered:
                if mgr.available_cycles:
                    logger.info(f"Background: Pre-loading latest {args.preload} {model_name.upper()} cycles...")
                    try:
                        mgr.preload_latest_cycles(n_cycles=args.preload)
                    except Exception as e:
                        logger.warning(f"{model_name.upper()} preload failed: {e}")
        threading.Thread(target=_startup_preload, daemon=True).start()

    # Background re-scan thread: HRRR scans every 30s, others every 60s
    def background_rescan():
        tick = 0
        while True:
            time.sleep(30)
            tick += 1
            # HRRR every tick (30s), others every other tick (60s)
            for model_name, mgr in model_registry.managers.items():
                if model_name != 'hrrr' and tick % 2 != 0:
                    continue
                try:
                    mgr.scan_available_cycles()
                    mgr.auto_load_latest()
                except Exception as e:
                    logger.warning(f"Background rescan failed for {model_name}: {e}")

            # Evict old NVMe cache + check GRIB disk usage every 10 minutes
            if int(time.time()) % 600 < 60:
                try:
                    cache_evict_old_cycles(model_registry.managers)
                except Exception as e:
                    logger.warning(f"Cache eviction failed: {e}")
                try:
                    usage = get_disk_usage_gb()
                    if usage > DISK_LIMIT_GB:
                        logger.info(f"Disk usage {usage:.1f}GB > {DISK_LIMIT_GB}GB limit, evicting...")
                        disk_evict_least_popular()
                        for _, mgr in model_registry.managers.items():
                            mgr.scan_available_cycles()
                except Exception as e:
                    logger.warning(f"Disk eviction check failed: {e}")

    rescan_thread = threading.Thread(target=background_rescan, daemon=True)
    rescan_thread.start()

    disk_gb = get_disk_usage_gb()
    logger.info("")
    logger.info("=" * 60)
    logger.info("Cross-Section Dashboard")
    logger.info(f"Models: {', '.join(m.upper() for m in enabled_models)}")
    logger.info(f"GRIB backend default: {os.environ.get('XSECT_GRIB_BACKEND', 'auto')}")
    logger.info(f"Workers: grib={grib_workers}, preload={preload_workers}")
    for model_name, mgr in model_registry.managers.items():
        mem_mb = mgr.xsect.get_memory_usage() if mgr.xsect else 0
        logger.info(f"  {model_name.upper()}: {len(mgr.loaded_cycles)} cycles loaded ({mem_mb:.0f} MB)")
    logger.info(f"Disk: {disk_gb:.1f}GB / {DISK_LIMIT_GB}GB")
    logger.info(f"Auto-refreshing cycle list every 60s")
    logger.info(f"Open: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()
