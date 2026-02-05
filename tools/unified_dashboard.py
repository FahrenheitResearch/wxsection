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
import sys
import time
import io
import threading
from pathlib import Path
from datetime import datetime
from functools import wraps
from collections import defaultdict

import imageio.v2 as imageio

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
DISK_LIMIT_GB = 500  # Max disk usage for HRRR data

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

PROGRESS = {}  # Global progress dict: op_id -> {op, label, step, total, detail, started, done, done_at}

def progress_update(op_id, step, total, detail, label=None):
    """Update progress for an operation."""
    if op_id not in PROGRESS:
        PROGRESS[op_id] = {
            'op': op_id.split(':')[0],
            'label': label or op_id,
            'step': step,
            'total': total,
            'detail': detail,
            'started': time.time(),
            'done': False,
            'done_at': None,
        }
    else:
        PROGRESS[op_id]['step'] = step
        PROGRESS[op_id]['total'] = total
        PROGRESS[op_id]['detail'] = detail
        if label:
            PROGRESS[op_id]['label'] = label

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
    """Remove entries that finished more than 5s ago."""
    now = time.time()
    to_remove = [k for k, v in PROGRESS.items() if v.get('done') and v.get('done_at') and now - v['done_at'] > 5]
    for k in to_remove:
        del PROGRESS[k]

# =============================================================================
# DATA MANAGER - On-Demand Loading for Memory Efficiency
# =============================================================================

class CrossSectionManager:
    """Manages cross-section data with smart pre-loading.

    Pre-loads latest N cycles at startup for instant access.
    Older cycles are available on-demand with loading indicator.
    """

    FORECAST_HOURS = list(range(19))  # F00-F18
    PRELOAD_CYCLES = 0  # Don't pre-load; load on demand
    MEM_LIMIT_MB = 117000  # 117 GB hard cap
    MEM_EVICT_MB = 115000  # Start evicting at 115 GB

    def __init__(self):
        self.xsect = None
        self.base_dir = Path("outputs/hrrr")
        self.available_cycles = []  # List of available cycles (metadata only)
        self.loaded_cycles = set()  # Cycle keys that are fully loaded
        self.loaded_items = []  # List of (cycle_key, fhr) currently in memory (ordered by load time = LRU)
        self.current_cycle = None  # Currently selected cycle
        self._lock = threading.Lock()  # Protects all state mutations
        self._engine_key_map = {}  # (cycle_key, fhr) -> unique engine int key
        self._next_engine_key = 0  # Counter for unique keys

    def _get_engine_key(self, cycle_key: str, fhr: int) -> int:
        """Get or create a unique engine key for a (cycle_key, fhr) pair."""
        pair = (cycle_key, fhr)
        if pair not in self._engine_key_map:
            self._engine_key_map[pair] = self._next_engine_key
            self._next_engine_key += 1
        return self._engine_key_map[pair]

    def _evict_if_needed(self):
        """Evict oldest loaded items if memory exceeds threshold."""
        if not self.xsect:
            return
        mem_mb = self.xsect.get_memory_usage()
        while mem_mb > self.MEM_EVICT_MB and self.loaded_items:
            # Evict oldest (first in list)
            old_key, old_fhr = self.loaded_items[0]
            logger.info(f"Memory {mem_mb:.0f}MB > {self.MEM_EVICT_MB}MB, evicting {old_key} F{old_fhr:02d}")
            self._unload_item(old_key, old_fhr)
            self.loaded_items.pop(0)
            # If all FHRs of a cycle are evicted, unmark it
            if not any(k == old_key for k, _ in self.loaded_items):
                self.loaded_cycles.discard(old_key)
            mem_mb = self.xsect.get_memory_usage()

    def init_engine(self):
        """Initialize the cross-section engine if needed."""
        if self.xsect is None:
            from core.cross_section_interactive import InteractiveCrossSection
            self.xsect = InteractiveCrossSection(cache_dir='cache/dashboard/xsect')

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
                for fhr in self.FORECAST_HOURS:
                    fhr_dir = hour_dir / f"F{fhr:02d}"
                    if fhr_dir.exists() and list(fhr_dir.glob("*wrfprs*.grib2")):
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
                        'display': f"HRRR - {init_dt.strftime('%b %d %HZ')}",
                    })

        return self.available_cycles

    def get_cycles_for_ui(self):
        """Return cycles formatted for UI dropdown, grouped by date."""
        return [
            {
                'key': c['cycle_key'],
                'display': c['display'],
                'date': c['date'],
                'hour': c['hour'],
                'fhrs': c['available_fhrs'],
                'fhr_count': len(c['available_fhrs']),
                'loaded': c['cycle_key'] in self.loaded_cycles,
            }
            for c in self.available_cycles
        ]

    def preload_latest_cycles(self, n_cycles: int = None):
        """Pre-load the latest N cycles that have all 4 forecast hours."""
        if n_cycles is None:
            n_cycles = self.PRELOAD_CYCLES

        self.init_engine()

        # Prefer cycles with all 4 FHRs (F00, F06, F12, F18)
        complete_cycles = [c for c in self.available_cycles
                          if all(fhr in c['available_fhrs'] for fhr in self.FORECAST_HOURS)]

        if len(complete_cycles) >= n_cycles:
            cycles_to_load = complete_cycles[:n_cycles]
            logger.info(f"Found {len(complete_cycles)} complete cycles (all 4 FHRs)")
        else:
            # Fall back to newest cycles if not enough complete ones
            cycles_to_load = self.available_cycles[:n_cycles]
            logger.info(f"Only {len(complete_cycles)} complete cycles, using newest {n_cycles}")

        for cycle in cycles_to_load:
            cycle_key = cycle['cycle_key']
            logger.info(f"Pre-loading {cycle['display']} (even FHRs only)...")

            for fhr in cycle['available_fhrs']:
                if fhr % 2 != 0:
                    continue  # Only preload even forecast hours into RAM
                with self._lock:
                    if (cycle_key, fhr) in self.loaded_items:
                        continue
                    self.xsect.init_date = cycle['date']
                    self.xsect.init_hour = cycle['hour']
                    engine_key = self._get_engine_key(cycle_key, fhr)

                run_path = Path(cycle['path'])
                fhr_dir = run_path / f"F{fhr:02d}"
                prs_files = list(fhr_dir.glob("*wrfprs*.grib2"))

                if prs_files:
                    if self.xsect.load_forecast_hour(str(prs_files[0]), engine_key):
                        with self._lock:
                            self.loaded_items.append((cycle_key, fhr))

            with self._lock:
                self.loaded_cycles.add(cycle_key)
            mem_mb = self.xsect.get_memory_usage()
            logger.info(f"  {cycle['display']} loaded ({mem_mb:.0f} MB total)")

    def get_loaded_status(self):
        """Return current memory status."""
        mem_mb = self.xsect.get_memory_usage() if self.xsect else 0
        return {
            'loaded': self.loaded_items.copy(),
            'loaded_cycles': list(self.loaded_cycles),
            'memory_mb': round(mem_mb, 0),
            'loading': self._lock.locked(),
        }

    def load_cycle(self, cycle_key: str) -> dict:
        """Load an entire cycle (all available FHRs) into memory."""
        with self._lock:
            if cycle_key in self.loaded_cycles:
                return {'success': True, 'already_loaded': True}

            cycle = next((c for c in self.available_cycles if c['cycle_key'] == cycle_key), None)
            if not cycle:
                return {'success': False, 'error': f'Cycle {cycle_key} not found'}

            self.init_engine()

        op_id = f"cycle:{cycle_key}"
        total_fhrs = len(cycle['available_fhrs'])
        progress_update(op_id, 0, total_fhrs, "Starting...", label=f"Loading cycle {cycle_key}")
        loaded_count = 0

        for idx, fhr in enumerate(cycle['available_fhrs']):
            with self._lock:
                if (cycle_key, fhr) in self.loaded_items:
                    loaded_count += 1
                    progress_update(op_id, loaded_count, total_fhrs, f"F{fhr:02d} (cached)")
                    continue

            run_path = Path(cycle['path'])
            fhr_dir = run_path / f"F{fhr:02d}"
            prs_files = list(fhr_dir.glob("*wrfprs*.grib2"))

            if prs_files:
                with self._lock:
                    self.xsect.init_date = cycle['date']
                    self.xsect.init_hour = cycle['hour']
                    self._evict_if_needed()
                    engine_key = self._get_engine_key(cycle_key, fhr)

                progress_update(op_id, loaded_count, total_fhrs, f"Loading F{fhr:02d}...")
                logger.info(f"Loading {cycle_key} F{fhr:02d} (engine key {engine_key})...")

                # Slow I/O without lock
                if self.xsect.load_forecast_hour(str(prs_files[0]), engine_key):
                    with self._lock:
                        self.loaded_items.append((cycle_key, fhr))
                    loaded_count += 1

        with self._lock:
            self.loaded_cycles.add(cycle_key)
        mem_mb = self.xsect.get_memory_usage()
        logger.info(f"Loaded {cycle['display']} ({loaded_count} FHRs, {mem_mb:.0f} MB total)")
        progress_done(op_id)

        return {
            'success': True,
            'cycle': cycle_key,
            'loaded_fhrs': loaded_count,
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
            prs_files = list(fhr_dir.glob("*wrfprs*.grib2"))
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

    def unload_forecast_hour(self, cycle_key: str, fhr: int) -> dict:
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

    def generate_cross_section(self, start, end, cycle_key, fhr, style, y_axis='pressure', vscale=1.0, y_top=100, units='km', terrain_data=None, temp_cmap='green_purple'):
        """Generate a cross-section for a loaded forecast hour."""
        if not self.xsect:
            return None

        if (cycle_key, fhr) not in self.loaded_items:
            return None

        # Set correct metadata for this cycle
        cycle = next((c for c in self.available_cycles if c['cycle_key'] == cycle_key), None)
        if cycle:
            self.xsect.init_date = cycle['date']
            self.xsect.init_hour = cycle['hour']

        engine_key = self._engine_key_map.get((cycle_key, fhr))
        if engine_key is None:
            return None

        try:
            # Pass real forecast hour for correct title labeling
            self.xsect._real_forecast_hour = fhr
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
                temp_cmap=temp_cmap
            )
            if png_bytes is None:
                return None
            return io.BytesIO(png_bytes)
        except Exception as e:
            logger.error(f"Cross-section error: {e}")
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


data_manager = CrossSectionManager()

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
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            border-bottom: 1px solid var(--border);
        }
        .control-row {
            display: flex;
            gap: 16px;
            align-items: center;
            flex-wrap: wrap;
        }
        .control-group { display: flex; align-items: center; gap: 8px; }
        label { color: var(--muted); font-size: 13px; font-weight: 500; }
        select {
            background: var(--card);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            min-width: 180px;
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
            min-width: 320px;
            max-width: 400px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            display: none;
            overflow: hidden;
        }
        #progress-panel.visible { display: block; animation: slideUp 0.3s ease; }
        .progress-header {
            padding: 8px 14px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--muted);
            border-bottom: 1px solid var(--border);
            background: rgba(255,255,255,0.03);
        }
        .progress-items { padding: 6px 0; max-height: 300px; overflow-y: auto; }
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
        .progress-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 200px;
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
        .progress-detail {
            font-size: 11px;
            color: var(--muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
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
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px;
            overflow: hidden;
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
                    <label>Model Run:</label>
                    <select id="cycle-select"></select>
                    <button id="request-cycle-btn" title="Request a specific init cycle" style="padding:4px 8px;font-size:13px;">📅</button>
                </div>
                <div class="control-group">
                    <label>Style:</label>
                    <select id="style-select"></select>
                    <select id="temp-cmap-select" style="display:none" title="Temperature color table">
                        <option value="green_purple">Green-Purple</option>
                        <option value="white_zero">White at 0°C</option>
                        <option value="nws_ndfd">NWS Classic</option>
                    </select>
                    <button class="help-btn" id="help-btn" title="Style explanations & feedback">?</button>
                    <button class="request-btn" id="request-btn" title="Submit feature request">💡</button>
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
                    <select id="vscale-select">
                        <option value="0.5">0.5x</option>
                        <option value="1.0" selected>1x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2.0">2x</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Top:</label>
                    <select id="ytop-select">
                        <option value="100" selected>100 hPa</option>
                        <option value="200">200 hPa</option>
                        <option value="300">300 hPa</option>
                        <option value="500">500 hPa</option>
                        <option value="700">700 hPa</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Units:</label>
                    <select id="units-select">
                        <option value="km" selected>km</option>
                        <option value="mi">mi</option>
                    </select>
                </div>
                <div class="control-group favorites-group">
                    <select id="favorites-select">
                        <option value="">⭐ Favorites</option>
                    </select>
                    <button id="save-favorite-btn" title="Save current view">💾</button>
                </div>
                <button id="swap-btn" title="Swap start/end points">⇄ Swap</button>
                <button id="gif-btn" title="Generate animated GIF of all loaded FHRs">GIF</button>
                <select id="gif-speed" title="GIF speed">
                    <option value="normal">Normal</option>
                    <option value="slow">Slow</option>
                </select>
                <button id="clear-btn">Clear Line</button>
                <div id="memory-status">
                    <span id="mem-text">0 MB</span>
                    <div class="mem-bar"><div class="mem-fill" id="mem-fill" style="width:0%"></div></div>
                </div>
            </div>
            <div class="control-row">
                <label>Forecast Hours:</label>
                <div class="chip-group" id="fhr-chips"></div>
            </div>
        </div>
        <div id="map"></div>
    </div>
    <div id="sidebar" style="position: relative;">
        <div id="xsect-header">
            <span>Cross-Section</span>
            <span id="active-fhr"></span>
        </div>
        <div id="xsect-container">
            <div id="instructions">
                Click two points on the map to draw a cross-section line.<br>
                Then select forecast hours to load.
            </div>
        </div>
    </div>
    <div id="toast-container"></div>
    <div id="progress-panel">
        <div class="progress-header">Activity</div>
        <div class="progress-items" id="progress-items"></div>
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

        // Initialize map
        const map = L.map('map', {
            center: [39, -98],
            zoom: 5,
            minZoom: 4,
            maxZoom: 10
        });
        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap, &copy; CARTO'
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
        styleSelect.onchange = () => { updateTempCmapVisibility(); generateCrossSection(); };
        tempCmapSelect.onchange = generateCrossSection;

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
                    startMarker.on('dragend', generateCrossSection);

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
                    endMarker.on('dragend', generateCrossSection);

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
        // Request Custom Date/Cycle
        // =========================================================================
        document.getElementById('request-cycle-btn').onclick = async function() {
            const dateStr = prompt('Request a specific init cycle (F00-F18)\\nEnter date (YYYYMMDD):', new Date().toISOString().slice(0,10).replace(/-/g,''));
            if (!dateStr || dateStr.length !== 8) return;

            const hourStr = prompt('Enter init hour (0-23):', '12');
            if (hourStr === null) return;
            const hour = parseInt(hourStr);
            if (isNaN(hour) || hour < 0 || hour > 23) {
                showToast('Invalid hour (0-23)', 'error');
                return;
            }

            const toast = showToast(`Requesting ${dateStr}/${String(hour).padStart(2,'0')}z F00-F18...`);
            try {
                const res = await fetch(`/api/request_cycle?date=${dateStr}&hour=${hour}`, {method: 'POST'});
                const data = await res.json();
                toast.remove();
                if (data.success) {
                    // Show persistent progress toast
                    const progressToast = showToast(`📡 ${data.cycle_key}: downloading 0/19 FHRs from ${data.source}...`);
                    let lastCount = 0;

                    const pollInterval = setInterval(async () => {
                        await refreshCycleList();
                        const found = cycles.find(c => c.key === data.cycle_key);
                        const count = found ? found.fhrs.length : 0;

                        if (count !== lastCount) {
                            lastCount = count;
                            const pct = Math.round(count / 19 * 100);
                            const bar = '█'.repeat(Math.round(pct / 5)) + '░'.repeat(20 - Math.round(pct / 5));
                            progressToast.querySelector('span').textContent =
                                `📡 ${data.cycle_key}: ${count}/19 FHRs [${bar}] ${pct}%`;
                        }

                        if (count >= 19) {
                            clearInterval(pollInterval);
                            progressToast.remove();
                            showToast(`${data.cycle_key} ready! All 19 forecast hours downloaded.`, 'success');
                        }
                    }, 10000);

                    // Stop polling after est + 10 min
                    setTimeout(() => {
                        clearInterval(pollInterval);
                        if (lastCount < 19) {
                            progressToast.remove();
                            showToast(`${data.cycle_key}: ${lastCount}/19 FHRs downloaded (timed out polling, download may still be running)`, 'error');
                        }
                    }, ((data.est_minutes || 10) + 10) * 60000);
                } else {
                    showToast(data.error || 'Request failed', 'error');
                }
            } catch (e) {
                toast.remove();
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
                const res = await fetch('/api/cycles');
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
                const res = await fetch('/api/cycles');
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
            } catch (e) {
                // Silent fail for background refresh
            }
        }
        setInterval(() => { refreshCycleList(); refreshLoadedStatus(); }, 60000);

        // === Progress Panel ===
        async function pollProgress() {
            try {
                const res = await fetch('/api/progress');
                const data = await res.json();
                const panel = document.getElementById('progress-panel');
                const container = document.getElementById('progress-items');
                const entries = Object.entries(data);

                if (entries.length === 0) {
                    panel.classList.remove('visible');
                    return;
                }

                panel.classList.add('visible');
                container.innerHTML = '';

                for (const [opId, info] of entries) {
                    const item = document.createElement('div');
                    item.className = 'progress-item' + (info.done ? ' done' : '');

                    const elapsed = info.elapsed;
                    const min = Math.floor(elapsed / 60);
                    const sec = elapsed % 60;
                    const timeStr = min > 0 ? `${min}m ${sec}s` : `${sec}s`;

                    item.innerHTML = `
                        <div class="progress-item-header">
                            <span class="progress-label">${info.label}</span>
                            <span class="progress-stats">${info.step}/${info.total} (${info.pct}%) · ${timeStr}</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width:${info.pct}%"></div>
                        </div>
                        <div class="progress-detail">${info.detail}</div>
                    `;
                    container.appendChild(item);
                }

                // Also update memory display from any active load
                refreshLoadedStatus();
            } catch (e) {
                // Silent fail
            }
        }
        setInterval(pollProgress, 2000);
        pollProgress();

        cycleSelect.onchange = async () => {
            const selected = cycleSelect.options[cycleSelect.selectedIndex];
            currentCycle = selected.value;
            const fhrs = JSON.parse(selected.dataset.fhrs || '[]');
            const isLoaded = selected.dataset.loaded === 'true';

            if (!isLoaded) {
                // Need to load this cycle first
                const toast = showToast(`Loading cycle (this may take a minute)...`);
                try {
                    const res = await fetch(`/api/load_cycle?cycle=${currentCycle}`, {method: 'POST'});
                    const data = await res.json();
                    toast.remove();

                    if (data.success) {
                        showToast(`Loaded ${data.loaded_fhrs} forecast hours`, 'success');
                        selected.textContent = selected.textContent.replace(' ⏳', '');
                        selected.dataset.loaded = 'true';
                        updateMemoryDisplay(data.memory_mb || 0);

                        // Refresh cycles list to update loaded status
                        const cyclesRes = await fetch('/api/cycles');
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

            const allFhrs = Array.from({length: 19}, (_, i) => i);

            allFhrs.forEach(fhr => {
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

                container.appendChild(chip);
            });
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
                    const res = await fetch(`/api/unload?cycle=${currentCycle}&fhr=${fhr}`, {method: 'POST'});
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
                const res = await fetch(`/api/load?cycle=${currentCycle}&fhr=${fhr}`, {method: 'POST'});
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
        }

        async function refreshLoadedStatus() {
            try {
                const res = await fetch('/api/status');
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
                startMarker.on('dragend', generateCrossSection);
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
                endMarker.on('dragend', generateCrossSection);

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
                `&y_axis=${currentYAxis}&vscale=${vscale}&y_top=${ytop}&units=${units}&temp_cmap=${tempCmap}`;

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
        }

        // Clear button
        document.getElementById('clear-btn').onclick = () => {
            if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
            if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
            if (line) { map.removeLayer(line); line = null; }
            document.getElementById('xsect-container').innerHTML =
                '<div id="instructions">Click two points on the map to draw a cross-section line</div>';
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
                `&temp_cmap=${document.getElementById('temp-cmap-select').value}`;
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

        document.getElementById('request-btn').onclick = async () => {
            await loadRequests();
            renderRequests();
            document.getElementById('request-modal').classList.add('visible');
        };

        document.getElementById('request-modal-close').onclick = () => {
            document.getElementById('request-modal').classList.remove('visible');
        };

        document.getElementById('request-modal').onclick = (e) => {
            if (e.target.id === 'request-modal') {
                document.getElementById('request-modal').classList.remove('visible');
            }
        };

        document.getElementById('request-form').onsubmit = async (e) => {
            e.preventDefault();
            const name = document.getElementById('request-name').value.trim();
            const text = document.getElementById('request-text').value.trim();

            if (!text) return;

            const btn = e.target.querySelector('.submit-btn');
            btn.disabled = true;
            btn.textContent = 'Submitting...';

            try {
                const res = await fetch('/api/request', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name, text})
                });
                if (res.ok) {
                    showToast('Request submitted! Thank you!', 'success');
                    document.getElementById('request-text').value = '';
                    await loadRequests();
                    renderRequests();
                } else {
                    showToast('Failed to submit request', 'error');
                }
            } catch (err) {
                showToast('Failed to submit request', 'error');
            }

            btn.disabled = false;
            btn.textContent = 'Submit Request';
        };

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
                const res = await fetch('/api/status');
                const data = await res.json();
                const loaded = data.loaded || [];
                const memMb = data.memory_mb || 0;

                if (loaded.length === 0) {
                    ramModalBody.innerHTML = '<p style="color:var(--muted);text-align:center;padding:20px;">Nothing loaded in RAM</p>';
                } else {
                    // Group by cycle
                    const groups = {};
                    loaded.forEach(([cycle, fhr]) => {
                        if (!groups[cycle]) groups[cycle] = [];
                        groups[cycle].push(fhr);
                    });

                    let html = '<table><tr><th>Cycle</th><th>Forecast Hours</th><th>~RAM</th></tr>';
                    const perFhr = loaded.length > 0 ? memMb / loaded.length : 0;

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
                    html += `<div class="summary">
                        <strong>${loaded.length}</strong> forecast hours loaded &bull;
                        <strong>${memMb >= 1000 ? (memMb/1000).toFixed(1) + ' GB' : Math.round(memMb) + ' MB'}</strong> total RAM &bull;
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

        loadCycles();
    </script>
</body>
</html>'''

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/api/cycles')
def api_cycles():
    """Return available cycles for the dropdown."""
    return jsonify({
        'cycles': data_manager.get_cycles_for_ui(),
    })

@app.route('/api/status')
def api_status():
    """Return current memory/loading status."""
    return jsonify(data_manager.get_loaded_status())

@app.route('/api/progress')
def api_progress():
    """Return all active progress operations."""
    progress_cleanup()
    result = {}
    for op_id, info in PROGRESS.items():
        elapsed = time.time() - info['started']
        result[op_id] = {
            'label': info['label'],
            'step': info['step'],
            'total': info['total'],
            'detail': info['detail'],
            'pct': round(100 * info['step'] / max(info['total'], 1)),
            'elapsed': round(elapsed),
            'done': info['done'],
        }
    return jsonify(result)

@app.route('/api/load', methods=['POST'])
@rate_limit
def api_load():
    """Load a forecast hour into memory."""
    cycle_key = request.args.get('cycle')
    fhr = request.args.get('fhr')

    if not cycle_key or fhr is None:
        return jsonify({'success': False, 'error': 'Missing cycle or fhr parameter'}), 400

    try:
        fhr = int(fhr)
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid fhr'}), 400

    result = data_manager.load_forecast_hour(cycle_key, fhr)
    return jsonify(result)

@app.route('/api/load_cycle', methods=['POST'])
@rate_limit
def api_load_cycle():
    """Load an entire cycle (all FHRs) into memory."""
    cycle_key = request.args.get('cycle')

    if not cycle_key:
        return jsonify({'success': False, 'error': 'Missing cycle parameter'}), 400

    result = data_manager.load_cycle(cycle_key)
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

    result = data_manager.unload_forecast_hour(cycle_key, fhr)
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
    temp_cmap_param = request.args.get('temp_cmap', 'green_purple')
    if temp_cmap_param not in ('green_purple', 'white_zero', 'nws_ndfd'):
        temp_cmap_param = 'green_purple'
    buf = data_manager.generate_cross_section(start, end, cycle_key, fhr, style, y_axis, vscale, y_top, units=dist_units, temp_cmap=temp_cmap_param)
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
    gif_temp_cmap = request.args.get('temp_cmap', 'green_purple')
    if gif_temp_cmap not in ('green_purple', 'white_zero', 'nws_ndfd'):
        gif_temp_cmap = 'green_purple'

    # Get loaded FHRs for this cycle, sorted
    loaded_fhrs = sorted(fhr for ck, fhr in data_manager.loaded_items if ck == cycle_key)
    if len(loaded_fhrs) < 2:
        return jsonify({'error': f'Need at least 2 loaded FHRs for GIF (have {len(loaded_fhrs)})'}), 400

    # Lock terrain to first FHR so elevation doesn't jitter between frames
    terrain_data = data_manager.get_terrain_data(start, end, cycle_key, loaded_fhrs[0], style)

    frames = []
    for fhr in loaded_fhrs:
        buf = data_manager.generate_cross_section(start, end, cycle_key, fhr, style, y_axis, vscale, y_top, units=dist_units, terrain_data=terrain_data, temp_cmap=gif_temp_cmap)
        if buf is not None:
            frames.append(imageio.imread(buf))

    if len(frames) < 2:
        return jsonify({'error': 'Failed to generate enough frames'}), 500

    speed = request.args.get('speed', 'normal')
    frame_duration = 1.5 if speed == 'slow' else 0.8

    gif_buf = io.BytesIO()
    imageio.mimwrite(gif_buf, frames, format='GIF', duration=frame_duration, loop=0)
    gif_buf.seek(0)

    touch_cycle_access(cycle_key)
    return send_file(gif_buf, mimetype='image/gif', download_name=f'xsect_{cycle_key}_{style}.gif')

# Legacy endpoint for compatibility
@app.route('/api/info')
def api_info():
    """Legacy endpoint - returns available times."""
    times = data_manager.get_available_times()
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
    """Download F00-F18 for a specific date/init cycle."""
    date_str = request.args.get('date', '')  # YYYYMMDD
    hour = int(request.args.get('hour', -1))
    max_fhr = int(request.args.get('max_fhr', 18))

    if not date_str:
        return jsonify({'error': 'date required (YYYYMMDD)'}), 400
    if hour < 0 or hour > 23:
        return jsonify({'error': 'hour required (0-23)'}), 400

    try:
        datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format, use YYYYMMDD'}), 400

    cycle_key = f"{date_str}/{hour:02d}z"

    # Determine source
    from datetime import timezone
    date_dt = datetime.strptime(f"{date_str}{hour:02d}", '%Y%m%d%H').replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - date_dt).total_seconds() / 3600
    source = "AWS archive" if age_hours > 48 else "NOMADS"

    # Download in background
    def download_cycle():
        from smart_hrrr.orchestrator import download_gribs_parallel
        from smart_hrrr.io import create_output_structure

        try:
            create_output_structure('hrrr', date_str, hour)
            fhrs = list(range(max_fhr + 1))
            results = download_gribs_parallel(
                model='hrrr',
                date_str=date_str,
                cycle_hour=hour,
                forecast_hours=fhrs,
                max_threads=4
            )
            success = sum(1 for ok in results.values() if ok)
            logger.info(f"Cycle request {cycle_key}: {success}/{len(fhrs)} forecast hours downloaded")
            data_manager.scan_available_cycles()
        except Exception as e:
            logger.warning(f"Cycle request {cycle_key} failed: {e}")

    t = threading.Thread(target=download_cycle, daemon=True)
    t.start()

    est_minutes = 5 if age_hours <= 48 else 15
    return jsonify({
        'success': True,
        'message': f'Downloading {cycle_key} F00-F{max_fhr:02d} from {source} (~{est_minutes} min)',
        'cycle_key': cycle_key,
        'source': source,
        'est_minutes': est_minutes,
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
    parser = argparse.ArgumentParser(description='HRRR Cross-Section Dashboard')
    parser.add_argument('--auto-update', action='store_true', help='Download latest data before starting')
    parser.add_argument('--preload', type=int, default=2, help='Number of latest cycles to pre-load')
    parser.add_argument('--max-hours', type=int, default=18, help='Max forecast hour to download')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--production', action='store_true', help='Enable rate limiting')

    args = parser.parse_args()
    app.config['PRODUCTION'] = args.production

    # Optionally download fresh data
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

    # Scan for available cycles
    logger.info(f"Scanning for available cycles...")
    cycles = data_manager.scan_available_cycles()

    if cycles:
        logger.info(f"Found {len(cycles)} cycles:")
        for c in cycles:
            fhrs_str = ', '.join(f'F{f:02d}' for f in c['available_fhrs'])
            logger.info(f"  {c['display']}: [{fhrs_str}]")
    else:
        logger.info("No data found yet. Waiting for auto_update.py to download data...")

    # Pre-load latest cycles in background so Flask starts immediately
    if args.preload > 0 and cycles:
        def _startup_preload():
            time.sleep(2)  # Let Flask bind first
            logger.info(f"Background: Pre-loading latest {args.preload} cycles...")
            try:
                data_manager.preload_latest_cycles(n_cycles=args.preload)
            except Exception as e:
                logger.warning(f"Cycle preload failed: {e}")
        threading.Thread(target=_startup_preload, daemon=True).start()

    # Background re-scan thread: periodically check for newly downloaded data + disk eviction
    def background_rescan():
        while True:
            time.sleep(60)  # Re-scan every 60 seconds
            try:
                data_manager.scan_available_cycles()
            except Exception as e:
                logger.warning(f"Background rescan failed: {e}")

            # Check disk usage every 10 minutes (use modulo on minute)
            if int(time.time()) % 600 < 60:
                try:
                    usage = get_disk_usage_gb()
                    if usage > DISK_LIMIT_GB:
                        logger.info(f"Disk usage {usage:.1f}GB > {DISK_LIMIT_GB}GB limit, evicting...")
                        disk_evict_least_popular()
                        data_manager.scan_available_cycles()
                except Exception as e:
                    logger.warning(f"Disk eviction check failed: {e}")

    rescan_thread = threading.Thread(target=background_rescan, daemon=True)
    rescan_thread.start()

    mem_mb = data_manager.xsect.get_memory_usage() if data_manager.xsect else 0
    disk_gb = get_disk_usage_gb()
    logger.info("")
    logger.info("=" * 60)
    logger.info("HRRR Cross-Section Dashboard")
    logger.info(f"Pre-loaded: {len(data_manager.loaded_cycles)} cycles ({mem_mb:.0f} MB)")
    logger.info(f"Disk: {disk_gb:.1f}GB / {DISK_LIMIT_GB}GB")
    logger.info(f"Auto-refreshing cycle list every 60s")
    logger.info(f"Open: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()
