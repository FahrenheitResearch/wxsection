#!/usr/bin/env python3
"""
Auto-Update Script for Cross-Section Dashboard (Multi-Model)

Continuously monitors for new model cycles and progressively downloads
forecast hours as they become available on NOAA servers.

Supports HRRR, GFS, and RRFS models.

Usage:
    python tools/auto_update.py                              # Default: HRRR only
    python tools/auto_update.py --models hrrr,gfs            # HRRR + GFS
    python tools/auto_update.py --models hrrr,gfs,rrfs       # All models
    python tools/auto_update.py --interval 3                  # Check every 3 minutes
    python tools/auto_update.py --once                        # Run once and exit
    python tools/auto_update.py --max-hours 18                # Download up to F18 (HRRR)
"""

import argparse
import json
import logging
import sys
import time
import signal
import shutil
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_config import get_model_registry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

running = True
OUTPUTS_BASE = Path("outputs")
STATUS_FILE = Path("/tmp/auto_update_status.json")

SYNOPTIC_HOURS = {0, 6, 12, 18}

# Per-model: which GRIB file patterns confirm a complete FHR download
MODEL_REQUIRED_PATTERNS = {
    'hrrr': ['*wrfprs*.grib2', '*wrfsfc*.grib2', '*wrfnat*.grib2'],
    'gfs':  ['*pgrb2.0p25*'],
    'rrfs': ['*prslev*.grib2'],
}

# Per-model: which file_types to request from the orchestrator
MODEL_FILE_TYPES = {
    'hrrr': ['pressure', 'surface', 'native'],  # wrfprs, wrfsfc, wrfnat
    'gfs':  ['pressure'],                        # pgrb2.0p25 (has surface data in it)
    'rrfs': ['pressure'],                        # prslev (has surface data in it)
}

# Per-model: which FHRs to download
MODEL_FORECAST_HOURS = {
    'hrrr': list(range(19)),                          # F00-F18 every hour (synoptic extends to F48)
    'gfs':  list(range(0, 385, 6)),                   # F00-F384 every 6 hours (65 FHRs)
    'rrfs': list(range(19)),                          # F00-F18 every hour
}
MODEL_DEFAULT_MAX_FHR = {
    'hrrr': 18,
    'gfs':  384,
    'rrfs': 18,
}

# Per-model: data availability lag (minutes after init time)
MODEL_AVAILABILITY_LAG = {
    'hrrr': 50,
    'gfs':  180,   # GFS takes ~3 hours to start publishing
    'rrfs': 120,   # RRFS takes ~2 hours
}


def write_status(status_dict):
    """Atomically write auto-update status to JSON file for dashboard to read."""
    try:
        tmp = STATUS_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(status_dict))
        tmp.rename(STATUS_FILE)
    except Exception:
        pass  # Best-effort, don't disrupt downloads


def clear_status():
    """Write idle status (no active downloads)."""
    write_status({"ts": time.time(), "models": {}})


def signal_handler(sig, frame):
    global running
    logger.info("Shutting down...")
    running = False


def get_base_dir(model):
    """Get output directory for a model."""
    return OUTPUTS_BASE / model


def get_latest_cycles(model='hrrr', count=2):
    """Determine the latest N expected cycles based on current UTC time.

    Returns list of (date_str, hour) tuples, newest first.
    """
    registry = get_model_registry()
    model_config = registry.get_model(model)
    if not model_config:
        return []

    now = datetime.now(timezone.utc)
    lag_minutes = MODEL_AVAILABILITY_LAG.get(model, 60)
    latest_possible = now - timedelta(minutes=lag_minutes)

    valid_hours = set(model_config.forecast_cycles)

    cycles = []
    for offset in range(0, 48):  # Look back up to 48 hours (GFS runs only 4x/day)
        cycle_time = latest_possible - timedelta(hours=offset)
        if cycle_time.hour in valid_hours:
            date_str = cycle_time.strftime("%Y%m%d")
            hour = cycle_time.hour
            cycles.append((date_str, hour))
            if len(cycles) >= count:
                break

    return cycles


def get_model_fhrs(model, max_fhr=None):
    """Get the list of FHRs this model should download."""
    fhrs = MODEL_FORECAST_HOURS.get(model, list(range(19)))
    if max_fhr is not None:
        fhrs = [f for f in fhrs if f <= max_fhr]
    return fhrs


def get_downloaded_fhrs(model, date_str, hour, max_fhr=None):
    """Check which forecast hours are already downloaded for a cycle.

    Uses per-model required patterns to determine completeness.
    """
    base_dir = get_base_dir(model)
    run_dir = base_dir / date_str / f"{hour:02d}z"
    patterns = MODEL_REQUIRED_PATTERNS.get(model, ['*.grib2'])

    fhrs_to_check = get_model_fhrs(model, max_fhr)
    downloaded = []
    for fhr in fhrs_to_check:
        fhr_dir = run_dir / f"F{fhr:02d}"
        if not fhr_dir.exists():
            continue
        # All required patterns must have at least one match
        if all(list(fhr_dir.glob(p)) for p in patterns):
            downloaded.append(fhr)
    return downloaded


def get_latest_synoptic_cycle():
    """Find the most recent synoptic cycle (00/06/12/18z) that should be available.

    Returns (date_str, hour) or None. Only used for HRRR extended forecasts.
    """
    now = datetime.now(timezone.utc)
    latest_possible = now - timedelta(minutes=50)

    for offset in range(0, 24):
        cycle_time = latest_possible - timedelta(hours=offset)
        if cycle_time.hour in SYNOPTIC_HOURS:
            return (cycle_time.strftime("%Y%m%d"), cycle_time.hour)
    return None


def download_missing_fhrs(model, date_str, hour, max_hours=18):
    """Download any missing forecast hours for a cycle.

    Returns number of newly downloaded hours.
    """
    from smart_hrrr.orchestrator import download_gribs_parallel
    from smart_hrrr.io import create_output_structure

    target_fhrs = get_model_fhrs(model, max_fhr=max_hours)
    downloaded = get_downloaded_fhrs(model, date_str, hour, max_fhr=max_hours)
    downloaded_set = set(downloaded)
    needed = [f for f in target_fhrs if f not in downloaded_set]

    if not needed:
        return 0

    logger.info(f"[{model.upper()}] Cycle {date_str}/{hour:02d}z: have {len(downloaded)} FHRs, "
                f"need {len(needed)} (F{needed[0]:02d}-F{needed[-1]:02d})")

    # Create output structure
    create_output_structure(model, date_str, hour)

    # Download missing forecast hours
    file_types = MODEL_FILE_TYPES.get(model, ['pressure'])
    results = download_gribs_parallel(
        model=model,
        date_str=date_str,
        cycle_hour=hour,
        forecast_hours=needed,
        max_threads=4,
        file_types=file_types,
    )

    new_count = sum(1 for ok in results.values() if ok)
    if new_count > 0:
        logger.info(f"  [{model.upper()}] Downloaded {new_count}/{len(needed)} new hours for {date_str}/{hour:02d}z")

    return new_count


def get_pending_work(model, max_hours=None):
    """Get list of (date_str, hour, fhr) tuples that need downloading for a model.

    Returns items sorted by cycle (newest first), then FHR ascending.
    """
    if max_hours is None:
        max_hours = MODEL_DEFAULT_MAX_FHR.get(model, 18)

    # HRRR: 2 latest cycles (current + previous for base FHRs)
    # GFS/RRFS: only latest cycle — no handoff
    n_cycles = 2 if model == 'hrrr' else 1
    cycles = get_latest_cycles(model, count=n_cycles)
    if not cycles:
        return []

    work = []
    for date_str, hour in cycles:
        target_fhrs = get_model_fhrs(model, max_fhr=max_hours)
        downloaded = set(get_downloaded_fhrs(model, date_str, hour, max_fhr=max_hours))
        needed = [f for f in target_fhrs if f not in downloaded]
        for fhr in needed:
            work.append((date_str, hour, fhr))

    # Extended HRRR: F19-F48 for latest synoptic cycle (lower priority, appended after base)
    if model == 'hrrr':
        syn = get_latest_synoptic_cycle()
        if syn:
            syn_date, syn_hour = syn
            existing = set(get_downloaded_fhrs(model, syn_date, syn_hour, max_fhr=48))
            # Deduplicate: only add FHRs not already in work list
            work_set = {(d, h, f) for d, h, f in work}
            extended_needed = [f for f in range(19, 49) if f not in existing]
            for fhr in extended_needed:
                if (syn_date, syn_hour, fhr) not in work_set:
                    work.append((syn_date, syn_hour, fhr))

    return work


def download_single_fhr(model, date_str, hour, fhr):
    """Download a single forecast hour for a model/cycle.

    Returns True if successfully downloaded, False otherwise.
    """
    from smart_hrrr.orchestrator import download_forecast_hour
    from smart_hrrr.io import create_output_structure, get_forecast_hour_dir

    run_dir = create_output_structure(model, date_str, hour)['run']
    fhr_dir = get_forecast_hour_dir(run_dir, fhr)
    file_types = MODEL_FILE_TYPES.get(model, ['pressure'])
    return download_forecast_hour(
        model=model,
        date_str=date_str,
        cycle_hour=hour,
        forecast_hour=fhr,
        output_dir=fhr_dir,
        file_types=file_types,
    )


DISK_LIMIT_GB = 500
DISK_META_FILE = Path(__file__).parent.parent / 'data' / 'disk_meta.json'

def get_disk_usage_gb(model='hrrr'):
    """Get total disk usage of a model's data directory in GB."""
    base_dir = get_base_dir(model)
    if not base_dir.exists():
        return 0
    total = sum(f.stat().st_size for f in base_dir.rglob("*") if f.is_file())
    return total / (1024 ** 3)

def load_disk_meta():
    if DISK_META_FILE.exists():
        try:
            with open(DISK_META_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_disk_meta(meta):
    DISK_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DISK_META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)

def cleanup_disk_if_needed(model='hrrr'):
    """Evict least-popular cycles if disk usage exceeds limit.

    Never evicts cycles from the latest 2 init hours or anything accessed
    in the last 2 hours.
    """
    base_dir = get_base_dir(model)
    usage = get_disk_usage_gb(model)
    if usage <= DISK_LIMIT_GB:
        return

    target = DISK_LIMIT_GB * 0.85
    meta = load_disk_meta()
    now = time.time()
    recent_cutoff = now - 7200  # 2 hours

    # Get latest 2 cycle keys (auto-update targets) - never evict these
    latest = get_latest_cycles(model, count=2)
    protected = {f"{d}_{h:02d}z" for d, h in latest}

    disk_cycles = []
    if not base_dir.exists():
        return
    for date_dir in base_dir.iterdir():
        if not date_dir.is_dir() or not date_dir.name.isdigit():
            continue
        for hour_dir in date_dir.iterdir():
            if not hour_dir.is_dir() or not hour_dir.name.endswith('z'):
                continue
            hour = hour_dir.name.replace('z', '')
            cycle_key = f"{date_dir.name}_{hour}z"
            if cycle_key in protected:
                continue
            last_access = meta.get(cycle_key, {}).get('last_accessed', 0)
            if last_access > recent_cutoff:
                continue
            disk_cycles.append((last_access, cycle_key, hour_dir))

    disk_cycles.sort()  # Oldest access first

    for last_access, cycle_key, hour_dir in disk_cycles:
        if get_disk_usage_gb(model) <= target:
            break
        logger.info(f"[{model.upper()}] Disk evict: {cycle_key} (last accessed {int((now - last_access)/3600)}h ago)")
        try:
            shutil.rmtree(hour_dir)
            parent = hour_dir.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            meta.pop(cycle_key, None)
        except Exception as e:
            logger.warning(f"Evict failed for {cycle_key}: {e}")

    save_disk_meta(meta)


def cleanup_old_extended(model='hrrr'):
    """Keep only 2 most recent synoptic runs with extended FHRs (F19-F48).

    Only applies to HRRR. Deletes F19+ directories from older synoptic cycles.
    """
    if model != 'hrrr':
        return

    base_dir = get_base_dir(model)
    synoptic_with_extended = []
    if not base_dir.exists():
        return

    for date_dir in sorted(base_dir.iterdir(), reverse=True):
        if not date_dir.is_dir() or not date_dir.name.isdigit():
            continue
        for hour_dir in sorted(date_dir.iterdir(), reverse=True):
            if not hour_dir.is_dir() or not hour_dir.name.endswith('z'):
                continue
            hour = int(hour_dir.name.replace('z', ''))
            if hour not in SYNOPTIC_HOURS:
                continue
            has_extended = any((hour_dir / f"F{f:02d}").exists() for f in range(19, 49))
            if has_extended:
                synoptic_with_extended.append(hour_dir)

    # Keep newest 2 with extended data, delete F19+ from the rest
    for old_dir in synoptic_with_extended[2:]:
        for fhr in range(19, 49):
            fhr_dir = old_dir / f"F{fhr:02d}"
            if fhr_dir.exists():
                try:
                    shutil.rmtree(fhr_dir)
                    logger.info(f"Cleaned extended F{fhr:02d} from {old_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean {fhr_dir}: {e}")


def run_update_cycle_for_model(model, max_hours=None):
    """Run one update pass for a single model.

    Returns total number of newly downloaded FHRs.
    """
    if max_hours is None:
        max_hours = MODEL_DEFAULT_MAX_FHR.get(model, 18)

    cycles = get_latest_cycles(model, count=2)
    if not cycles:
        logger.info(f"[{model.upper()}] No cycles available yet")
        return 0

    total_new = 0
    for date_str, hour in cycles:
        try:
            new_count = download_missing_fhrs(model, date_str, hour, max_hours)
            total_new += new_count
        except Exception as e:
            logger.warning(f"[{model.upper()}] Failed to update {date_str}/{hour:02d}z: {e}")

    # Extended HRRR: download F19-F48 for the most recent synoptic cycle
    if model == 'hrrr':
        syn = get_latest_synoptic_cycle()
        if syn:
            syn_date, syn_hour = syn
            try:
                existing = get_downloaded_fhrs(model, syn_date, syn_hour, max_fhr=48)
                extended_needed = [f for f in range(19, 49) if f not in existing]
                if extended_needed:
                    logger.info(f"[HRRR] Extended: {syn_date}/{syn_hour:02d}z needs "
                                f"F{extended_needed[0]:02d}-F{extended_needed[-1]:02d}")
                    new_ext = download_missing_fhrs(model, syn_date, syn_hour, max_hours=48)
                    total_new += new_ext
            except Exception as e:
                logger.warning(f"[HRRR] Extended download failed for {syn_date}/{syn_hour:02d}z: {e}")

    # Cleanup
    cleanup_disk_if_needed(model)
    cleanup_old_extended(model)

    return total_new


def run_download_pass_concurrent(work_queues, slot_limits, hrrr_max_fhr=None):
    """Run one concurrent download pass.

    Uses per-model slot limits (e.g., HRRR=3, GFS=1, RRFS=1) so slow RRFS
    transfers no longer block HRRR/GFS progress.
    """
    global running

    # Keep only models with work and at least one slot
    queues = {
        model: deque(items)
        for model, items in work_queues.items()
        if items and slot_limits.get(model, 0) > 0
    }
    if not queues:
        return 0

    active_by_model = {m: 0 for m in slot_limits}
    in_flight = {}
    total_new = 0
    hrrr_refresh_interval_sec = 45
    last_hrrr_refresh = 0.0

    # Prefer scheduling HRRR first, then others in name order.
    model_order = sorted(slot_limits.keys(), key=lambda m: (m != 'hrrr', m))
    total_workers = sum(max(0, slot_limits.get(m, 0)) for m in model_order)
    if total_workers <= 0:
        return 0

    # --- Per-model progress tracking for status file ---
    # Track cycle, totals, completions, in-flight FHRs, and last results per model
    model_status = {}
    for model, items in work_queues.items():
        if items and slot_limits.get(model, 0) > 0:
            # Determine cycle(s) — use the first item's cycle as primary
            d0, h0, _ = items[0]
            model_status[model] = {
                "cycle": f"{d0}/{h0:02d}z",
                "total": len(items),
                "done": 0,
                "in_flight": [],
                "last_ok": None,
                "last_fail": None,
            }
    pass_start = time.time()

    def _flush_status():
        write_status({"ts": time.time(), "started": pass_start, "models": model_status})

    def _update_in_flight():
        """Rebuild in_flight lists from the actual in_flight futures dict."""
        flying = {}
        for fut, (mn, ds, hr, fhr, ts) in in_flight.items():
            flying.setdefault(mn, []).append(f"F{fhr:02d}")
        for m in model_status:
            model_status[m]["in_flight"] = sorted(flying.get(m, []))

    def _drop_if_empty(model_name):
        q = queues.get(model_name)
        if q is not None and not q:
            queues.pop(model_name, None)

    def _schedule(model_name, pool):
        q = queues.get(model_name)
        if not q:
            return

        limit = max(0, slot_limits.get(model_name, 0))
        while running and q and active_by_model.get(model_name, 0) < limit:
            date_str, hour, fhr = q.popleft()
            future = pool.submit(download_single_fhr, model_name, date_str, hour, fhr)
            in_flight[future] = (model_name, date_str, hour, fhr, time.time())
            active_by_model[model_name] = active_by_model.get(model_name, 0) + 1
            logger.info(
                f"[{model_name.upper()}] Downloading {date_str}/{hour:02d}z F{fhr:02d} "
                f"(queued={len(q)}, active={active_by_model[model_name]}/{limit})"
            )
        _drop_if_empty(model_name)

    _flush_status()

    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        while running and (queues or in_flight):
            # Refresh HRRR queue while other models are still active so newly
            # published hours can start without waiting for the next outer pass.
            if (
                slot_limits.get('hrrr', 0) > 0
                and 'hrrr' not in queues
                and active_by_model.get('hrrr', 0) == 0
            ):
                now = time.time()
                if now - last_hrrr_refresh >= hrrr_refresh_interval_sec:
                    last_hrrr_refresh = now
                    refreshed = get_pending_work('hrrr', hrrr_max_fhr)
                    if refreshed:
                        queues['hrrr'] = deque(refreshed)
                        logger.info(f"[HRRR] Refreshed pending queue: {len(refreshed)} FHRs")
                        if 'hrrr' in model_status:
                            model_status['hrrr']['total'] += len(refreshed)
                        else:
                            d0, h0, _ = refreshed[0]
                            model_status['hrrr'] = {
                                "cycle": f"{d0}/{h0:02d}z",
                                "total": len(refreshed),
                                "done": 0,
                                "in_flight": [],
                                "last_ok": None,
                                "last_fail": None,
                            }

            for model_name in model_order:
                _schedule(model_name, pool)

            _update_in_flight()
            _flush_status()

            if not in_flight:
                break

            done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                model_name, date_str, hour, fhr, start_ts = in_flight.pop(fut)
                active_by_model[model_name] = max(0, active_by_model.get(model_name, 0) - 1)
                dur = time.time() - start_ts

                ok = False
                try:
                    ok = fut.result()
                except Exception as e:
                    logger.warning(f"[{model_name.upper()}] Failed {date_str}/{hour:02d}z F{fhr:02d}: {e}")

                if model_name in model_status:
                    model_status[model_name]["done"] += 1

                if ok:
                    total_new += 1
                    logger.info(f"[{model_name.upper()}] F{fhr:02d} complete ({dur:.1f}s)")
                    if model_name in model_status:
                        model_status[model_name]["last_ok"] = f"F{fhr:02d}"
                else:
                    logger.info(f"[{model_name.upper()}] F{fhr:02d} unavailable/failed ({dur:.1f}s)")
                    if model_name in model_status:
                        model_status[model_name]["last_fail"] = f"F{fhr:02d}"
                    # HRRR fail-fast: if Fxx isn't available, higher FHRs in same cycle
                    # are generally not available either.
                    if model_name == 'hrrr' and 'hrrr' in queues:
                        before = len(queues['hrrr'])
                        queues['hrrr'] = deque(
                            (d, h, f)
                            for d, h, f in queues['hrrr']
                            if not (d == date_str and h == hour and f > fhr)
                        )
                        pruned = before - len(queues['hrrr'])
                        if pruned > 0:
                            logger.info(
                                f"[HRRR] Pruned {pruned} higher FHRs after unavailable "
                                f"{date_str}/{hour:02d}z F{fhr:02d}"
                            )
                            if 'hrrr' in model_status:
                                model_status['hrrr']['total'] -= pruned
                        _drop_if_empty('hrrr')

            _update_in_flight()
            _flush_status()

    clear_status()
    return total_new


def main():
    global running

    parser = argparse.ArgumentParser(description="Multi-Model Auto-Update - Progressive Download")
    parser.add_argument("--models", type=str, default="hrrr",
                        help="Comma-separated models to update (default: hrrr)")
    parser.add_argument("--interval", type=int, default=2,
                        help="Check interval in minutes (default: 2)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--max-hours", type=int, default=None,
                        help="Max forecast hours for HRRR (default: per-model)")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up old data")
    parser.add_argument("--hrrr-slots", type=int, default=3,
                        help="Concurrent HRRR FHR downloads (default: 3)")
    parser.add_argument("--gfs-slots", type=int, default=1,
                        help="Concurrent GFS FHR downloads (default: 1)")
    parser.add_argument("--rrfs-slots", type=int, default=1,
                        help="Concurrent RRFS FHR downloads (default: 1)")

    args = parser.parse_args()
    models = [m.strip().lower() for m in args.models.split(',')]
    slot_limits = {
        'hrrr': max(0, args.hrrr_slots),
        'gfs': max(0, args.gfs_slots),
        'rrfs': max(0, args.rrfs_slots),
    }

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info(f"Multi-Model Auto-Update Service")
    logger.info(f"Models: {', '.join(m.upper() for m in models)}")
    logger.info(f"Check interval: {args.interval} min")
    for m in models:
        mfhr = args.max_hours if (args.max_hours and m == 'hrrr') else MODEL_DEFAULT_MAX_FHR.get(m, 18)
        slots = slot_limits.get(m, 1)
        logger.info(f"  {m.upper()}: F00-F{mfhr:02d} (base), slots={slots}")
    logger.info("=" * 60)

    if args.once:
        total = 0
        for model in models:
            mfhr = args.max_hours if (args.max_hours and model == 'hrrr') else None
            total += run_update_cycle_for_model(model, mfhr)
        logger.info(f"Downloaded {total} new forecast hours total")
        return

    # --------------- Concurrent slot-based download loop ---------------
    # Each model gets dedicated concurrency slots so slow RRFS/GFS transfers
    # cannot block HRRR progression.
    hrrr_max_fhr = args.max_hours if args.max_hours else None

    while running:
        try:
            # Build work queues per model
            work_queues = {}
            for model in models:
                if slot_limits.get(model, 0) <= 0:
                    continue
                mfhr = hrrr_max_fhr if model == 'hrrr' else None
                pending = get_pending_work(model, mfhr)
                if pending:
                    work_queues[model] = pending
                    cycles_str = set(f"{d}/{h:02d}z" for d, h, _ in pending)
                    logger.info(f"[{model.upper()}] Pending: {len(pending)} FHRs across {', '.join(sorted(cycles_str))}")

            if not work_queues:
                logger.info("All models up to date")
                clear_status()
                # Sleep and re-check
                for _ in range(args.interval * 60):
                    if not running:
                        break
                    time.sleep(1)
                continue

            total_new = run_download_pass_concurrent(
                work_queues=work_queues,
                slot_limits=slot_limits,
                hrrr_max_fhr=hrrr_max_fhr,
            )

            if total_new > 0:
                logger.info(f"Total new downloads this pass: {total_new}")

            # Run cleanup after each pass
            for model in models:
                cleanup_disk_if_needed(model)
                cleanup_old_extended(model)

        except Exception as e:
            logger.exception(f"Update failed: {e}")

        # Brief pause before re-scanning (much shorter since we're interleaved)
        for _ in range(30):  # 30 seconds between passes
            if not running:
                break
            time.sleep(1)

    clear_status()
    logger.info("Auto-update service stopped")


if __name__ == "__main__":
    main()
