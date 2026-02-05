#!/usr/bin/env python3
"""
Auto-Update Script for HRRR Cross-Section Dashboard

Continuously monitors for new HRRR cycles and progressively downloads
forecast hours (F00-F18) as they become available on NOAA servers.

Maintains data for the latest 2 init cycles.

Usage:
    python tools/auto_update.py                    # Default: check every 2 min
    python tools/auto_update.py --interval 3       # Check every 3 minutes
    python tools/auto_update.py --once             # Run once and exit
    python tools/auto_update.py --max-hours 18     # Download up to F18
"""

import argparse
import json
import logging
import sys
import time
import signal
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

running = True
BASE_DIR = Path("outputs/hrrr")


def signal_handler(sig, frame):
    global running
    logger.info("Shutting down...")
    running = False


def get_latest_two_cycles():
    """Determine the latest 2 expected HRRR init cycles based on current UTC time.

    HRRR runs every hour. Data typically available ~45-90 min after init time.
    Returns list of (date_str, hour) tuples, newest first.
    """
    now = datetime.now(timezone.utc)

    # HRRR data typically available ~45 min after init time
    latest_possible = now - timedelta(minutes=50)

    cycles = []
    for offset in range(0, 6):  # Look back up to 6 hours
        cycle_time = latest_possible - timedelta(hours=offset)
        date_str = cycle_time.strftime("%Y%m%d")
        hour = cycle_time.hour
        cycles.append((date_str, hour))
        if len(cycles) >= 2:
            break

    return cycles


def get_downloaded_fhrs(date_str, hour):
    """Check which forecast hours are already downloaded for a cycle.

    An FHR is considered complete only if wrfprs, wrfsfc, AND wrfnat all exist.
    """
    run_dir = BASE_DIR / date_str / f"{hour:02d}z"
    downloaded = []
    for fhr in range(19):  # F00-F18
        fhr_dir = run_dir / f"F{fhr:02d}"
        if (fhr_dir.exists()
            and list(fhr_dir.glob("*wrfprs*.grib2"))
            and list(fhr_dir.glob("*wrfnat*.grib2"))):
            downloaded.append(fhr)
    return downloaded


def download_missing_fhrs(date_str, hour, max_hours=18):
    """Download any missing forecast hours for a cycle.

    Returns number of newly downloaded hours.
    """
    from smart_hrrr.orchestrator import download_gribs_parallel
    from smart_hrrr.io import create_output_structure

    downloaded = get_downloaded_fhrs(date_str, hour)
    needed = [f for f in range(max_hours + 1) if f not in downloaded]

    if not needed:
        return 0

    logger.info(f"Cycle {date_str}/{hour:02d}z: have F{downloaded if downloaded else 'none'}, "
                f"need F{needed}")

    # Create output structure
    create_output_structure('hrrr', date_str, hour)

    # Download missing forecast hours
    results = download_gribs_parallel(
        model='hrrr',
        date_str=date_str,
        cycle_hour=hour,
        forecast_hours=needed,
        max_threads=4  # Don't saturate bandwidth
    )

    new_count = sum(1 for ok in results.values() if ok)
    if new_count > 0:
        logger.info(f"  Downloaded {new_count}/{len(needed)} new hours for {date_str}/{hour:02d}z")

    return new_count


DISK_LIMIT_GB = 500
DISK_META_FILE = Path(__file__).parent.parent / 'data' / 'disk_meta.json'

def get_disk_usage_gb():
    """Get total disk usage of HRRR data directory in GB."""
    if not BASE_DIR.exists():
        return 0
    total = sum(f.stat().st_size for f in BASE_DIR.rglob("*") if f.is_file())
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

def cleanup_disk_if_needed():
    """Evict least-popular cycles if disk usage exceeds limit.

    Never evicts cycles from the latest 2 init hours or anything accessed
    in the last 2 hours.
    """
    usage = get_disk_usage_gb()
    if usage <= DISK_LIMIT_GB:
        return

    target = DISK_LIMIT_GB * 0.85
    meta = load_disk_meta()
    now = time.time()
    recent_cutoff = now - 7200  # 2 hours

    # Get latest 2 cycle keys (auto-update targets) - never evict these
    latest = get_latest_two_cycles()
    protected = {f"{d}_{h:02d}z" for d, h in latest}

    disk_cycles = []
    for date_dir in BASE_DIR.iterdir():
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
        if get_disk_usage_gb() <= target:
            break
        logger.info(f"Disk evict: {cycle_key} (last accessed {int((now - last_access)/3600)}h ago)")
        try:
            shutil.rmtree(hour_dir)
            parent = hour_dir.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            meta.pop(cycle_key, None)
        except Exception as e:
            logger.warning(f"Evict failed for {cycle_key}: {e}")

    save_disk_meta(meta)


def run_update_cycle(max_hours=18):
    """Check for and download new HRRR forecast hours for latest 2 cycles."""
    cycles = get_latest_two_cycles()

    total_new = 0
    for date_str, hour in cycles:
        try:
            new_count = download_missing_fhrs(date_str, hour, max_hours)
            total_new += new_count
        except Exception as e:
            logger.warning(f"Failed to update {date_str}/{hour:02d}z: {e}")

    # Space-based cleanup instead of age-based
    cleanup_disk_if_needed()

    return total_new


def main():
    global running

    parser = argparse.ArgumentParser(description="HRRR Auto-Update - Progressive Download")
    parser.add_argument("--interval", type=int, default=2, help="Check interval in minutes (default: 2)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--max-hours", type=int, default=18, help="Max forecast hours (default: 18)")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up old data")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("HRRR Progressive Auto-Update Service")
    logger.info(f"Check interval: {args.interval} min | Max FHR: F{args.max_hours:02d}")
    logger.info(f"Maintaining latest 2 init cycles with F00-F{args.max_hours:02d}")
    logger.info("=" * 60)

    if args.once:
        new = run_update_cycle(args.max_hours)
        logger.info(f"Downloaded {new} new forecast hours")
        return

    while running:
        try:
            cycles = get_latest_two_cycles()
            logger.info(f"Tracking cycles: {', '.join(f'{d}/{h:02d}z' for d, h in cycles)}")

            new = run_update_cycle(args.max_hours)
            if new > 0:
                logger.info(f"Total new downloads this pass: {new}")
            else:
                logger.info("All forecast hours up to date")

        except Exception as e:
            logger.exception(f"Update failed: {e}")

        # Sleep in 1-second increments so we can respond to signals
        for _ in range(args.interval * 60):
            if not running:
                break
            time.sleep(1)

    logger.info("Auto-update service stopped")


if __name__ == "__main__":
    main()
