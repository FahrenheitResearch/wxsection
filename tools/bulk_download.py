#!/usr/bin/env python3
"""
Bulk HRRR Archive Downloader

Downloads HRRR pressure + surface GRIB files from AWS for archival storage.
Designed for filling an external drive with historical data for fast local access.

Usage:
    # Download all of January 2025
    python tools/bulk_download.py --start 20250101 --end 20250131 --output /mnt/e/hrrr

    # Download a single day
    python tools/bulk_download.py --start 20250314 --end 20250314 --output /mnt/e/hrrr

    # Download with smoke data (wrfnat) — much larger (~7x)
    python tools/bulk_download.py --start 20250701 --end 20250731 --output /mnt/e/hrrr --include-smoke

    # Custom init hours and forecast hours
    python tools/bulk_download.py --start 20250101 --end 20250131 --output /mnt/e/hrrr \
        --inits 0 12 --fhrs 0 6 12 18

    # Dry run to see what would be downloaded
    python tools/bulk_download.py --start 20250101 --end 20250107 --output /mnt/e/hrrr --dry-run

Directory structure matches the dashboard layout:
    {output}/YYYYMMDD/HHz/F##/hrrr.tHHz.wrfprsf##.grib2
                              /hrrr.tHHz.wrfsfcf##.grib2
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_hrrr.orchestrator import download_forecast_hour
from model_config import get_model_registry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_INITS = [0, 6, 12, 18]
DEFAULT_FHRS = [0, 3, 6, 9, 12, 15, 18]


def date_range(start_str, end_str):
    """Generate dates from start to end (inclusive)."""
    start = datetime.strptime(start_str, '%Y%m%d')
    end = datetime.strptime(end_str, '%Y%m%d')
    current = start
    while current <= end:
        yield current.strftime('%Y%m%d')
        current += timedelta(days=1)


def count_existing(output_dir, date_str, hour, fhrs, file_types):
    """Count how many FHRs already have all required files."""
    existing = 0
    registry = get_model_registry()
    model = registry.get_model('hrrr')
    for fhr in fhrs:
        fhr_dir = output_dir / date_str / f"{hour:02d}z" / f"F{fhr:02d}"
        all_present = True
        for ft in file_types:
            filename = model.get_filename(hour, ft, fhr)
            if not (fhr_dir / filename).exists():
                all_present = False
                break
        if all_present:
            existing += 1
    return existing


def download_init(output_dir, date_str, hour, fhrs, file_types, max_threads=4):
    """Download all FHRs for a single init cycle. Returns (success, skipped, failed)."""
    registry = get_model_registry()
    model = registry.get_model('hrrr')

    success = 0
    skipped = 0
    failed = 0

    def _download_one(fhr):
        fhr_dir = output_dir / date_str / f"{hour:02d}z" / f"F{fhr:02d}"
        fhr_dir.mkdir(parents=True, exist_ok=True)

        # Check if all files already exist
        all_present = True
        for ft in file_types:
            filename = model.get_filename(hour, ft, fhr)
            if not (fhr_dir / filename).exists():
                all_present = False
                break
        if all_present:
            return fhr, 'skipped'

        ok = download_forecast_hour('hrrr', date_str, hour, fhr, fhr_dir, file_types)
        return fhr, 'ok' if ok else 'failed'

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = {pool.submit(_download_one, fhr): fhr for fhr in fhrs}
        for future in as_completed(futures):
            fhr, status = future.result()
            if status == 'ok':
                success += 1
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1
                logger.warning(f"  F{fhr:02d} FAILED")

    return success, skipped, failed


def estimate_size(n_fhrs, include_smoke):
    """Estimate download size in GB."""
    # wrfprs ~100MB, wrfsfc ~2MB, wrfnat ~670MB
    per_fhr_mb = 772 if include_smoke else 102
    return (n_fhrs * per_fhr_mb) / 1024


def main():
    parser = argparse.ArgumentParser(
        description='Bulk download HRRR archive data from AWS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--start', required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('--end', required=True, help='End date (YYYYMMDD)')
    parser.add_argument('--output', required=True, help='Output directory (e.g. /mnt/e/hrrr)')
    parser.add_argument('--inits', type=int, nargs='+', default=DEFAULT_INITS,
                        help=f'Init hours (default: {DEFAULT_INITS})')
    parser.add_argument('--fhrs', type=int, nargs='+', default=DEFAULT_FHRS,
                        help=f'Forecast hours (default: {DEFAULT_FHRS})')
    parser.add_argument('--include-smoke', action='store_true',
                        help='Include wrfnat files for smoke (~7x more data)')
    parser.add_argument('--threads', type=int, default=4,
                        help='Parallel download threads per init (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without downloading')
    args = parser.parse_args()

    # Validate dates
    try:
        datetime.strptime(args.start, '%Y%m%d')
        datetime.strptime(args.end, '%Y%m%d')
    except ValueError:
        print("Error: dates must be YYYYMMDD format")
        sys.exit(1)

    if args.start > args.end:
        print("Error: start date must be <= end date")
        sys.exit(1)

    output_dir = Path(args.output)
    file_types = ['pressure', 'surface']
    if args.include_smoke:
        file_types.append('native')

    dates = list(date_range(args.start, args.end))
    n_days = len(dates)
    n_inits = len(args.inits)
    n_fhrs = len(args.fhrs)
    total_cycles = n_days * n_inits
    total_fhrs = total_cycles * n_fhrs
    est_gb = estimate_size(total_fhrs, args.include_smoke)

    print("=" * 60)
    print("  HRRR Bulk Archive Downloader")
    print("=" * 60)
    print(f"  Date range:  {args.start} → {args.end} ({n_days} days)")
    print(f"  Init hours:  {', '.join(f'{h:02d}z' for h in args.inits)}")
    print(f"  FHRs:        {', '.join(f'F{f:02d}' for f in args.fhrs)}")
    print(f"  File types:  {', '.join(file_types)}")
    print(f"  Output:      {output_dir}")
    print(f"  Threads:     {args.threads}")
    print(f"  Total:       {total_cycles} cycles × {n_fhrs} FHRs = {total_fhrs} downloads")
    print(f"  Est. size:   ~{est_gb:.1f} GB")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run — listing cycles:")
        for date_str in dates:
            for hour in args.inits:
                existing = count_existing(output_dir, date_str, hour, args.fhrs, file_types)
                status = f"({existing}/{n_fhrs} exist)" if existing > 0 else ""
                print(f"  {date_str} {hour:02d}z  F{args.fhrs[0]:02d}-F{args.fhrs[-1]:02d}  {status}")
        print(f"\nTotal estimated: ~{est_gb:.1f} GB")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    overall_start = time.time()
    total_success = 0
    total_skipped = 0
    total_failed = 0
    cycles_done = 0

    for date_str in dates:
        for hour in args.inits:
            cycles_done += 1
            cycle_label = f"{date_str} {hour:02d}z"
            existing = count_existing(output_dir, date_str, hour, args.fhrs, file_types)

            if existing == n_fhrs:
                total_skipped += n_fhrs
                logger.info(f"[{cycles_done}/{total_cycles}] {cycle_label} — all {n_fhrs} FHRs exist, skipping")
                continue

            logger.info(f"[{cycles_done}/{total_cycles}] {cycle_label} — downloading ({existing}/{n_fhrs} exist)...")
            cycle_start = time.time()

            success, skipped, failed = download_init(
                output_dir, date_str, hour, args.fhrs, file_types, args.threads
            )

            dur = time.time() - cycle_start
            total_success += success
            total_skipped += skipped
            total_failed += failed

            elapsed = time.time() - overall_start
            fhrs_done = total_success + total_skipped + total_failed
            rate = fhrs_done / (elapsed / 60) if elapsed > 0 else 0
            remaining_fhrs = total_fhrs - fhrs_done
            eta_min = remaining_fhrs / rate if rate > 0 else 0

            logger.info(
                f"  {cycle_label} done in {dur:.0f}s — "
                f"{success} new, {skipped} skipped, {failed} failed  |  "
                f"Overall: {fhrs_done}/{total_fhrs}  ETA: {eta_min:.0f}m"
            )

    elapsed = time.time() - overall_start
    elapsed_min = elapsed / 60

    print()
    print("=" * 60)
    print("  Download Complete")
    print("=" * 60)
    print(f"  Time:     {elapsed_min:.1f} minutes")
    print(f"  New:      {total_success}")
    print(f"  Skipped:  {total_skipped} (already existed)")
    print(f"  Failed:   {total_failed}")
    print(f"  Output:   {output_dir}")
    print("=" * 60)

    if total_failed > 0:
        print(f"\n  {total_failed} FHRs failed — re-run the same command to retry (existing files are skipped)")


if __name__ == '__main__':
    main()
