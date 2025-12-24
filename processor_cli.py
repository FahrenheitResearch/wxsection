#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from pathlib import Path

from smart_hrrr.utils import setup_logging, parse_hour_range
from smart_hrrr.io import create_output_structure, move_old_files
from smart_hrrr.orchestrator import process_model_run, monitor_and_process_latest


def main():
    parser = argparse.ArgumentParser(description="Smart HRRR processor")
    parser.add_argument("date", nargs="?", help="Date in YYYYMMDD format (or use --latest)")
    parser.add_argument("hour", type=int, nargs="?", help="Model run hour (0-23)")
    parser.add_argument("--latest", action="store_true", help="Monitor and process latest model run")
    parser.add_argument("--model", default="HRRR", help="Model name (default: HRRR)")
    parser.add_argument("--max-hours", type=int, help="Maximum forecast hours (auto-detected by default)")
    parser.add_argument("--hours", help="Forecast hour range (e.g. 6-12 or 0,1,2)")
    parser.add_argument("--categories", help="Categories to process (comma-separated)")
    parser.add_argument("--fields", help="Specific fields to process (comma-separated)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--force", action="store_true", help="Force reprocess existing files")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument("--cleanup", action="store_true", help="Move old files before processing")
    parser.add_argument("--check-interval", type=int, default=30, help="Check interval (seconds) for --latest")
    parser.add_argument("--download-threads", type=int, default=8, help="Parallel download threads (default: 8)")

    args = parser.parse_args()

    if args.latest and (args.date or args.hour is not None):
        parser.error("Cannot specify date/hour with --latest mode")

    if not args.latest and (not args.date or args.hour is None):
        parser.error("Must specify date and hour (or use --latest)")

    if args.latest:
        logger = setup_logging(debug=args.debug)
    else:
        output_dirs = create_output_structure(args.model, args.date, args.hour)
        logger = setup_logging(debug=args.debug, output_dir=output_dirs["run"])

    if args.cleanup:
        move_old_files()

    categories = [s.strip() for s in args.categories.split(",")] if args.categories else None
    fields = [s.strip() for s in args.fields.split(",")] if args.fields else None

    workers = min(args.workers, mp.cpu_count()) if args.workers > 0 else 1

    if args.latest:
        hour_range = parse_hour_range(args.hours)
        monitor_and_process_latest(categories=categories, fields=fields, workers=workers,
                                 check_interval=args.check_interval, force_reprocess=args.force,
                                 hour_range=hour_range, max_hours=args.max_hours, model=args.model,
                                 download_threads=args.download_threads)
    else:
        hour_range = parse_hour_range(args.hours)
        if hour_range is not None:
            forecast_hours = hour_range
        else:
            if args.max_hours is None:
                max_hours = 48 if args.hour in (0, 6, 12, 18) else 18
            else:
                max_hours = args.max_hours
            forecast_hours = list(range(0, max_hours + 1))

        process_model_run(model=args.model, date=args.date, hour=args.hour, forecast_hours=forecast_hours,
                         categories=categories, fields=fields, max_workers=workers, force_reprocess=args.force)


if __name__ == "__main__":
    main()