#!/usr/bin/env python3
"""
Full Pipeline Benchmark

Downloads, processes, and serves a complete HRRR cycle with timing.
Tests everything including diurnal analysis.

Usage:
    python tools/benchmark_pipeline.py --latest
    python tools/benchmark_pipeline.py --date 20251227 --hour 09
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Benchmark:
    """Tracks timing for pipeline stages."""

    def __init__(self):
        self.stages = {}
        self.current_stage = None
        self.stage_start = None

    def start(self, stage: str):
        self.current_stage = stage
        self.stage_start = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"STAGE: {stage}")
        logger.info(f"{'='*60}")

    def end(self, details: dict = None):
        if self.current_stage and self.stage_start:
            duration = time.time() - self.stage_start
            self.stages[self.current_stage] = {
                'duration': duration,
                'details': details or {}
            }
            logger.info(f"Completed in {duration:.1f}s")
        self.current_stage = None
        self.stage_start = None

    def report(self):
        logger.info("")
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)

        total = 0
        for stage, data in self.stages.items():
            duration = data['duration']
            total += duration
            details = data.get('details', {})

            logger.info(f"{stage:40} {duration:8.1f}s")
            for k, v in details.items():
                logger.info(f"  - {k}: {v}")

        logger.info("-" * 60)
        logger.info(f"{'TOTAL':40} {total:8.1f}s ({total/60:.1f} min)")
        logger.info("=" * 60)

        return self.stages


def get_latest_cycle():
    """Get the latest available HRRR cycle."""
    from smart_hrrr.availability import get_latest_cycle as _get_latest
    cycle, cycle_time = _get_latest('hrrr')
    return cycle, cycle_time


def download_cycle(date_str: str, hour: int, forecast_hours: list, benchmark: Benchmark):
    """Download GRIB files."""
    benchmark.start("Download GRIB Files")

    from smart_hrrr.orchestrator import download_gribs_parallel
    from smart_hrrr.io import create_output_structure

    cycle = f"{date_str}{hour:02d}"
    output_dirs = create_output_structure('hrrr', date_str, hour)

    results = download_gribs_parallel(
        cycle=cycle,
        forecast_hours=forecast_hours,
        output_dirs=output_dirs,
        model='hrrr',
        max_threads=8,
    )

    success_count = sum(1 for ok in results.values() if ok)

    # Count downloaded files
    run_dir = Path(f"outputs/hrrr/{date_str}/{hour:02d}z")
    prs_files = list(run_dir.glob("F*/*wrfprs*.grib2"))
    sfc_files = list(run_dir.glob("F*/*wrfsfc*.grib2"))

    benchmark.end({
        'hours': len(forecast_hours),
        'downloaded': success_count,
        'prs_files': len(prs_files),
        'sfc_files': len(sfc_files),
    })

    return success_count > 0


def process_maps(date_str: str, hour: int, forecast_hours: list, benchmark: Benchmark, workers: int = 4):
    """Process all map products."""
    benchmark.start("Process Map Products")

    from smart_hrrr.orchestrator import process_model_run

    results = process_model_run(
        model='hrrr',
        date=date_str,
        hour=hour,
        forecast_hours=forecast_hours,
        max_workers=workers,
    )

    successful = sum(1 for r in results if r.get('success'))

    # Count output files
    run_dir = Path(f"outputs/hrrr/{date_str}/{hour:02d}z")
    png_files = list(run_dir.glob("F*/**/*.png"))

    benchmark.end({
        'hours_processed': successful,
        'total_hours': len(forecast_hours),
        'png_files': len(png_files),
    })

    return successful > 0


def run_diurnal(date_str: str, hour: int, forecast_hours: list, benchmark: Benchmark):
    """Run diurnal temperature analysis."""
    benchmark.start("Diurnal Analysis")

    import subprocess

    end_fhr = max(forecast_hours)

    cmd = [
        sys.executable, "tools/process_diurnal.py",
        "--date", date_str,
        "--hour", str(hour),
        "--end-fhr", str(end_fhr),
        "--workers", "4",
        "--rolling",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        success = result.returncode == 0

        # Count diurnal outputs
        diurnal_dir = Path(f"outputs/diurnal/{date_str}/{hour:02d}z")
        diurnal_files = list(diurnal_dir.glob("**/*.png")) if diurnal_dir.exists() else []

        benchmark.end({
            'success': success,
            'end_fhr': end_fhr,
            'output_files': len(diurnal_files),
        })

        return success
    except subprocess.TimeoutExpired:
        benchmark.end({'error': 'timeout'})
        return False
    except Exception as e:
        benchmark.end({'error': str(e)})
        return False


def precache_xsect(date_str: str, hour: int, max_hours: int, benchmark: Benchmark):
    """Pre-cache cross-section data."""
    benchmark.start("Pre-cache Cross-Sections")

    from core.cross_section_interactive import InteractiveCrossSection

    run_dir = f"outputs/hrrr/{date_str}/{hour:02d}z"

    ixs = InteractiveCrossSection(cache_dir='cache/dashboard/xsect')
    loaded = ixs.load_run(run_dir, max_hours=max_hours, workers=1)

    memory_mb = ixs.get_memory_usage()

    # Count cache files
    cache_dir = Path('cache/dashboard/xsect')
    cache_files = list(cache_dir.glob("*.npz")) if cache_dir.exists() else []

    benchmark.end({
        'hours_loaded': loaded,
        'memory_mb': int(memory_mb),
        'cache_files': len(cache_files),
    })

    return loaded > 0


def verify_outputs(date_str: str, hour: int, benchmark: Benchmark):
    """Verify all outputs are present."""
    benchmark.start("Verify Outputs")

    run_dir = Path(f"outputs/hrrr/{date_str}/{hour:02d}z")
    diurnal_dir = Path(f"outputs/diurnal/{date_str}/{hour:02d}z")
    cache_dir = Path("cache/dashboard/xsect")

    results = {
        'run_dir_exists': run_dir.exists(),
        'forecast_hours': len(list(run_dir.glob("F*"))),
        'png_products': len(list(run_dir.glob("F*/**/*.png"))),
        'grib_files': len(list(run_dir.glob("F*/*.grib2"))),
        'diurnal_dir_exists': diurnal_dir.exists(),
        'diurnal_files': len(list(diurnal_dir.glob("**/*.png"))) if diurnal_dir.exists() else 0,
        'cache_files': len(list(cache_dir.glob("*.npz"))) if cache_dir.exists() else 0,
    }

    benchmark.end(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Benchmark")
    parser.add_argument("--latest", action="store_true", help="Use latest available cycle")
    parser.add_argument("--date", type=str, help="Date (YYYYMMDD)")
    parser.add_argument("--hour", type=int, help="Hour (0-23)")
    parser.add_argument("--max-fhr", type=int, default=48, help="Max forecast hour to process")
    parser.add_argument("--xsect-hours", type=int, default=6, help="Hours to pre-cache for cross-sections")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes for map generation")
    parser.add_argument("--skip-download", action="store_true", help="Skip download (use existing files)")
    parser.add_argument("--skip-maps", action="store_true", help="Skip map processing")
    parser.add_argument("--skip-diurnal", action="store_true", help="Skip diurnal analysis")
    parser.add_argument("--skip-cache", action="store_true", help="Skip cross-section caching")

    args = parser.parse_args()

    benchmark = Benchmark()

    # Determine cycle
    if args.latest:
        cycle, cycle_time = get_latest_cycle()
        if cycle is None:
            logger.error("No available cycles found")
            sys.exit(1)
        date_str = cycle_time.strftime("%Y%m%d")
        hour = cycle_time.hour
    elif args.date and args.hour is not None:
        date_str = args.date
        hour = args.hour
    else:
        parser.error("Must specify --latest or --date and --hour")

    forecast_hours = list(range(args.max_fhr + 1))

    logger.info("=" * 60)
    logger.info("HRRR FULL PIPELINE BENCHMARK")
    logger.info(f"Cycle: {date_str} {hour:02d}Z")
    logger.info(f"Forecast hours: F00-F{args.max_fhr:02d} ({len(forecast_hours)} hours)")
    logger.info(f"Workers: {args.workers}")
    logger.info("=" * 60)

    pipeline_start = time.time()

    # 1. Download
    if not args.skip_download:
        download_cycle(date_str, hour, forecast_hours, benchmark)
    else:
        logger.info("Skipping download (--skip-download)")

    # 2. Process maps
    if not args.skip_maps:
        process_maps(date_str, hour, forecast_hours, benchmark, args.workers)
    else:
        logger.info("Skipping map processing (--skip-maps)")

    # 3. Diurnal analysis
    if not args.skip_diurnal:
        run_diurnal(date_str, hour, forecast_hours, benchmark)
    else:
        logger.info("Skipping diurnal analysis (--skip-diurnal)")

    # 4. Pre-cache cross-sections
    if not args.skip_cache:
        precache_xsect(date_str, hour, args.xsect_hours, benchmark)
    else:
        logger.info("Skipping cross-section caching (--skip-cache)")

    # 5. Verify
    results = verify_outputs(date_str, hour, benchmark)

    # Report
    benchmark.report()

    pipeline_duration = time.time() - pipeline_start

    logger.info("")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {pipeline_duration:.1f}s ({pipeline_duration/60:.1f} min)")
    logger.info("")
    logger.info(f"Dashboard ready at: outputs/hrrr/{date_str}/{hour:02d}z")
    logger.info(f"Start with: python tools/unified_dashboard.py --data-dir outputs/hrrr/{date_str}/{hour:02d}z")

    # Save benchmark results
    results_file = Path(f"outputs/benchmark_{date_str}_{hour:02d}z.json")
    with open(results_file, 'w') as f:
        json.dump({
            'cycle': f"{date_str}_{hour:02d}z",
            'forecast_hours': len(forecast_hours),
            'total_duration': pipeline_duration,
            'stages': {k: v['duration'] for k, v in benchmark.stages.items()},
            'details': {k: v['details'] for k, v in benchmark.stages.items()},
        }, f, indent=2)

    logger.info(f"Benchmark saved to: {results_file}")


if __name__ == "__main__":
    main()
