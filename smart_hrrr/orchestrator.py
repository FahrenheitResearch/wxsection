import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import check_system_memory
from .products import get_available_products, get_missing_products, check_existing_products
from .io import create_output_structure, get_forecast_hour_dir
from .availability import check_forecast_hour_availability
from processor_base import HRRRProcessor  # keeps compatibility path


def _use_parallel_default() -> bool:
    """Check if parallel processing should be used by default"""
    return os.environ.get('HRRR_USE_PARALLEL', 'true').lower() in ('1', 'true', 'yes', 'on')


def download_gribs_parallel(cycle: str, forecast_hours: List[int], output_dirs: Dict[str, Path],
                            model: str, max_threads: int = 8) -> Dict[int, bool]:
    """Download GRIB files for multiple forecast hours in parallel using threads.

    Returns dict mapping forecast_hour -> success status
    """
    logger = logging.getLogger(__name__)

    def download_single(fhr: int) -> tuple[int, bool]:
        fhr_dir = get_forecast_hour_dir(output_dirs["run"], fhr)
        start = time.time()
        ok = download_grib_to_forecast_dir(cycle, fhr, fhr_dir, model)
        dur = time.time() - start
        if ok:
            logger.info(f"  F{fhr:02d} downloaded ({dur:.1f}s)")
        return fhr, ok

    results = {}
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(download_single, fhr): fhr for fhr in forecast_hours}
        for future in as_completed(futures):
            fhr, ok = future.result()
            results[fhr] = ok

    return results


def download_grib_to_forecast_dir(cycle: str, forecast_hour: int, fhr_dir: Path, model: str) -> bool:
    """Download GRIB files directly to forecast hour directory"""
    logger = logging.getLogger(__name__)

    try:
        proc = HRRRProcessor(model=model)
        ok = False
        for file_type in ["wrfprs", "wrfsfc"]:
            try:
                proc.download_hrrr_file(cycle, forecast_hour, fhr_dir, file_type)
                ok = True
                logger.info(f"Downloaded {file_type} for F{forecast_hour:02d}")
            except Exception as e:
                logger.warning(f"Failed to download {file_type} for F{forecast_hour:02d}: {e}")
        return ok
    except Exception as e:
        logger.error(f"GRIB download failed for F{forecast_hour:02d}: {e}")
        return False


def _run_subprocess_single(cycle: str, forecast_hour: int, fhr_dir: Path,
                          model: str, categories: Optional[List[str]], fields: Optional[List[str]],
                          timeout_s: int = 1800):
    """Run single hour processing via subprocess"""
    cmd = [
        sys.executable, "tools/process_single_hour.py", cycle, str(forecast_hour),
        "--output-dir", str(fhr_dir),
        "--use-local-grib",
        "--model", model,
    ]

    if categories:
        cmd += ["--categories", ",".join(categories)]
    if fields:
        cmd += ["--fields", ",".join(fields)]

    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=Path(__file__).resolve().parent.parent)


def process_forecast_hour_smart(
    cycle: str,
    forecast_hour: int,
    output_dirs: Dict[str, Path],
    categories: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    force_reprocess: bool = False,
    model: str = "hrrr",
):
    """Process a single forecast hour with smart duplicate detection and shared GRIB downloads"""
    logger = logging.getLogger(__name__)

    fhr_dir = get_forecast_hour_dir(output_dirs["run"], forecast_hour)

    # Check what's already been processed
    if not force_reprocess:
        all_products = get_available_products()
        if fields:
            all_products = fields
        elif categories:
            # Filter products by categories if specified
            from field_registry import FieldRegistry
            reg = FieldRegistry()
            filtered = []
            for cat in categories:
                filtered.extend(reg.get_fields_by_category(cat).keys())
            all_products = filtered

        missing, existing = get_missing_products(fhr_dir, all_products)

        if not missing:
            logger.info(f"✓ F{forecast_hour:02d} already complete ({len(existing)} products)")
            return {"success": True, "forecast_hour": forecast_hour, "skipped": True, "existing_count": len(existing)}

        logger.info(f"F{forecast_hour:02d}: {len(existing)} existing, {len(missing)} missing")

    # STEP 1: Download GRIB files if not already present
    existing_gribs = list(fhr_dir.glob("*.grib2"))
    if len(existing_gribs) < 2:
        logger.info(f"Downloading GRIB files for F{forecast_hour:02d}")
        _ = download_grib_to_forecast_dir(cycle, forecast_hour, fhr_dir, model)

    # STEP 2: Process all categories using parallel approach
    use_parallel = _use_parallel_default() and not fields

    if use_parallel:
        logger.info(f"Using parallel map processor for F{forecast_hour:02d}")
        try:
            start = time.time()
            from processor_batch import process_hrrr_parallel
            
            output_files = process_hrrr_parallel(
                cycle=cycle,
                forecast_hour=forecast_hour,
                output_dir=fhr_dir,
                categories=categories,
                model=model
            )
            
            dur = time.time() - start
            
            final_products = check_existing_products(fhr_dir)
            if final_products:
                logger.info(f"F{forecast_hour:02d} completed in {dur:.1f}s ({len(final_products)} products)")
                return {"success": True, "forecast_hour": forecast_hour, "duration": dur,
                       "product_count": len(final_products), "skipped": False}
            else:
                logger.warning(f"Parallel produced no output for F{forecast_hour:02d}; falling back to subprocess")
        except Exception as e:
            logger.error(f"Parallel path failed for F{forecast_hour:02d}: {e}; falling back to subprocess")

    # Subprocess fallback or chosen path
    try:
        start = time.time()
        res = _run_subprocess_single(cycle, forecast_hour, fhr_dir, model, categories, fields)
        dur = time.time() - start

        if res.returncode == 0:
            final_products = check_existing_products(fhr_dir)
            logger.info(f"F{forecast_hour:02d} completed in {dur:.1f}s ({len(final_products)} products)")
            return {"success": True, "forecast_hour": forecast_hour, "duration": dur, "product_count": len(final_products), "skipped": False}
        else:
            logger.error(f"F{forecast_hour:02d} failed with return code {res.returncode}")
            if res.stderr:
                logger.error(f"Stderr: {res.stderr[-500:]}")
            if res.stdout:
                logger.error(f"Stdout: {res.stdout[-500:]}")
            return {"success": False, "forecast_hour": forecast_hour, "error": f"Return code {res.returncode}"}
    except subprocess.TimeoutExpired:
        logger.error(f"F{forecast_hour:02d} timed out")
        return {"success": False, "forecast_hour": forecast_hour, "error": "Timeout"}
    except Exception as e:
        logger.error(f"F{forecast_hour:02d} crashed: {e}")
        return {"success": False, "forecast_hour": forecast_hour, "error": str(e)}


def process_model_run(
    model: str,
    date: str,
    hour: int,
    forecast_hours: List[int],
    categories: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    max_workers: int = 1,
    force_reprocess: bool = False,
    profiler=None,
):
    """Process an entire model run with smart organization"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from .utils import check_system_memory

    logger = logging.getLogger(__name__)

    output_dirs = create_output_structure(model, date, hour)
    cycle = f"{date}{hour:02d}"

    logger.info(f"Processing {model.upper()} run {date} {hour:02d}Z F{min(forecast_hours):02d}-F{max(forecast_hours):02d}")
    logger.info(f"Output directory: {output_dirs['run']}")
    logger.info(f"Categories: {categories if categories else 'all'}")
    if fields:
        logger.info(f"Fields: {fields}")

    sysmem = check_system_memory()
    if sysmem:
        logger.info(f"System memory: {sysmem['used_mb']:.0f}MB/{sysmem['total_mb']:.0f}MB ({sysmem['percent']:.1f}%)")

    cpu_count = mp.cpu_count()
    logger.info(f"Available CPUs: {cpu_count}")

    results = []

    if max_workers > 1:
        logger.info(f"Parallel over forecast hours with {max_workers} workers")
        args_list = [(cycle, fhr, output_dirs, categories, fields, force_reprocess, model) for fhr in forecast_hours]

        def worker(args):
            cyc, fhr, od, cats, flds, force, mdl = args
            return process_forecast_hour_smart(cyc, fhr, od, cats, flds, force, mdl)

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(worker, a): a[1] for a in args_list}
            for fut in as_completed(fut_map):
                fhr = fut_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logging.getLogger(__name__).error(f"F{fhr:02d} crashed in pool: {e}")
                    res = {"success": False, "forecast_hour": fhr, "error": str(e)}
                results.append(res)
    else:
        logger.info("Sequential pipeline (download+process per hour)")
        for i, fhr in enumerate(forecast_hours, 1):
            logger.info(f"Processing F{fhr:02d} ({i}/{len(forecast_hours)})")
            res = process_forecast_hour_smart(cycle, fhr, output_dirs, categories, fields, force_reprocess, model)
            results.append(res)

    successful = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r.get("skipped", False))
    failed = len(results) - successful

    logger.info(f"Complete. Total: {len(forecast_hours)}, Successful: {successful}, Skipped: {skipped}, Failed: {failed}")

    return results


def monitor_and_process_latest(
    categories: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    workers: int = 1,
    check_interval: int = 30,
    force_reprocess: bool = False,
    hour_range: Optional[List[int]] = None,
    max_hours: Optional[int] = None,
    model: str = "hrrr",
    download_threads: int = 8,
):
    """Monitor for new forecast hours and process them as they become available.

    Uses parallel GRIB downloads for speed, then processes each hour sequentially.
    """
    from .availability import get_latest_cycle, get_expected_max_forecast_hour, check_forecast_hour_availability

    logger = logging.getLogger(__name__)

    cycle, cycle_time = get_latest_cycle(model)
    if cycle is None:
        logger.error(f"No available cycles for {model}")
        return [], None, None, 0

    date_str = cycle_time.strftime("%Y%m%d")
    hour = cycle_time.hour

    output_dirs = create_output_structure(model, date_str, hour)

    expected_max_fhr = get_expected_max_forecast_hour(cycle)

    if hour_range is not None:
        forecast_hours = [h for h in hour_range if h <= expected_max_fhr]
    elif max_hours is not None:
        forecast_hours = list(range(0, min(max_hours, expected_max_fhr) + 1))
    else:
        forecast_hours = list(range(0, expected_max_fhr + 1))

    processed_hours = set()
    downloaded_hours = set()
    consecutive_no_new = 0
    max_consec = 10

    try:
        while True:
            logger.info("Checking for new forecast hours...")

            # Find all newly available hours
            newly_available = []
            for fhr in forecast_hours:
                if fhr in processed_hours:
                    continue
                available_files = check_forecast_hour_availability(cycle, fhr)
                if available_files and fhr not in downloaded_hours:
                    logger.info(f"New forecast hour: F{fhr:02d} ({', '.join(available_files)})")
                    newly_available.append(fhr)

            if newly_available:
                consecutive_no_new = 0

                # Parallel download all newly available GRIB files
                logger.info(f"Downloading GRIB files for {len(newly_available)} hours with {download_threads} threads...")
                download_start = time.time()
                download_results = download_gribs_parallel(cycle, newly_available, output_dirs, model, download_threads)
                download_dur = time.time() - download_start

                success_count = sum(1 for ok in download_results.values() if ok)
                logger.info(f"Downloads complete: {success_count}/{len(newly_available)} in {download_dur:.1f}s")

                for fhr, ok in download_results.items():
                    if ok:
                        downloaded_hours.add(fhr)

            # Process ONE downloaded hour, then loop back to check for new hours
            pending = sorted(downloaded_hours - processed_hours)
            if pending:
                fhr = pending[0]
                res = process_forecast_hour_smart(cycle, fhr, output_dirs, categories, fields, force_reprocess, model)
                if res["success"]:
                    processed_hours.add(fhr)
                else:
                    logger.error(f"F{fhr:02d} failed: {res.get('error')}")
                consecutive_no_new = 0
            elif not newly_available:
                consecutive_no_new += 1

            if len(processed_hours) >= len(forecast_hours):
                break

            if consecutive_no_new >= max_consec:
                break

            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")

    return list(processed_hours), date_str, hour, forecast_hours