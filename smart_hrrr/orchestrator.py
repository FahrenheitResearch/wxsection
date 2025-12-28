"""
Simplified GRIB Download Orchestrator

Handles parallel downloading of HRRR GRIB files for cross-section processing.
"""

import time
import logging
import urllib.request
import urllib.error
import socket
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_config import get_model_registry
from .io import create_output_structure, get_forecast_hour_dir

logger = logging.getLogger(__name__)


def download_grib_file(url: str, output_path: Path, timeout: int = 600) -> bool:
    """Download a single GRIB file from URL."""
    try:
        socket.setdefaulttimeout(timeout)
        urllib.request.urlretrieve(url, output_path)
        return True
    except (urllib.error.URLError, socket.timeout) as e:
        logger.debug(f"Failed to download from {url}: {e}")
        return False


def download_forecast_hour(
    model: str,
    date_str: str,
    cycle_hour: int,
    forecast_hour: int,
    output_dir: Path,
    file_types: List[str] = None
) -> bool:
    """Download GRIB files for a single forecast hour."""

    if file_types is None:
        file_types = ['pressure', 'surface']  # wrfprs and wrfsfc

    registry = get_model_registry()
    model_config = registry.get_model(model)

    if not model_config:
        logger.error(f"Unknown model: {model}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    success = False

    for file_type in file_types:
        filename = model_config.get_filename(cycle_hour, file_type, forecast_hour)
        output_path = output_dir / filename

        if output_path.exists():
            logger.debug(f"File exists: {filename}")
            success = True
            continue

        urls = model_config.get_download_urls(date_str, cycle_hour, file_type, forecast_hour)

        for i, url in enumerate(urls):
            source = "NOMADS" if "nomads" in url else "AWS" if "s3.amazonaws" in url or "noaa-" in url else "Pando"
            logger.info(f"Downloading {filename} from {source}...")

            if download_grib_file(url, output_path):
                logger.info(f"Downloaded {filename}")
                success = True
                break
            else:
                if i < len(urls) - 1:
                    logger.warning(f"{source} failed, trying next source...")
                else:
                    logger.error(f"Failed to download {filename} from all sources")

    return success


def download_gribs_parallel(
    model: str,
    date_str: str,
    cycle_hour: int,
    forecast_hours: List[int],
    output_base_dir: Path = None,
    max_threads: int = 8,
    file_types: List[str] = None
) -> Dict[int, bool]:
    """Download GRIB files for multiple forecast hours in parallel.

    Returns dict mapping forecast_hour -> success status.
    """

    if output_base_dir is None:
        output_dirs = create_output_structure(model, date_str, cycle_hour)
        output_base_dir = output_dirs['run']

    def download_single(fhr: int) -> tuple:
        fhr_dir = get_forecast_hour_dir(output_base_dir, fhr)
        start = time.time()
        ok = download_forecast_hour(model, date_str, cycle_hour, fhr, fhr_dir, file_types)
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


def download_latest_cycle(
    model: str = 'hrrr',
    max_hours: int = 18,
    max_threads: int = 8,
    forecast_hours: List[int] = None
) -> tuple:
    """Download the latest available model cycle.

    Args:
        model: Model name (default 'hrrr')
        max_hours: Maximum forecast hour to download
        max_threads: Number of parallel download threads
        forecast_hours: Specific forecast hours to download (e.g., [0, 6, 12, 18]).
                       If None, downloads all hours from 0 to max_hours.

    Returns (date_str, cycle_hour, results_dict) or (None, None, {}) if failed.
    """
    from .availability import get_latest_cycle

    cycle, cycle_time = get_latest_cycle(model)
    if cycle is None:
        logger.error(f"No available cycles for {model}")
        return None, None, {}

    date_str = cycle_time.strftime("%Y%m%d")
    cycle_hour = cycle_time.hour

    # Determine forecast hours based on cycle type
    registry = get_model_registry()
    model_config = registry.get_model(model)
    max_fhr = model_config.get_max_forecast_hour(cycle_hour) if model_config else 18
    max_fhr = min(max_fhr, max_hours)

    if forecast_hours is not None:
        # Use specific forecast hours, filtered by what's available
        fhrs_to_download = [f for f in forecast_hours if f <= max_fhr]
    else:
        # Download all hours
        fhrs_to_download = list(range(max_fhr + 1))

    fhr_str = ','.join(f'F{f:02d}' for f in fhrs_to_download)
    logger.info(f"Downloading {model.upper()} {date_str} {cycle_hour:02d}Z [{fhr_str}]")

    results = download_gribs_parallel(
        model=model,
        date_str=date_str,
        cycle_hour=cycle_hour,
        forecast_hours=fhrs_to_download,
        max_threads=max_threads
    )

    success_count = sum(1 for ok in results.values() if ok)
    logger.info(f"Downloaded {success_count}/{len(fhrs_to_download)} forecast hours")

    return date_str, cycle_hour, results
