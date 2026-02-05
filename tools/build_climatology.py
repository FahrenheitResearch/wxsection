#!/usr/bin/env python3
"""
Build 5-Year HRRR Climatology for Anomaly Mode

Computes monthly mean 3D fields from archived HRRR wrfprs GRIBs.
Stores coarsened (every 5th point) means as compressed NPZ files.

Usage:
    # Build all months from available data
    python tools/build_climatology.py --archive /mnt/d/hrrr --output /mnt/d/hrrr/climatology

    # Build specific month
    python tools/build_climatology.py --archive /mnt/d/hrrr --output /mnt/d/hrrr/climatology --month 1

    # Dry run to see data availability
    python tools/build_climatology.py --archive /mnt/d/hrrr --output /mnt/d/hrrr/climatology --dry-run

Output structure:
    {output}/climo_01_00z_F06.npz   (January, 00z init, F06)
    {output}/climo_07_12z_F00.npz   (July, 12z init, F00)
    {output}/meta.json              (build metadata)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cfgrib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

COARSEN_STEP = 5  # Every 5th grid point (~15km from 3km)

# GRIB shortName -> our field name
FIELD_MAP = {
    't': 'temperature',        # K
    'u': 'u_wind',             # m/s
    'v': 'v_wind',             # m/s
    'r': 'rh',                 # %
    'w': 'omega',              # Pa/s
    'q': 'specific_humidity',  # kg/kg
    'gh': 'geopotential_height',  # gpm
    'absv': 'vorticity',       # 1/s
}

DEFAULT_INITS = [0, 6, 12, 18]
DEFAULT_FHRS = [0, 3, 6, 9, 12, 15, 18]


def find_grib_files(archive_dir: Path, month: int, init_hour: int, fhr: int):
    """Find all wrfprs GRIB files for a given month/init/FHR across all years.

    Returns list of (Path, date_str, year) tuples sorted by date.
    """
    results = []
    filename = f"hrrr.t{init_hour:02d}z.wrfprsf{fhr:02d}.grib2"

    # Scan all date directories
    for date_dir in sorted(archive_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        if len(date_str) != 8 or not date_str.isdigit():
            continue

        # Check month matches
        try:
            dt = datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            continue

        if dt.month != month:
            continue

        grib_path = date_dir / f"{init_hour:02d}z" / f"F{fhr:02d}" / filename
        if grib_path.exists():
            results.append((grib_path, date_str, dt.year))

    return results


def load_and_coarsen(grib_path: Path, step: int = COARSEN_STEP):
    """Load climatology fields from a wrfprs GRIB, coarsen by step.

    Returns dict with coarsened arrays + coordinate info, or None on failure.
    """
    try:
        datasets = cfgrib.open_datasets(str(grib_path))
    except Exception as e:
        logger.warning(f"  cfgrib failed on {grib_path.name}: {e}")
        return None

    result = {}
    pressure_levels = None
    lats = None
    lons = None

    for ds in datasets:
        # Look for isobaric datasets
        level_dim = None
        for dim in ('isobaricInhPa', 'level'):
            if dim in ds.dims:
                level_dim = dim
                break

        if level_dim is None:
            continue

        levels = ds[level_dim].values
        if len(levels) < 20:
            continue  # Skip datasets with few levels (not the main one)

        if pressure_levels is None:
            pressure_levels = levels.astype(np.float32)

        # Get lat/lon
        if lats is None:
            if 'latitude' in ds.coords:
                lat_arr = ds['latitude'].values
                lon_arr = ds['longitude'].values
                if lat_arr.ndim == 2:
                    lats = lat_arr[::step, ::step].astype(np.float32)
                    lons = lon_arr[::step, ::step].astype(np.float32)
                else:
                    lats = lat_arr[::step].astype(np.float32)
                    lons = lon_arr[::step].astype(np.float32)

        # Extract fields
        for short_name, our_name in FIELD_MAP.items():
            if our_name in result:
                continue
            if short_name in ds.data_vars:
                data = ds[short_name].values  # (n_levels, ny, nx)
                if data.ndim == 3:
                    result[our_name] = data[:, ::step, ::step].astype(np.float32)
                elif data.ndim == 2:
                    result[our_name] = data[::step, ::step].astype(np.float32)

    if pressure_levels is None or lats is None:
        return None

    result['pressure_levels'] = pressure_levels
    result['lats'] = lats
    result['lons'] = lons
    return result


def accumulate_mean(running_sum, running_count, new_data):
    """Accumulate running sum and count for incremental mean computation."""
    for field_name in FIELD_MAP.values():
        if field_name not in new_data or new_data[field_name] is None:
            continue

        arr = new_data[field_name].astype(np.float64)

        if field_name not in running_sum:
            running_sum[field_name] = np.zeros_like(arr, dtype=np.float64)
            running_count[field_name] = np.zeros(arr.shape, dtype=np.int32)

        mask = np.isfinite(arr)
        running_sum[field_name][mask] += arr[mask]
        running_count[field_name][mask] += 1


def finalize_mean(running_sum, running_count):
    """Convert accumulated sums to means."""
    means = {}
    for field_name in running_sum:
        with np.errstate(invalid='ignore'):
            means[field_name] = np.where(
                running_count[field_name] > 0,
                running_sum[field_name] / running_count[field_name],
                np.nan
            ).astype(np.float32)
    return means


def save_climatology(output_dir, month, init_hour, fhr, means,
                     lats, lons, pressure_levels, n_samples, years):
    """Save climatology to compressed NPZ."""
    filename = f"climo_{month:02d}_{init_hour:02d}z_F{fhr:02d}.npz"
    data = {
        'lats': lats,
        'lons': lons,
        'pressure_levels': pressure_levels,
        'n_samples': np.array([n_samples]),
        'years': np.array(sorted(years)),
    }
    data.update(means)
    np.savez_compressed(output_dir / filename, **data)
    size_mb = (output_dir / filename).stat().st_size / 1e6
    logger.info(f"  Saved {filename} ({n_samples} samples, {size_mb:.1f} MB)")


def build_combination(archive_dir, output_dir, month, init_hour, fhr,
                      min_samples=3, force=False):
    """Build climatology for a single (month, init, fhr) combination.

    Returns (n_samples, success) tuple.
    """
    filename = f"climo_{month:02d}_{init_hour:02d}z_F{fhr:02d}.npz"
    out_path = output_dir / filename

    # Find available GRIBs
    grib_files = find_grib_files(archive_dir, month, init_hour, fhr)

    if len(grib_files) < min_samples:
        return len(grib_files), False

    # Check if output already exists with same or more samples
    if out_path.exists() and not force:
        try:
            existing = np.load(out_path)
            existing_n = int(existing['n_samples'][0])
            if existing_n >= len(grib_files):
                logger.info(f"  {filename} already up to date ({existing_n} samples)")
                return existing_n, True
        except Exception:
            pass  # Corrupted, rebuild

    # Accumulate means
    running_sum = {}
    running_count = {}
    n_loaded = 0
    lats = lons = pressure_levels = None
    years_seen = set()

    for grib_path, date_str, year in grib_files:
        try:
            data = load_and_coarsen(grib_path)
            if data is None:
                continue

            if lats is None:
                lats = data['lats']
                lons = data['lons']
                pressure_levels = data['pressure_levels']

            accumulate_mean(running_sum, running_count, data)
            n_loaded += 1
            years_seen.add(year)

            if n_loaded % 50 == 0:
                logger.info(f"    Processed {n_loaded}/{len(grib_files)} GRIBs...")

        except Exception as e:
            logger.warning(f"  Failed: {grib_path.name}: {e}")

    if n_loaded < min_samples:
        logger.warning(f"  Only loaded {n_loaded}/{len(grib_files)} — below min_samples={min_samples}")
        return n_loaded, False

    means = finalize_mean(running_sum, running_count)
    save_climatology(output_dir, month, init_hour, fhr, means,
                     lats, lons, pressure_levels, n_loaded, years_seen)
    return n_loaded, True


def save_meta(output_dir, build_info):
    """Save build metadata to meta.json."""
    meta_path = output_dir / 'meta.json'
    meta = {
        'built': datetime.now().isoformat(),
        'coarsen_step': COARSEN_STEP,
        'fields': list(FIELD_MAP.values()),
        'combinations': build_info,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Build HRRR climatology for anomaly mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--archive', required=True,
                        help='Root of HRRR archive (e.g., /mnt/d/hrrr)')
    parser.add_argument('--output', required=True,
                        help='Output directory for climatology NPZ files')
    parser.add_argument('--month', type=int, default=None,
                        help='Build only this month (1-12). Default: all')
    parser.add_argument('--inits', type=int, nargs='+', default=DEFAULT_INITS,
                        help=f'Init hours (default: {DEFAULT_INITS})')
    parser.add_argument('--fhrs', type=int, nargs='+', default=DEFAULT_FHRS,
                        help=f'Forecast hours (default: {DEFAULT_FHRS})')
    parser.add_argument('--min-samples', type=int, default=3,
                        help='Minimum samples to produce a file (default: 3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show data availability without building')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild even if output exists')
    args = parser.parse_args()

    archive_dir = Path(args.archive)
    output_dir = Path(args.output)

    if not archive_dir.exists():
        print(f"Error: archive directory not found: {archive_dir}")
        sys.exit(1)

    months = [args.month] if args.month else list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    total_combos = len(months) * len(args.inits) * len(args.fhrs)

    print("=" * 60)
    print("  HRRR Climatology Builder")
    print("=" * 60)
    print(f"  Archive:     {archive_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Months:      {', '.join(month_names[m-1] for m in months)}")
    print(f"  Inits:       {', '.join(f'{h:02d}z' for h in args.inits)}")
    print(f"  FHRs:        {', '.join(f'F{f:02d}' for f in args.fhrs)}")
    print(f"  Coarsening:  every {COARSEN_STEP}th point")
    print(f"  Min samples: {args.min_samples}")
    print(f"  Total combos: {total_combos}")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run — data availability:\n")
        for month in months:
            print(f"  {month_names[month-1]}:")
            for init_hour in args.inits:
                counts = []
                for fhr in args.fhrs:
                    grib_files = find_grib_files(archive_dir, month, init_hour, fhr)
                    years = sorted(set(y for _, _, y in grib_files))
                    counts.append(len(grib_files))
                count_str = '/'.join(str(c) for c in counts)
                year_range = f"{min(years)}-{max(years)}" if years else "none"
                ready = "ready" if min(counts) >= args.min_samples else "insufficient"
                print(f"    {init_hour:02d}z: {count_str} samples ({year_range}) — {ready}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()
    build_info = []
    built = 0
    skipped = 0
    failed = 0

    for month in months:
        for init_hour in args.inits:
            for fhr in args.fhrs:
                combo = f"{month_names[month-1]} {init_hour:02d}z F{fhr:02d}"
                logger.info(f"[{built+skipped+failed+1}/{total_combos}] {combo}...")

                n_samples, ok = build_combination(
                    archive_dir, output_dir, month, init_hour, fhr,
                    min_samples=args.min_samples, force=args.force
                )

                build_info.append({
                    'month': month,
                    'init': init_hour,
                    'fhr': fhr,
                    'n_samples': n_samples,
                    'success': ok,
                })

                if ok:
                    built += 1
                elif n_samples > 0:
                    skipped += 1
                    logger.info(f"  Skipped (only {n_samples} samples)")
                else:
                    failed += 1
                    logger.info(f"  No data found")

    save_meta(output_dir, build_info)

    elapsed = time.time() - overall_start
    print()
    print("=" * 60)
    print("  Climatology Build Complete")
    print("=" * 60)
    print(f"  Time:     {elapsed/60:.1f} minutes")
    print(f"  Built:    {built}")
    print(f"  Skipped:  {skipped} (insufficient samples)")
    print(f"  No data:  {failed}")
    print(f"  Output:   {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
