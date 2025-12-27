#!/usr/bin/env python3
"""Test production-quality cross-sections with RH shading, theta contours, wind barbs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cross_section_production import (
    extract_cross_section_multi_fields,
    create_production_cross_section,
    create_cross_section_animation,
)


def main():
    # Use the newest available run: 20251224 19z
    run_dir = Path("outputs/hrrr/20251224/19z")

    if not run_dir.exists():
        # Fall back to finding any run
        outputs_dir = Path("outputs/hrrr")
        prs_files = sorted(outputs_dir.glob("**/F*/hrrr.*.wrfprs*.grib2"))
        if not prs_files:
            print("No pressure GRIB files found. Run processor first.")
            return
        grib_file = prs_files[-1]  # Most recent
        run_dir = grib_file.parent.parent
    else:
        grib_file = list(run_dir.glob("F00/*wrfprs*.grib2"))[0]

    print(f"Using run directory: {run_dir}")
    print(f"Using GRIB file: {grib_file}")

    # Extract cycle info from path
    path_parts = str(grib_file).split("/")
    cycle = "unknown"
    for i, part in enumerate(path_parts):
        if part == "hrrr" and i + 2 < len(path_parts):
            cycle = f"{path_parts[i+1]}_{path_parts[i+2].upper()}"
            break

    output_dir = Path("outputs/production_xsect")
    output_dir.mkdir(exist_ok=True)

    # Cross-section: Denver to Chicago
    print("\n--- Production Cross-Section: Denver to Chicago ---")
    start_point = (39.74, -104.99)  # Denver
    end_point = (41.88, -87.63)     # Chicago

    # Extract data
    print("Extracting multi-field data...")
    data = extract_cross_section_multi_fields(
        str(grib_file),
        start_point,
        end_point,
        n_points=120,
    )

    if data is None:
        print("Failed to extract data")
        return

    print(f"\nData extracted successfully!")
    print(f"  Pressure levels: {len(data['pressure_levels'])} ({data['pressure_levels'].min():.0f} to {data['pressure_levels'].max():.0f} hPa)")
    print(f"  Path length: {data['distances'][-1]:.0f} km")
    print(f"  RH range: {data['rh'].min():.0f} to {data['rh'].max():.0f} %")
    print(f"  Theta range: {data['theta'].min():.0f} to {data['theta'].max():.0f} K")
    if 'u_wind' in data:
        wspd = np.sqrt(data['u_wind']**2 + data['v_wind']**2) * 1.944
        print(f"  Wind speed range: {np.nanmin(wspd):.0f} to {np.nanmax(wspd):.0f} kt")

    # Create cross-section
    print("\nCreating production cross-section...")
    output_path = create_production_cross_section(
        data=data,
        cycle=cycle,
        forecast_hour=0,
        output_dir=output_dir,
        dpi=150,
    )

    if output_path:
        print(f"Created: {output_path}")
    else:
        print("Failed to create cross-section")

    # Try creating animation if multiple forecast hours available
    print("\n--- Checking for animation data ---")

    # Find all forecast hours for this run
    fhr_dirs = sorted(run_dir.glob("F*"))

    grib_files_for_anim = []
    for fhr_dir in fhr_dirs[:13]:  # First 12 hours
        prs = list(fhr_dir.glob("*wrfprs*.grib2"))
        if prs:
            fhr_str = fhr_dir.name.replace("F", "")
            try:
                fhr = int(fhr_str)
                grib_files_for_anim.append((str(prs[0]), fhr))
            except ValueError:
                pass

    if len(grib_files_for_anim) >= 3:
        print(f"Found {len(grib_files_for_anim)} forecast hours for animation")
        print("Creating animation (this may take a minute)...")

        anim_path = create_cross_section_animation(
            grib_files=grib_files_for_anim,
            start_point=start_point,
            end_point=end_point,
            cycle=cycle,
            output_dir=output_dir,
            n_points=100,
            fps=2,
        )

        if anim_path:
            print(f"Created animation: {anim_path}")
    else:
        print(f"Only {len(grib_files_for_anim)} forecast hours available, skipping animation")

    print("\n--- Done ---")


if __name__ == "__main__":
    import numpy as np
    main()
