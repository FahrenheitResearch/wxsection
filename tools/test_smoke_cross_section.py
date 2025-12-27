#!/usr/bin/env python3
"""Test smoke cross-section generation for HRRR smoke data.

Since HRRR smoke (MASSDEN) is only available at surface heights (1m, 2m, 8m AGL),
not on pressure levels, this creates a horizontal cross-section showing surface
smoke concentration along a path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cross_section import (
    extract_surface_smoke_cross_section,
    create_smoke_cross_section,
)


def main():
    # Use the HRRR run with smoke impact in North Texas/Oklahoma
    # User mentioned: outputs/hrrr/20250826/16z
    outputs_dir = Path("outputs/hrrr/20250826/16z")

    if not outputs_dir.exists():
        # Fall back to any available run
        outputs_dir = Path("outputs/hrrr")
        run_dirs = sorted(outputs_dir.glob("*/*/F00"))
        if not run_dirs:
            print("No HRRR output directories found. Run processor first.")
            return
        outputs_dir = run_dirs[-1]  # Use most recent
        print(f"Using fallback directory: {outputs_dir}")
    else:
        outputs_dir = outputs_dir / "F00"

    # Find GRIB files - smoke is in wrfsfc file
    sfc_files = list(outputs_dir.glob("*wrfsfc*.grib2"))
    prs_files = list(outputs_dir.glob("*wrfprs*.grib2"))

    if not sfc_files and not prs_files:
        print(f"No GRIB files found in {outputs_dir}")
        return

    # Prefer sfc file for smoke, but try prs as backup
    grib_file = sfc_files[0] if sfc_files else prs_files[0]
    print(f"Using GRIB file: {grib_file}")

    output_dir = Path("outputs/interactive_test")
    output_dir.mkdir(exist_ok=True)

    # Cross-section path: Through Oregon smoke region (actual max smoke location)
    # The max smoke is at 43.49°N, -122.33°W (southern Oregon wildfires)
    start_point = (41.5, -125.0)   # Northern California coast
    end_point = (45.5, -120.0)     # Eastern Oregon

    print(f"\nSmoke Cross-Section through Oregon fires: {start_point} to {end_point}")

    # Extract smoke data
    result = extract_surface_smoke_cross_section(
        str(grib_file),
        start_point,
        end_point,
        n_points=150,
    )

    if result is None:
        print("Failed to extract smoke data")
        print("\nTrying alternative path: NM fires to Kansas...")

        # Alternative path: Through known smoke regions
        start_point = (33.0, -106.0)  # Southern NM (fire region)
        end_point = (37.0, -98.0)     # Kansas

        result = extract_surface_smoke_cross_section(
            str(grib_file),
            start_point,
            end_point,
            n_points=150,
        )

    if result is not None:
        smoke_data, heights, lats, lons, distances = result

        print(f"\nSmoke data extracted successfully!")
        print(f"  Heights: {heights} m")
        print(f"  Path length: {distances[-1]:.0f} km")
        print(f"  Data shape: {smoke_data.shape}")
        print(f"  Smoke range: {smoke_data.min():.2f} to {smoke_data.max():.2f} µg/m³")

        # Get cycle info from path
        cycle = "20250826_16Z"  # Default
        path_parts = str(grib_file).split("/")
        for i, part in enumerate(path_parts):
            if part == "hrrr" and i + 2 < len(path_parts):
                cycle = f"{path_parts[i+1]}_{path_parts[i+2].upper()}"
                break

        # Create the cross-section plot
        output_path = create_smoke_cross_section(
            smoke_data=smoke_data,
            heights=heights,
            lats=lats,
            lons=lons,
            distances=distances,
            cycle=cycle,
            forecast_hour=0,
            output_dir=output_dir,
        )

        if output_path:
            print(f"\nCreated: {output_path}")
            print(f"Open in browser: file://{output_path.absolute()}")
        else:
            print("Failed to create smoke cross-section plot")
    else:
        print("\nCould not extract smoke data from this GRIB file.")
        print("Smoke data may not be present in this particular forecast.")

    # Also try Texas/Oklahoma path (user's original request)
    print("\n\n--- Texas/Oklahoma Cross-Section ---")
    start_point_tx = (35.0, -102.0)   # Texas Panhandle
    end_point_tx = (35.5, -97.5)      # Oklahoma City

    result_tx = extract_surface_smoke_cross_section(
        str(grib_file),
        start_point_tx,
        end_point_tx,
        n_points=150,
    )

    if result_tx is not None:
        smoke_data, heights, lats, lons, distances = result_tx
        print(f"TX/OK smoke range: {smoke_data.min():.2f} to {smoke_data.max():.2f} µg/m³")

        output_path = create_smoke_cross_section(
            smoke_data=smoke_data,
            heights=heights,
            lats=lats,
            lons=lons,
            distances=distances,
            cycle=cycle,
            forecast_hour=0,
            output_dir=output_dir / "texas_oklahoma",
        )
        if output_path:
            print(f"Created: {output_path}")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
