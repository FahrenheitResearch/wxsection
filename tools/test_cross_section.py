#!/usr/bin/env python3
"""Test cross-section generation with real HRRR data and terrain masking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cross_section import (
    create_cross_section,
    extract_cross_section_data,
    extract_surface_pressure,
    create_cross_section_multi,
)


def main():
    # Find a GRIB file to test with
    outputs_dir = Path("outputs/hrrr")
    grib_files = list(outputs_dir.glob("**/F00/*wrfprs*.grib2"))

    if not grib_files:
        print("No pressure GRIB files found. Run processor first.")
        return

    grib_file = grib_files[0]
    print(f"Using GRIB file: {grib_file}")

    output_dir = Path("outputs/interactive_test")
    output_dir.mkdir(exist_ok=True)

    # Cross-section: Denver to Chicago (good range across midwest)
    start_point = (39.74, -104.99)  # Denver
    end_point = (41.88, -87.63)     # Chicago

    print(f"\nCross-section from Denver {start_point} to Chicago {end_point}")

    # Test 1: Temperature cross-section (weathernerds.org style - turbo colormap)
    print("\n--- Temperature Cross-Section ---")
    field_config = {
        'grib_key': 't',
        'typeOfLevel': 'isobaricInhPa',
        'title': 'Temperature',
        'units': '°F',
        'convert': 'K_to_F',
        'colormap': 'turbo',  # Like weathernerds style
        'vmin': -80,
        'vmax': 100,
    }

    result = extract_cross_section_data(
        str(grib_file),
        'temperature',
        field_config,
        start_point,
        end_point,
        n_points=100,
    )

    if result:
        data_3d, pressure_levels, lats, lons = result
        print(f"Data shape: {data_3d.shape}")
        print(f"Pressure levels: {pressure_levels}")
        print(f"Data range: {data_3d.min():.1f} to {data_3d.max():.1f}")

        # Extract surface pressure for terrain masking
        print("Extracting surface pressure for terrain masking...")
        surface_pressure = extract_surface_pressure(str(grib_file), lats, lons)

        output_path = create_cross_section(
            data_3d=data_3d,
            pressure_levels=pressure_levels,
            lats=lats,
            lons=lons,
            field_name='temperature',
            field_config=field_config,
            cycle='2025122600',
            forecast_hour=0,
            output_dir=output_dir,
            colormap='turbo',
            surface_pressure=surface_pressure,  # Enable terrain masking
        )

        if output_path:
            print(f"Created: {output_path}")
    else:
        print("Failed to extract temperature data")

    # Test 2: Wind speed cross-section (different path: LA to Dallas)
    print("\n--- Wind Speed Cross-Section (LA to Dallas) ---")
    start_point2 = (34.05, -118.24)  # LA
    end_point2 = (32.78, -96.80)     # Dallas

    field_config_wind = {
        'grib_key': 'ws',  # Try wind speed first
        'typeOfLevel': 'isobaricInhPa',
        'title': 'Wind Speed',
        'units': 'kt',
        'colormap': 'turbo',
        'vmin': 0,
        'vmax': 100,
    }

    # Try to get wind speed, or compute from u/v
    result = extract_cross_section_data(
        str(grib_file),
        'wind_speed',
        field_config_wind,
        start_point2,
        end_point2,
        n_points=100,
    )

    if result:
        data_3d, pressure_levels, lats, lons = result
        # Convert m/s to knots
        data_3d = data_3d * 1.944

        print(f"Wind data shape: {data_3d.shape}")
        print(f"Wind range: {data_3d.min():.1f} to {data_3d.max():.1f} kt")

        output_path = create_cross_section(
            data_3d=data_3d,
            pressure_levels=pressure_levels,
            lats=lats,
            lons=lons,
            field_name='wind_speed',
            field_config=field_config_wind,
            cycle='2025122600',
            forecast_hour=0,
            output_dir=output_dir,
            colormap='turbo',
        )

        if output_path:
            print(f"Created: {output_path}")
    else:
        print("Wind speed not directly available, trying u/v components...")

        # Try getting u and v components separately
        u_config = {
            'grib_key': 'u',
            'typeOfLevel': 'isobaricInhPa',
            'title': 'U Wind',
            'units': 'm/s',
        }
        v_config = {
            'grib_key': 'v',
            'typeOfLevel': 'isobaricInhPa',
            'title': 'V Wind',
            'units': 'm/s',
        }

        u_result = extract_cross_section_data(
            str(grib_file), 'u', u_config, start_point2, end_point2, n_points=100
        )
        v_result = extract_cross_section_data(
            str(grib_file), 'v', v_config, start_point2, end_point2, n_points=100
        )

        if u_result and v_result:
            u_data, pressure_levels, lats, lons = u_result
            v_data, _, _, _ = v_result
            wspd = np.sqrt(u_data**2 + v_data**2) * 1.944  # to knots

            print(f"Computed wind speed range: {wspd.min():.1f} to {wspd.max():.1f} kt")

            # Get surface pressure for terrain masking
            surface_pressure = extract_surface_pressure(str(grib_file), lats, lons)

            output_path = create_cross_section(
                data_3d=wspd,
                pressure_levels=pressure_levels,
                lats=lats,
                lons=lons,
                field_name='wind_speed',
                field_config=field_config_wind,
                cycle='2025122600',
                forecast_hour=0,
                output_dir=output_dir,
                colormap='turbo',
                surface_pressure=surface_pressure,
            )
            if output_path:
                print(f"Created: {output_path}")

    # Test 3: Relative Humidity cross-section
    print("\n--- Relative Humidity Cross-Section (Tornado Alley) ---")
    start_point3 = (35.0, -100.0)   # Texas panhandle
    end_point3 = (42.0, -95.0)      # Nebraska

    rh_config = {
        'grib_key': 'r',  # relative humidity
        'typeOfLevel': 'isobaricInhPa',
        'title': 'Relative Humidity',
        'units': '%',
        'colormap': 'YlGnBu',
        'vmin': 0,
        'vmax': 100,
    }

    result = extract_cross_section_data(
        str(grib_file),
        'rh',
        rh_config,
        start_point3,
        end_point3,
        n_points=100,
    )

    if result:
        data_3d, pressure_levels, lats, lons = result
        print(f"RH data range: {data_3d.min():.1f} to {data_3d.max():.1f} %")

        # Get surface pressure for terrain masking
        surface_pressure = extract_surface_pressure(str(grib_file), lats, lons)

        output_path = create_cross_section(
            data_3d=data_3d,
            pressure_levels=pressure_levels,
            lats=lats,
            lons=lons,
            field_name='rh',
            field_config=rh_config,
            cycle='2025122600',
            forecast_hour=0,
            output_dir=output_dir,
            colormap='YlGnBu',
            surface_pressure=surface_pressure,
        )
        if output_path:
            print(f"Created: {output_path}")
    else:
        print("RH not found")

    print("\n--- Done! Check outputs/interactive_test/ ---")


if __name__ == "__main__":
    import numpy as np
    main()
