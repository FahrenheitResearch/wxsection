# derived_params/diurnal_temperature.py
"""
Diurnal Temperature Analysis Functions

Comprehensive suite for computing diurnal temperature variations from HRRR model data.
Designed to work with multiple forecast hours from a single model run.

Diurnal Products:
- Diurnal Temperature Range (DTR): Max - Min over time window
- Maximum Temperature (T_max): Highest temperature in period
- Minimum Temperature (T_min): Lowest temperature in period
- Day-Night Difference: Afternoon temp minus overnight temp
- Temperature Departure from Mean: How far each hour deviates from period mean
- Rate of Heating: Morning temperature rise rate (°C/hr)
- Rate of Cooling: Evening temperature fall rate (°C/hr)

Author: HRRR Maps Pipeline
Version: 1.0
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from .common import _dbg


def diurnal_temperature_range(temp_arrays: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Compute Diurnal Temperature Range (DTR) = T_max - T_min

    The DTR is a fundamental measure of daily temperature variability.
    High DTR indicates clear/dry conditions, low DTR indicates cloudy/humid conditions.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)
                    Should contain at least 12-24 hours of data

    Returns:
        2D array of diurnal temperature range in °C

    Typical values:
        - Humid/cloudy: 5-10°C
        - Moderate: 10-15°C
        - Arid/clear: 15-25°C
        - Desert: 20-35°C
    """
    if not temp_arrays:
        raise ValueError("No temperature data provided")

    # Stack all temperature arrays
    temps = np.stack(list(temp_arrays.values()), axis=0)

    # Compute max and min along time axis
    t_max = np.nanmax(temps, axis=0)
    t_min = np.nanmin(temps, axis=0)

    dtr = t_max - t_min

    _dbg(f"DTR: range {np.nanmin(dtr):.1f} to {np.nanmax(dtr):.1f}°C")

    return dtr.astype(np.float32)


def diurnal_max_temperature(temp_arrays: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Extract maximum temperature over the forecast period.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)

    Returns:
        2D array of maximum temperatures in °C
    """
    if not temp_arrays:
        raise ValueError("No temperature data provided")

    temps = np.stack(list(temp_arrays.values()), axis=0)
    t_max = np.nanmax(temps, axis=0)

    _dbg(f"T_max: range {np.nanmin(t_max):.1f} to {np.nanmax(t_max):.1f}°C")

    return t_max.astype(np.float32)


def diurnal_min_temperature(temp_arrays: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Extract minimum temperature over the forecast period.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)

    Returns:
        2D array of minimum temperatures in °C
    """
    if not temp_arrays:
        raise ValueError("No temperature data provided")

    temps = np.stack(list(temp_arrays.values()), axis=0)
    t_min = np.nanmin(temps, axis=0)

    _dbg(f"T_min: range {np.nanmin(t_min):.1f} to {np.nanmax(t_min):.1f}°C")

    return t_min.astype(np.float32)


def diurnal_mean_temperature(temp_arrays: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Compute mean temperature over the forecast period.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)

    Returns:
        2D array of mean temperatures in °C
    """
    if not temp_arrays:
        raise ValueError("No temperature data provided")

    temps = np.stack(list(temp_arrays.values()), axis=0)
    t_mean = np.nanmean(temps, axis=0)

    _dbg(f"T_mean: range {np.nanmin(t_mean):.1f} to {np.nanmax(t_mean):.1f}°C")

    return t_mean.astype(np.float32)


def day_night_temperature_difference(
    temp_arrays: Dict[int, np.ndarray],
    cycle_hour: int,
    day_offset: int = 18,  # ~2pm local for 12Z run
    night_offset: int = 6   # ~12am local for 12Z run
) -> np.ndarray:
    """
    Compute day-night temperature difference (afternoon - overnight).

    This metric highlights the amplitude of the diurnal cycle.
    Positive values (normal) indicate warmer afternoons than nights.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)
        cycle_hour: Model initialization hour (UTC)
        day_offset: Forecast hour offset for daytime reading (default 18h for ~2pm local)
        night_offset: Forecast hour offset for nighttime reading (default 6h for ~midnight local)

    Returns:
        2D array of day-night temperature difference in °C
    """
    fhrs = sorted(temp_arrays.keys())

    # Find closest available forecast hours
    day_fhr = min(fhrs, key=lambda x: abs(x - day_offset))
    night_fhr = min(fhrs, key=lambda x: abs(x - night_offset))

    if day_fhr not in temp_arrays or night_fhr not in temp_arrays:
        raise ValueError(f"Required forecast hours not available. Need ~{day_offset}h and ~{night_offset}h")

    diff = temp_arrays[day_fhr] - temp_arrays[night_fhr]

    _dbg(f"Day-Night diff (f{day_fhr}-f{night_fhr}): {np.nanmin(diff):.1f} to {np.nanmax(diff):.1f}°C")

    return diff.astype(np.float32)


def temperature_departure_from_mean(
    temp_arrays: Dict[int, np.ndarray],
    target_hour: int
) -> np.ndarray:
    """
    Compute temperature departure from the period mean for a specific hour.

    Useful for identifying anomalously warm or cool periods.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)
        target_hour: Forecast hour to compute departure for

    Returns:
        2D array of temperature departure in °C (positive = warmer than mean)
    """
    if target_hour not in temp_arrays:
        raise ValueError(f"Forecast hour {target_hour} not in data")

    t_mean = diurnal_mean_temperature(temp_arrays)
    departure = temp_arrays[target_hour] - t_mean

    _dbg(f"Departure f{target_hour}: {np.nanmin(departure):.1f} to {np.nanmax(departure):.1f}°C")

    return departure.astype(np.float32)


def heating_rate(
    temp_arrays: Dict[int, np.ndarray],
    start_hour: int = 6,
    end_hour: int = 12
) -> np.ndarray:
    """
    Compute morning heating rate (°C per hour).

    High heating rates indicate strong insolation and low thermal inertia.
    Important for fire weather and convective initiation timing.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)
        start_hour: Starting forecast hour (default 6, ~sunrise)
        end_hour: Ending forecast hour (default 12, ~midday)

    Returns:
        2D array of heating rate in °C/hr
    """
    fhrs = sorted(temp_arrays.keys())

    # Find closest available hours
    actual_start = min(fhrs, key=lambda x: abs(x - start_hour))
    actual_end = min(fhrs, key=lambda x: abs(x - end_hour))

    if actual_start >= actual_end:
        raise ValueError("Start hour must be before end hour")

    delta_t = temp_arrays[actual_end] - temp_arrays[actual_start]
    delta_hr = actual_end - actual_start

    rate = delta_t / delta_hr

    _dbg(f"Heating rate (f{actual_start}-f{actual_end}): {np.nanmin(rate):.2f} to {np.nanmax(rate):.2f}°C/hr")

    return rate.astype(np.float32)


def cooling_rate(
    temp_arrays: Dict[int, np.ndarray],
    start_hour: int = 18,
    end_hour: int = 24
) -> np.ndarray:
    """
    Compute evening cooling rate (°C per hour, positive = cooling).

    High cooling rates indicate clear skies and dry air.
    Important for fog/frost prediction and overnight lows.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)
        start_hour: Starting forecast hour (default 18, ~evening)
        end_hour: Ending forecast hour (default 24, ~midnight)

    Returns:
        2D array of cooling rate in °C/hr (positive values indicate cooling)
    """
    fhrs = sorted(temp_arrays.keys())

    # Find closest available hours
    actual_start = min(fhrs, key=lambda x: abs(x - start_hour))
    actual_end = min(fhrs, key=lambda x: abs(x - end_hour))

    if actual_start >= actual_end:
        raise ValueError("Start hour must be before end hour")

    # Cooling is positive when temp decreases
    delta_t = temp_arrays[actual_start] - temp_arrays[actual_end]
    delta_hr = actual_end - actual_start

    rate = delta_t / delta_hr

    _dbg(f"Cooling rate (f{actual_start}-f{actual_end}): {np.nanmin(rate):.2f} to {np.nanmax(rate):.2f}°C/hr")

    return rate.astype(np.float32)


def hour_of_maximum_temperature(temp_arrays: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Find the forecast hour when maximum temperature occurs.

    Useful for identifying timing of peak heating - earlier peaks
    may indicate dry/unstable conditions, later peaks suggest moisture.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)

    Returns:
        2D array of forecast hours when T_max occurs
    """
    fhrs = sorted(temp_arrays.keys())
    temps = np.stack([temp_arrays[fhr] for fhr in fhrs], axis=0)

    # Find index of max along time axis
    max_idx = np.nanargmax(temps, axis=0)

    # Convert index to forecast hour
    fhr_array = np.array(fhrs)
    hour_of_max = fhr_array[max_idx]

    _dbg(f"Hour of T_max: {np.min(hour_of_max)} to {np.max(hour_of_max)}")

    return hour_of_max.astype(np.float32)


def hour_of_minimum_temperature(temp_arrays: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Find the forecast hour when minimum temperature occurs.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)

    Returns:
        2D array of forecast hours when T_min occurs
    """
    fhrs = sorted(temp_arrays.keys())
    temps = np.stack([temp_arrays[fhr] for fhr in fhrs], axis=0)

    # Find index of min along time axis
    min_idx = np.nanargmin(temps, axis=0)

    # Convert index to forecast hour
    fhr_array = np.array(fhrs)
    hour_of_min = fhr_array[min_idx]

    _dbg(f"Hour of T_min: {np.min(hour_of_min)} to {np.max(hour_of_min)}")

    return hour_of_min.astype(np.float32)


def diurnal_temperature_amplitude(temp_arrays: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Compute diurnal temperature amplitude (half the range).

    Amplitude = (T_max - T_min) / 2

    This is useful for harmonic analysis and comparing to climatology.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)

    Returns:
        2D array of diurnal amplitude in °C
    """
    dtr = diurnal_temperature_range(temp_arrays)
    amplitude = dtr / 2.0

    _dbg(f"Amplitude: {np.nanmin(amplitude):.1f} to {np.nanmax(amplitude):.1f}°C")

    return amplitude.astype(np.float32)


def temperature_at_hour(
    temp_arrays: Dict[int, np.ndarray],
    target_hour: int
) -> np.ndarray:
    """
    Extract temperature at a specific forecast hour.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)
        target_hour: Forecast hour to extract

    Returns:
        2D array of temperature in °C
    """
    if target_hour not in temp_arrays:
        # Find closest available hour
        fhrs = list(temp_arrays.keys())
        target_hour = min(fhrs, key=lambda x: abs(x - target_hour))
        _dbg(f"Using closest available hour: f{target_hour}")

    return temp_arrays[target_hour].astype(np.float32)


def compute_all_diurnal_products(
    temp_arrays: Dict[int, np.ndarray],
    cycle_hour: int = 12
) -> Dict[str, np.ndarray]:
    """
    Compute all diurnal temperature products at once.

    Convenience function that returns a dictionary of all products.

    Args:
        temp_arrays: Dictionary mapping forecast hour -> 2m temperature array (°C)
        cycle_hour: Model initialization hour (UTC)

    Returns:
        Dictionary of product_name -> 2D numpy array
    """
    products = {}

    # Core products
    products['dtr'] = diurnal_temperature_range(temp_arrays)
    products['t_max'] = diurnal_max_temperature(temp_arrays)
    products['t_min'] = diurnal_min_temperature(temp_arrays)
    products['t_mean'] = diurnal_mean_temperature(temp_arrays)
    products['amplitude'] = diurnal_temperature_amplitude(temp_arrays)

    # Timing products
    products['hour_of_max'] = hour_of_maximum_temperature(temp_arrays)
    products['hour_of_min'] = hour_of_minimum_temperature(temp_arrays)

    # Rate products (try, may fail if hours not available)
    try:
        products['heating_rate'] = heating_rate(temp_arrays)
    except Exception as e:
        _dbg(f"Could not compute heating rate: {e}")

    try:
        products['cooling_rate'] = cooling_rate(temp_arrays)
    except Exception as e:
        _dbg(f"Could not compute cooling rate: {e}")

    # Day-night difference
    try:
        products['day_night_diff'] = day_night_temperature_difference(temp_arrays, cycle_hour)
    except Exception as e:
        _dbg(f"Could not compute day-night diff: {e}")

    return products
