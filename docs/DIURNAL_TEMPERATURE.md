# Diurnal Temperature Analysis

Custom implementation for computing diurnal (daily) temperature variations from HRRR model data.

## What is Diurnal Temperature Range?

The **Diurnal Temperature Range (DTR)** is the difference between daily maximum and minimum temperatures. It's a key indicator of:
- **Atmospheric moisture**: Low DTR = humid/cloudy, High DTR = dry/clear
- **Cloud cover**: Clouds trap heat at night, reducing DTR
- **Soil moisture**: Wet soil moderates temperature swings
- **Fire weather**: High DTR often correlates with low humidity and fire risk

## How We Implemented It

### The Core Insight

HRRR produces hourly forecasts up to 48 hours. A single model run contains a complete diurnal cycle:

```
12Z Run Example:
  f00 = 12:00 UTC (noon)
  f06 = 18:00 UTC (evening)
  f12 = 00:00 UTC (midnight)
  f18 = 06:00 UTC (morning)
  f24 = 12:00 UTC (noon, next day)
```

So `f00 → f24` from one run = 24 hours of temperature evolution.

### The Algorithm

```python
# 1. Load 2m temperature for each forecast hour
temp_data = {}
for fhr in range(0, 25):
    temp_data[fhr] = load_grib(f"hrrr.t12z.wrfprsf{fhr:02d}.grib2", var="t2m")

# 2. Stack into 3D array (time, lat, lon)
temps = np.stack(list(temp_data.values()), axis=0)

# 3. Compute max/min along time axis
t_max = np.nanmax(temps, axis=0)  # Shape: (lat, lon)
t_min = np.nanmin(temps, axis=0)

# 4. Diurnal range
dtr = t_max - t_min
```

That's it. The rest is just I/O and plotting.

### Rolling Windows

For a 48-hour forecast, we generate multiple 24-hour DTR maps:
- `f00-f24`: First diurnal cycle
- `f01-f25`: Shifted 1 hour
- `f02-f26`: Shifted 2 hours
- ...
- `f24-f48`: Second diurnal cycle

This shows how the diurnal pattern evolves through the forecast.

## File Structure

```
derived_params/diurnal_temperature.py   # Core calculation functions
parameters/diurnal.json                  # Product configurations
tools/process_diurnal.py                 # CLI processor
config/colormaps.py                      # Added DiurnalRange, HeatingRate, CoolingRate
```

## Products Available

| Product | Description | Units |
|---------|-------------|-------|
| `dtr` | Diurnal Temperature Range (Tmax - Tmin) | °C |
| `t_max_diurnal` | Maximum temperature in period | °C |
| `t_min_diurnal` | Minimum temperature in period | °C |
| `t_mean_diurnal` | Mean temperature in period | °C |
| `heating_rate` | Morning warming rate | °C/hr |
| `cooling_rate` | Evening cooling rate | °C/hr |
| `day_night_diff` | Afternoon minus overnight | °C |
| `diurnal_amplitude` | Half the range | °C |
| `hour_of_max_temp` | Forecast hour of Tmax | hour |
| `hour_of_min_temp` | Forecast hour of Tmin | hour |

## Usage

```bash
# Single 24h window
python tools/process_diurnal.py 20251224 12 --end-fhr 24

# Rolling 24h windows across 48h forecast (25 maps)
python tools/process_diurnal.py 20251224 06 --end-fhr 48 --rolling --workers 12

# Generate GIF animation
python tools/process_diurnal.py 20251224 06 --end-fhr 48 --rolling --workers 12 --gif

# Only synoptic cycles have 48h forecasts
python tools/process_diurnal.py --latest --synoptic --end-fhr 48 --rolling --gif
```

---

## Implementing with Herbie + MetPy

If you're using [Herbie](https://herbie.readthedocs.io/) for data access, here's how to implement the same thing:

### Minimal Example

```python
from herbie import Herbie
import numpy as np
import xarray as xr

# Download 24 hours of 2m temperature
date = "2024-06-15 12:00"
temps = []

for fhr in range(0, 25):
    H = Herbie(date, model="hrrr", fxx=fhr)
    ds = H.xarray("TMP:2 m above ground")
    temps.append(ds["t2m"].values)

# Stack and compute DTR
temps = np.stack(temps, axis=0)
t_max = np.nanmax(temps, axis=0)
t_min = np.nanmin(temps, axis=0)
dtr = t_max - t_min

# Get coordinates from last dataset for plotting
lons = ds.longitude.values
lats = ds.latitude.values
```

### With MetPy Units (Optional)

```python
from metpy.units import units

# MetPy can handle unit conversions
t_max_F = (t_max * units.degC).to(units.degF)
dtr_F = (dtr * units.delta_degC).to(units.delta_degF)
```

### Full Example with Plotting

```python
from herbie import Herbie
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_diurnal_temperature_range(date, model="hrrr"):
    """
    Compute 24h diurnal temperature range from HRRR.

    Args:
        date: Model init time (e.g., "2024-06-15 12:00")
        model: "hrrr" or "rrfs"

    Returns:
        dtr: 2D array of diurnal temperature range (°C)
        lons, lats: Coordinate arrays
    """
    temps = []

    for fhr in range(0, 25):
        H = Herbie(date, model=model, fxx=fhr)
        ds = H.xarray("TMP:2 m above ground")

        # Convert K to C if needed
        t2m = ds["t2m"].values
        if np.nanmean(t2m) > 200:
            t2m = t2m - 273.15

        temps.append(t2m)

    # Stack and compute
    temps = np.stack(temps, axis=0)
    t_max = np.nanmax(temps, axis=0)
    t_min = np.nanmin(temps, axis=0)
    dtr = t_max - t_min

    return dtr, ds.longitude.values, ds.latitude.values


def plot_dtr(dtr, lons, lats, title="Diurnal Temperature Range"):
    """Plot DTR on a CONUS map."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66, 22, 50])

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    levels = [0, 5, 8, 10, 12, 15, 18, 20, 25, 30]
    cf = ax.contourf(lons, lats, dtr, levels=levels,
                     cmap="YlOrRd", extend="max",
                     transform=ccrs.PlateCarree())

    plt.colorbar(cf, label="DTR (°C)", orientation="horizontal", pad=0.05)
    ax.set_title(title)
    plt.savefig("dtr.png", dpi=150, bbox_inches="tight")
    plt.close()


# Usage
dtr, lons, lats = get_diurnal_temperature_range("2024-06-15 12:00")
plot_dtr(dtr, lons, lats, "HRRR 24h Diurnal Temperature Range\n2024-06-15 12Z f00-f24")
```

### Rolling Windows with Herbie

```python
def get_rolling_dtr(date, max_fhr=48, window=24):
    """Generate rolling 24h DTR windows."""

    # Load all forecast hours once
    temps = {}
    for fhr in range(0, max_fhr + 1):
        H = Herbie(date, model="hrrr", fxx=fhr)
        ds = H.xarray("TMP:2 m above ground")
        temps[fhr] = ds["t2m"].values - 273.15  # K to C

    lons = ds.longitude.values
    lats = ds.latitude.values

    # Generate rolling windows
    results = {}
    for start in range(0, max_fhr - window + 1):
        end = start + window
        window_temps = np.stack([temps[h] for h in range(start, end + 1)], axis=0)
        dtr = np.nanmax(window_temps, axis=0) - np.nanmin(window_temps, axis=0)
        results[f"f{start:02d}-f{end:02d}"] = dtr

    return results, lons, lats


# Usage
dtrs, lons, lats = get_rolling_dtr("2024-06-15 06:00", max_fhr=48)
# Returns 25 DTR maps: f00-f24, f01-f25, ..., f24-f48
```

---

## Key Considerations

### Which Cycles Have 48h Forecasts?

Only synoptic cycles (00Z, 06Z, 12Z, 18Z) have 48-hour forecasts. Other hours only go to 18h.

| Cycle | Max Forecast |
|-------|-------------|
| 00Z, 06Z, 12Z, 18Z | 48h |
| All others | 18h |

### Memory Usage

Loading 49 GRIB files into RAM takes ~2-3 GB. For parallel processing, each worker gets a copy of the data arrays passed to it.

### Performance Tips

1. **Use `pcolormesh` instead of `contourf`** for plotting - 5-10x faster
2. **Parallel plotting** with multiprocessing - we use 12 workers
3. **Cache downloaded files** - our pipeline checks for existing files before downloading
4. **Lower DPI** for faster plots (we use 120 instead of 300)

---

## References

- [NOAA HRRR Model](https://rapidrefresh.noaa.gov/hrrr/)
- [Herbie Documentation](https://herbie.readthedocs.io/)
- [Diurnal Temperature Range - Wikipedia](https://en.wikipedia.org/wiki/Diurnal_temperature_variation)
