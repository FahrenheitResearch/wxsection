# Atmospheric Cross-Sections from HRRR Data

This document describes our implementation of vertical cross-section visualizations from HRRR (High-Resolution Rapid Refresh) model data, including lessons learned and guidance for implementing similar systems.

## Overview

Cross-sections slice through the atmosphere along a path between two geographic points, showing vertical structure of meteorological fields. Our implementation supports multiple visualization styles with production-quality output suitable for operational meteorology.

**Example Output:**
- Wind speed + potential temperature (theta) contours
- Relative humidity shading
- Vertical velocity (omega) showing rising/sinking motion
- Absolute vorticity patterns
- Cloud mixing ratio

All styles include wind barbs, freezing level (0°C isotherm), terrain masking, and an inset map showing the cross-section path.

## Quick Start

```python
from pathlib import Path
from core.cross_section_production import (
    extract_cross_section_multi_fields,
    create_production_cross_section,
    create_cross_section_animation,
)

# Single frame
grib_file = "outputs/hrrr/20251224/19z/F00/hrrr.t19z.wrfprsf00.grib2"
start_point = (39.74, -104.99)  # Denver (lat, lon)
end_point = (41.88, -87.63)     # Chicago

data = extract_cross_section_multi_fields(
    grib_file, start_point, end_point,
    n_points=100,
    style="wind_speed"
)

create_production_cross_section(
    data=data,
    cycle="20251224_19Z",
    forecast_hour=0,
    output_dir=Path("outputs/xsect"),
    style="wind_speed",  # or: rh, omega, vorticity, cloud
)

# Animation across forecast hours
grib_files = [
    ("path/to/F00/hrrr.grib2", 0),
    ("path/to/F01/hrrr.grib2", 1),
    ("path/to/F02/hrrr.grib2", 2),
]

create_cross_section_animation(
    grib_files=grib_files,
    start_point=start_point,
    end_point=end_point,
    cycle="20251224_19Z",
    output_dir=Path("outputs/xsect"),
    style="omega",
    n_points=80,
    fps=2,
)
```

## Available Styles

| Style | Shading Variable | Colormap | Use Case |
|-------|-----------------|----------|----------|
| `wind_speed` | Wind speed (kts) | White→Blue→Orange→Red | Jet stream, wind maxima |
| `rh` | Relative Humidity (%) | Brown→Green | Moisture transport, dry intrusions |
| `omega` | Vertical velocity (hPa/hr) | Blue↔Red (diverging) | Rising (blue) / sinking (red) motion |
| `vorticity` | Absolute vorticity (10⁻⁵ s⁻¹) | Blue↔Red (diverging) | Cyclonic/anticyclonic patterns |
| `cloud` | Cloud mixing ratio (g/kg) | White→Blue | Cloud layers, precipitation potential |

All styles include:
- **Theta contours** (black lines, every 4K) - shows atmospheric stability
- **Wind barbs** - rotated to cross-section orientation
- **Freezing level** (magenta line) - 0°C isotherm
- **Terrain masking** - grays out below-ground areas
- **Inset map** - shows cross-section path on terrain background

## Implementation Details

### Data Extraction

We use `cfgrib` to read GRIB2 files, extracting fields on isobaric (pressure) levels:

```python
# Fields extracted from wrfprs files
fields = {
    't': 'temperature',      # Temperature (K)
    'r': 'rh',               # Relative Humidity (%)
    'u': 'u_wind',           # U-wind component (m/s)
    'v': 'v_wind',           # V-wind component (m/s)
    'w': 'omega',            # Vertical velocity (Pa/s)
    'absv': 'vorticity',     # Absolute vorticity (1/s)
    'clwmr': 'cloud',        # Cloud mixing ratio (kg/kg)
}
```

**Key insight:** Only extract fields needed for the requested style. This significantly speeds up processing (2x improvement in our tests).

### Interpolation to Cross-Section Path

The HRRR grid is Lambert Conformal, but cfgrib provides lat/lon coordinates. We interpolate to the cross-section path using:

1. **For curvilinear grids:** `scipy.spatial.cKDTree` for nearest-neighbor lookup
2. **For regular grids:** `scipy.interpolate.RegularGridInterpolator` for bilinear interpolation

```python
# Create path points
path_lats = np.linspace(start_lat, end_lat, n_points)
path_lons = np.linspace(start_lon, end_lon, n_points)

# Calculate cumulative distance along path (Haversine formula)
distances = calculate_distances(path_lats, path_lons)  # Returns km
```

### Computing Derived Fields

**Potential Temperature (Theta):**
```python
# θ = T × (P_ref / P)^κ
# where κ = R/cp ≈ 0.286 for dry air
P_ref = 1000.0  # hPa
kappa = 0.286
theta = temperature * (P_ref / pressure) ** kappa
```

**Wind Speed:**
```python
wind_speed = np.sqrt(u_wind**2 + v_wind**2) * 1.944  # m/s to knots
```

### Wind Barb Rotation

Wind barbs in cross-sections need rotation to show the component along/across the section:

```python
# Calculate section azimuth
dlat = lats[-1] - lats[0]
dlon = lons[-1] - lons[0]
section_azimuth = np.arctan2(dlon * np.cos(np.radians(np.mean(lats))), dlat)

# Rotate wind vectors
U_rot = U * np.sin(section_azimuth) + V * np.cos(section_azimuth)
V_rot = -U * np.cos(section_azimuth) + V * np.sin(section_azimuth)
```

### Terrain Masking

Surface pressure defines where the ground is. Mask data below ground level:

```python
for i in range(n_points):
    surface_p = surface_pressure[i]  # hPa
    for lev_idx, plev in enumerate(pressure_levels):
        if plev > surface_p:  # Below ground
            data[lev_idx, i] = np.nan
```

Also add a terrain fill polygon:
```python
ax.fill(distances, surface_pressure, color='saddlebrown', alpha=0.9)
```

## Lessons Learned

### 1. GRIB File Handling

**Problem:** Corrupted/truncated GRIB files from interrupted downloads cause cfgrib to hang or crash.

**Solution:**
- Verify file integrity before processing
- Use `errors='ignore'` when opening datasets
- Cache surface pressure (terrain) from successful hours for use when files are corrupted

```python
cached_surface_pressure = None
for grib_file, fhr in grib_files:
    data = extract_data(grib_file)
    if 'surface_pressure' not in data and cached_surface_pressure is not None:
        data['surface_pressure'] = cached_surface_pressure
    elif 'surface_pressure' in data:
        cached_surface_pressure = data['surface_pressure']
```

### 2. Performance Optimization

**Problem:** Initial implementation took ~30 seconds per frame.

**Solutions applied:**
1. **Selective field extraction** - Only load variables needed for the style (2x speedup)
2. **Skip inset map for animations** - Cartopy's `stock_img()` is slow
3. **Reduce resolution** - 60-80 points along path is sufficient, lower DPI for animations
4. **Cache terrain** - Surface pressure doesn't change between forecast hours

**Result:** ~14 seconds per frame (2x improvement)

### 3. Coordinate Systems

**Problem:** HRRR uses Lambert Conformal projection, but we need lat/lon for cross-section paths.

**Solution:** cfgrib automatically provides lat/lon coordinates. Handle longitude convention:
```python
# HRRR uses 0-360° longitude, convert to -180 to 180°
if lons.max() > 180:
    lons = np.where(lons > 180, lons - 360, lons)
```

### 4. Colormap Selection

**For diverging data (omega, vorticity):** Use symmetric colormaps centered on zero
```python
data_max = np.nanmax(np.abs(data))
levels = np.linspace(-data_max, data_max, 21)
cf = ax.contourf(X, Y, data, levels=levels, cmap='RdBu_r')
```

**For sequential data (wind speed, cloud):** Use perceptually uniform colormaps
```python
# Custom colormap for wind speed
colors = ['#FFFFFF', '#E0F0FF', '#A0D0FF', '#60B0FF',
          '#FFFF80', '#FFC000', '#FF6000', '#FF0000']
cmap = mcolors.LinearSegmentedColormap.from_list('wspd', colors)
```

### 5. Matplotlib Layout with Inset Maps

**Problem:** `tight_layout()` conflicts with manually positioned inset axes.

**Solution:** Manually position all axes:
```python
fig, ax = plt.subplots(figsize=(14, 8.5))
ax.set_position([0.08, 0.08, 0.85, 0.72])  # Main plot

# Inset map positioned above
axins = fig.add_axes([0.08, 0.82, 0.25, 0.16], projection=ccrs.PlateCarree())
```

### 6. Pressure Level Handling

**Problem:** Different variables may have different numbers of pressure levels in GRIB files.

**Solution:** Handle variable-length level arrays:
```python
n_levels = len(pressure_levels)
data_3d = np.full((n_levels, n_points), np.nan)
actual_levels = data.shape[0]

for lev_idx in range(min(actual_levels, n_levels)):
    # Interpolate this level to path
    data_3d[lev_idx, :] = interpolate_to_path(data[lev_idx])
```

## File Structure

```
core/
├── cross_section_production.py   # Main cross-section module
│   ├── extract_cross_section_multi_fields()  # Data extraction
│   ├── create_production_cross_section()     # Single frame
│   └── create_cross_section_animation()      # Animated GIF

tools/
├── test_production_xsect.py      # Test script for production cross-sections
└── test_smoke_cross_section.py   # Surface-level smoke cross-sections

outputs/
└── xsect_suite/
    ├── wind_speed/
    ├── rh/
    ├── omega/
    ├── vorticity/
    └── cloud/
```

## Dependencies

```
numpy
scipy
matplotlib
cfgrib
cartopy  # Optional, for inset maps
imageio  # For GIF animation
Pillow   # For GIF optimization
```

## Future Improvements

1. **Parallel processing** - Process multiple forecast hours concurrently
2. **Height coordinates** - Option to display in meters/feet instead of pressure
3. **Interactive cross-sections** - Web-based tool to draw paths on map
4. **Additional fields** - CAPE, CIN, equivalent potential temperature (theta-e)
5. **Smoother terrain** - Use actual topography instead of surface pressure

## References

- HRRR Model Documentation: https://rapidrefresh.noaa.gov/hrrr/
- cfgrib: https://github.com/ecmwf/cfgrib
- Matplotlib Barbs: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barbs.html
- Potential Temperature: https://glossary.ametsoc.org/wiki/Potential_temperature
