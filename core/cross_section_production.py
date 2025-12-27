"""Production-quality cross-section generator.

Creates operational meteorological cross-sections with:
- Relative Humidity shading (filled contours)
- Potential Temperature (theta) contours
- Wind barbs showing speed and direction
- Terrain masking
- Animation support for multiple forecast hours
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import warnings


def extract_cross_section_multi_fields(
    grib_file: str,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    n_points: int = 100,
    style: str = "wind_speed",  # Only extract fields needed for this style
) -> Optional[Dict[str, Any]]:
    """Extract fields for production cross-section.

    Args:
        grib_file: Path to pressure-level GRIB file
        start_point: (lat, lon) start
        end_point: (lat, lon) end
        n_points: Points along cross-section
        style: Cross-section style - determines which fields to extract

    Returns:
        Dict with extracted/computed fields, or None
    """
    try:
        import cfgrib
        from scipy.spatial import cKDTree

        path_lats = np.linspace(start_point[0], end_point[0], n_points)
        path_lons = np.linspace(start_point[1], end_point[1], n_points)

        # Base fields always needed
        fields_to_load = {
            't': 'temperature',      # Temperature (K) - needed for theta & freezing level
            'u': 'u_wind',           # U-wind component (m/s) - needed for wind barbs
            'v': 'v_wind',           # V-wind component (m/s) - needed for wind barbs
        }

        # Add style-specific fields
        if style == "wind_speed":
            pass  # Just need u/v for wind speed
        elif style == "rh":
            fields_to_load['r'] = 'rh'
        elif style == "omega":
            fields_to_load['w'] = 'omega'
        elif style == "vorticity":
            fields_to_load['absv'] = 'vorticity'
        elif style == "cloud":
            fields_to_load['clwmr'] = 'cloud'
        else:
            # Load all for unknown styles
            fields_to_load.update({
                'r': 'rh', 'w': 'omega', 'absv': 'vorticity', 'clwmr': 'cloud'
            })

        result = {
            'lats': path_lats,
            'lons': path_lons,
            'distances': _calculate_distances(path_lats, path_lons),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Load each field on pressure levels
            for grib_key, field_name in fields_to_load.items():
                try:
                    ds = cfgrib.open_dataset(
                        grib_file,
                        filter_by_keys={
                            'typeOfLevel': 'isobaricInhPa',
                            'shortName': grib_key,
                        },
                        backend_kwargs={'indexpath': ''},
                    )

                    if ds is None or len(ds.data_vars) == 0:
                        print(f"Could not load {grib_key}")
                        continue

                    var_name = list(ds.data_vars)[0]
                    data = ds[var_name]

                    # Get pressure levels (only once)
                    if 'pressure_levels' not in result:
                        if 'isobaricInhPa' in data.dims:
                            result['pressure_levels'] = data.isobaricInhPa.values
                        elif 'level' in data.dims:
                            result['pressure_levels'] = data.level.values

                    # Get lat/lon grid (only once)
                    if 'lats_grid' not in result:
                        if 'latitude' in data.coords:
                            result['lats_grid'] = data.latitude.values
                            result['lons_grid'] = data.longitude.values
                        else:
                            result['lats_grid'] = data.lat.values
                            result['lons_grid'] = data.lon.values

                        # Convert lons
                        if result['lons_grid'].max() > 180:
                            result['lons_grid'] = np.where(
                                result['lons_grid'] > 180,
                                result['lons_grid'] - 360,
                                result['lons_grid']
                            )

                    # Interpolate to path
                    data_values = data.values
                    n_levels = len(result['pressure_levels'])

                    lats_grid = result['lats_grid']
                    lons_grid = result['lons_grid']

                    # Handle case where field has different number of levels
                    actual_levels = data_values.shape[0]
                    data_3d = np.full((n_levels, n_points), np.nan)

                    if lats_grid.ndim == 2:  # Curvilinear
                        src_pts = np.column_stack([lats_grid.ravel(), lons_grid.ravel()])
                        tree = cKDTree(src_pts)
                        tgt_pts = np.column_stack([path_lats, path_lons])
                        _, indices = tree.query(tgt_pts, k=1)

                        for lev_idx in range(min(actual_levels, n_levels)):
                            level_data = data_values[lev_idx].ravel()
                            data_3d[lev_idx, :] = level_data[indices]
                    else:
                        from scipy.interpolate import RegularGridInterpolator
                        lats_1d = lats_grid if lats_grid.ndim == 1 else lats_grid[:, 0]
                        lons_1d = lons_grid if lons_grid.ndim == 1 else lons_grid[0, :]

                        for lev_idx in range(min(actual_levels, n_levels)):
                            interp = RegularGridInterpolator(
                                (lats_1d, lons_1d), data_values[lev_idx],
                                method='linear', bounds_error=False, fill_value=np.nan
                            )
                            data_3d[lev_idx, :] = interp(np.column_stack([path_lats, path_lons]))

                    result[field_name] = data_3d
                    ds.close()

                except Exception as e:
                    print(f"Error loading {grib_key}: {e}")
                    continue

            # Load surface pressure for terrain masking
            # Try multiple sources: wrfprs file, then wrfsfc file
            sp_loaded = False
            grib_files_to_try = [grib_file]

            # Also try wrfsfc file in same directory
            grib_path = Path(grib_file)
            sfc_file = grib_path.parent / grib_path.name.replace('wrfprs', 'wrfsfc')
            if sfc_file.exists() and str(sfc_file) != grib_file:
                grib_files_to_try.append(str(sfc_file))

            for sp_grib in grib_files_to_try:
                if sp_loaded:
                    break
                try:
                    ds_sp = cfgrib.open_dataset(
                        sp_grib,
                        filter_by_keys={'typeOfLevel': 'surface', 'shortName': 'sp'},
                        backend_kwargs={'indexpath': ''},
                    )
                    if ds_sp and len(ds_sp.data_vars) > 0:
                        sp_var = list(ds_sp.data_vars)[0]
                        sp_data = ds_sp[sp_var].values
                        while sp_data.ndim > 2:
                            sp_data = sp_data[0]

                        lats_grid = result['lats_grid']
                        lons_grid = result['lons_grid']

                        if lats_grid.ndim == 2:
                            src_pts = np.column_stack([lats_grid.ravel(), lons_grid.ravel()])
                            tree = cKDTree(src_pts)
                            tgt_pts = np.column_stack([path_lats, path_lons])
                            _, indices = tree.query(tgt_pts, k=1)
                            sp_path = sp_data.ravel()[indices]
                        else:
                            from scipy.interpolate import RegularGridInterpolator
                            lats_1d = lats_grid if lats_grid.ndim == 1 else lats_grid[:, 0]
                            lons_1d = lons_grid if lons_grid.ndim == 1 else lons_grid[0, :]
                            interp = RegularGridInterpolator(
                                (lats_1d, lons_1d), sp_data,
                                method='linear', bounds_error=False, fill_value=np.nan
                            )
                            sp_path = interp(np.column_stack([path_lats, path_lons]))

                        # Convert Pa to hPa
                        if sp_path.max() > 2000:
                            sp_path = sp_path / 100.0
                        result['surface_pressure'] = sp_path
                        sp_loaded = True
                        ds_sp.close()
                except Exception as e:
                    # Try next file
                    continue

            if not sp_loaded:
                print("Could not load surface pressure from any source")

        # Compute potential temperature (theta)
        if 'temperature' in result and 'pressure_levels' in result:
            T = result['temperature']  # K
            P = result['pressure_levels']  # hPa
            P_ref = 1000.0  # Reference pressure (hPa)
            kappa = 0.286  # R/cp for dry air

            # theta = T * (P_ref / P)^kappa
            theta = np.zeros_like(T)
            for lev_idx, p in enumerate(P):
                theta[lev_idx, :] = T[lev_idx, :] * (P_ref / p) ** kappa

            result['theta'] = theta
            print(f"Computed theta: {theta.min():.1f} to {theta.max():.1f} K")

        # Check we have minimum required fields (theta and pressure are always needed)
        required = ['theta', 'pressure_levels']
        if not all(k in result for k in required):
            missing = [k for k in required if k not in result]
            print(f"Missing required fields: {missing}")
            return None

        print(f"Extracted fields: {[k for k in result.keys() if k not in ['lats_grid', 'lons_grid']]}")
        return result

    except Exception as e:
        print(f"Error extracting cross-section data: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_production_cross_section(
    data: Dict[str, Any],
    cycle: str,
    forecast_hour: int,
    output_dir: Path,
    title_suffix: str = "",
    dpi: int = 150,
    style: str = "wind_speed",  # "wind_speed", "rh", "omega", "vorticity", "cloud"
    fast_mode: bool = False,  # Skip inset map for speed
) -> Optional[Path]:
    """Create production-quality cross-section with configurable shading style.

    Available styles:
    - "wind_speed": Wind speed shading (blue→orange/red)
    - "rh": Relative humidity shading (brown=dry, green=moist)
    - "omega": Vertical velocity (blue=rising, red=sinking)
    - "vorticity": Absolute vorticity (cyclonic/anticyclonic)
    - "cloud": Cloud mixing ratio (white→blue)

    All styles include:
    - Theta contours (black lines)
    - Wind barbs showing actual wind direction
    - Freezing level (magenta line)
    - Terrain masking
    - Inset map with cross-section path

    Args:
        data: Dict from extract_cross_section_multi_fields
        cycle: Model cycle string
        forecast_hour: Forecast hour
        output_dir: Output directory
        title_suffix: Optional suffix for title
        dpi: Output resolution
        style: Shading style (see above)

    Returns:
        Path to saved image
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.ticker import MultipleLocator
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Extract data
        distances = data['distances']
        pressure_levels = data['pressure_levels']
        theta = data['theta'].copy()
        temperature = data.get('temperature')  # For freezing level
        surface_pressure = data.get('surface_pressure')

        u_wind = data.get('u_wind')
        v_wind = data.get('v_wind')

        lats = data['lats']
        lons = data['lons']

        n_levels, n_points = theta.shape

        # Compute wind speed if we have u/v
        if u_wind is not None and v_wind is not None:
            wind_speed = np.sqrt(u_wind**2 + v_wind**2) * 1.944  # m/s to knots
        else:
            wind_speed = None

        # Apply terrain masking
        if surface_pressure is not None:
            for i in range(n_points):
                sp = surface_pressure[i]
                for lev_idx, plev in enumerate(pressure_levels):
                    if plev > sp:
                        theta[lev_idx, i] = np.nan
                        if wind_speed is not None:
                            wind_speed[lev_idx, i] = np.nan
                        if u_wind is not None:
                            u_wind[lev_idx, i] = np.nan
                        if v_wind is not None:
                            v_wind[lev_idx, i] = np.nan
                        if temperature is not None:
                            temperature[lev_idx, i] = np.nan

        # Create figure with extra space at top for inset map
        fig, ax = plt.subplots(figsize=(14, 8.5), facecolor='white')
        # Adjust main axes to leave room for inset map above
        ax.set_position([0.08, 0.08, 0.85, 0.72])

        # Create meshgrid for plotting
        X, Y = np.meshgrid(distances, pressure_levels)

        # Shading based on style
        shading_label = "Unknown"

        if style == "wind_speed" and wind_speed is not None:
            # Wind speed (white/blue→orange/red)
            wspd_colors = [
                '#FFFFFF', '#E0F0FF', '#A0D0FF', '#60B0FF',
                '#FFFF80', '#FFC000', '#FF6000', '#FF0000',
            ]
            wspd_cmap = mcolors.LinearSegmentedColormap.from_list('wspd', wspd_colors, N=256)
            wspd_levels = np.arange(0, 105, 5)
            cf = ax.contourf(X, Y, wind_speed, levels=wspd_levels, cmap=wspd_cmap, extend='max')
            cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.9)
            cbar.set_label('Wind Speed (kts)', fontsize=11)
            shading_label = "Wind Speed(kts)"

        elif style == "omega":
            # Vertical velocity (omega): blue=rising (negative), red=sinking (positive)
            omega = data.get('omega')
            if omega is not None:
                # Apply terrain mask to omega
                if surface_pressure is not None:
                    omega = omega.copy()
                    for i in range(n_points):
                        sp = surface_pressure[i]
                        for lev_idx, plev in enumerate(pressure_levels):
                            if plev > sp:
                                omega[lev_idx, i] = np.nan

                # Convert Pa/s to hPa/hr for better visualization
                omega_display = omega * 36.0  # Pa/s to hPa/hr
                omega_max = np.nanmax(np.abs(omega_display))
                omega_lim = min(omega_max, 20)  # Cap at ±20 hPa/hr

                omega_levels = np.linspace(-omega_lim, omega_lim, 21)
                cf = ax.contourf(X, Y, omega_display, levels=omega_levels, cmap='RdBu_r', extend='both')
                cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.9)
                cbar.set_label('Vertical Velocity (hPa/hr)', fontsize=11)
                shading_label = "ω(hPa/hr)"
            else:
                style = "rh"  # Fallback

        elif style == "vorticity":
            # Absolute vorticity: blue=anticyclonic, red=cyclonic
            vort = data.get('vorticity')
            if vort is not None:
                # Apply terrain mask
                if surface_pressure is not None:
                    vort = vort.copy()
                    for i in range(n_points):
                        sp = surface_pressure[i]
                        for lev_idx, plev in enumerate(pressure_levels):
                            if plev > sp:
                                vort[lev_idx, i] = np.nan

                # Scale to 10^-5 /s for display
                vort_display = vort * 1e5
                vort_max = np.nanmax(np.abs(vort_display))
                vort_lim = min(vort_max, 30)

                vort_levels = np.linspace(-vort_lim, vort_lim, 21)
                cf = ax.contourf(X, Y, vort_display, levels=vort_levels, cmap='RdBu_r', extend='both')
                cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.9)
                cbar.set_label('Abs Vorticity (10⁻⁵ s⁻¹)', fontsize=11)
                shading_label = "Vorticity"
            else:
                style = "rh"

        elif style == "cloud":
            # Cloud mixing ratio: white→blue
            cloud = data.get('cloud')
            if cloud is not None:
                # Apply terrain mask
                if surface_pressure is not None:
                    cloud = cloud.copy()
                    for i in range(n_points):
                        sp = surface_pressure[i]
                        for lev_idx, plev in enumerate(pressure_levels):
                            if plev > sp:
                                cloud[lev_idx, i] = np.nan

                # Convert kg/kg to g/kg
                cloud_display = cloud * 1000
                cloud_colors = ['#FFFFFF', '#E0E8FF', '#B0C4FF', '#6090FF', '#3060D0', '#1040A0']
                cloud_cmap = mcolors.LinearSegmentedColormap.from_list('cloud', cloud_colors, N=256)
                cloud_levels = np.linspace(0, 0.5, 11)  # 0 to 0.5 g/kg
                cf = ax.contourf(X, Y, cloud_display, levels=cloud_levels, cmap=cloud_cmap, extend='max')
                cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.9)
                cbar.set_label('Cloud Mixing Ratio (g/kg)', fontsize=11)
                shading_label = "Cloud(g/kg)"
            else:
                style = "rh"

        if style == "rh":
            # RH shading (brown=dry, green=moist)
            rh = data.get('rh', theta)
            rh_colors = [
                (0.6, 0.4, 0.2), (0.7, 0.5, 0.3), (0.85, 0.75, 0.5),
                (0.9, 0.9, 0.7), (0.7, 0.9, 0.7), (0.4, 0.8, 0.4),
                (0.2, 0.6, 0.3), (0.1, 0.4, 0.2),
            ]
            rh_cmap = mcolors.LinearSegmentedColormap.from_list('rh_cmap', rh_colors, N=256)
            rh_levels = np.arange(0, 105, 5)
            cf = ax.contourf(X, Y, rh, levels=rh_levels, cmap=rh_cmap, extend='both')
            cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.9)
            cbar.set_label('Relative Humidity (%)', fontsize=11)
            shading_label = "RH(%)"

        cbar.ax.tick_params(labelsize=9)

        # Theta contours (black lines) - every 4K like reference
        theta_levels = np.arange(270, 330, 4)
        cs = ax.contour(X, Y, theta, levels=theta_levels, colors='black', linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.0f')

        # Freezing level (0°C isotherm) - magenta line like reference
        if temperature is not None:
            temp_c = temperature - 273.15  # K to C
            try:
                cs_freeze = ax.contour(X, Y, temp_c, levels=[0], colors='magenta', linewidths=2.5)
            except:
                pass  # May not have 0°C in domain

        # Wind barbs - use ACTUAL u/v components, not projected
        if u_wind is not None and v_wind is not None:
            # Subsample for readability - more barbs like reference
            x_skip = max(1, n_points // 25)
            y_skip = max(1, n_levels // 12)

            x_idx = np.arange(0, n_points, x_skip)
            y_idx = np.arange(0, n_levels, y_skip)

            X_barb = distances[x_idx]
            Y_barb = pressure_levels[y_idx]
            XX_barb, YY_barb = np.meshgrid(X_barb, Y_barb)

            # Get wind at subsampled points
            U_barb = u_wind[np.ix_(y_idx, x_idx)].copy()
            V_barb = v_wind[np.ix_(y_idx, x_idx)].copy()

            # Convert m/s to knots
            U_kt = U_barb * 1.944
            V_kt = V_barb * 1.944

            # Rotate winds to cross-section view
            # Calculate section azimuth
            dlat = lats[-1] - lats[0]
            dlon = lons[-1] - lons[0]
            section_azimuth = np.arctan2(dlon * np.cos(np.radians(np.mean(lats))), dlat)

            # Rotate wind vectors so section-parallel is horizontal
            U_rot = U_kt * np.sin(section_azimuth) + V_kt * np.cos(section_azimuth)
            V_rot = -U_kt * np.cos(section_azimuth) + V_kt * np.sin(section_azimuth)

            # Plot wind barbs - V_rot shows cross-section perpendicular component
            # For display, we show the rotated barbs
            ax.barbs(
                XX_barb, YY_barb, U_rot, V_rot,
                length=5, barbcolor='black', flagcolor='black',
                linewidth=0.6, pivot='middle',
                sizes=dict(emptybarb=0.04, spacing=0.12, height=0.35),
            )

        # Add terrain fill
        if surface_pressure is not None:
            max_p = max(pressure_levels.max(), surface_pressure.max()) + 20
            terrain_x = np.concatenate([[distances[0]], distances, [distances[-1]]])
            terrain_y = np.concatenate([[max_p], surface_pressure, [max_p]])
            ax.fill(terrain_x, terrain_y, color='saddlebrown', alpha=0.9, zorder=5)
            ax.plot(distances, surface_pressure, 'k-', linewidth=1.5, zorder=6)

        # Configure axes
        ax.set_ylim(max(pressure_levels), min(pressure_levels))  # Invert (high P at bottom)
        ax.set_xlim(0, distances[-1])

        ax.set_xlabel('Distance (km)', fontsize=11)
        ax.set_ylabel('Pressure (hPa)', fontsize=11)

        # Y-axis ticks every 100 hPa
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(50))

        # Grid
        ax.grid(True, which='major', linestyle='-', alpha=0.3, color='gray')
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray')

        # Add x-axis lat/lon labels
        n_labels = 5
        label_indices = np.linspace(0, n_points - 1, n_labels, dtype=int)
        x_labels = []
        for idx in label_indices:
            lat, lon = lats[idx], lons[idx]
            x_labels.append(f"{lat:.1f}, {lon:.1f}")

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([distances[i] for i in label_indices])
        ax2.set_xticklabels(x_labels, fontsize=8)
        ax2.tick_params(axis='x', length=0)

        # Title - like reference style
        title = f"HRRR 3km θ(K), {shading_label}, and Freezing Level    "
        title += f"Forecast Hour:{forecast_hour:03d}    "
        title += f"Valid: {cycle}"
        ax.set_title(title, fontsize=11, loc='left')

        # Add inset map showing cross-section path (skip in fast_mode)
        if not fast_mode:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature

                lon_min, lon_max = min(lons) - 3, max(lons) + 3
                lat_min, lat_max = min(lats) - 2, max(lats) + 2

                axins = fig.add_axes([0.08, 0.82, 0.25, 0.16],
                                     projection=ccrs.PlateCarree())
                axins.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

                try:
                    axins.stock_img()
                except:
                    axins.add_feature(cfeature.LAND, facecolor='#C4B896', zorder=0)
                    axins.add_feature(cfeature.OCEAN, facecolor='#97B6C8', zorder=0)

                axins.add_feature(cfeature.LAKES, facecolor='#97B6C8', edgecolor='#6090A0', linewidth=0.3, zorder=1)
                axins.add_feature(cfeature.STATES, edgecolor='#444444', linewidth=0.5, zorder=2)
                axins.add_feature(cfeature.BORDERS, edgecolor='#222222', linewidth=0.8, zorder=2)
                axins.add_feature(cfeature.COASTLINE, edgecolor='#444444', linewidth=0.5, zorder=2)

                axins.plot(lons, lats, 'r-', linewidth=3, transform=ccrs.PlateCarree(), zorder=10)
                axins.plot(lons[0], lats[0], 'ro', markersize=6, transform=ccrs.PlateCarree(), zorder=11)
                axins.plot(lons[-1], lats[-1], 'ro', markersize=6, transform=ccrs.PlateCarree(), zorder=11)

                for spine in axins.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)

            except Exception as e:
                pass  # Skip inset if it fails

        # Save (don't use tight_layout - conflicts with inset map)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        style_short = {'wind_speed': 'wspd', 'rh': 'rh', 'omega': 'omega', 'vorticity': 'vort', 'cloud': 'cloud'}.get(style, style)
        output_path = output_dir / f"xsect_{style_short}_theta_f{forecast_hour:02d}.png"
        plt.savefig(output_path, dpi=dpi, facecolor='white')
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating cross-section: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_cross_section_animation(
    grib_files: List[Tuple[str, int]],  # List of (grib_path, forecast_hour)
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    cycle: str,
    output_dir: Path,
    n_points: int = 100,
    fps: int = 2,
    style: str = "wind_speed",
) -> Optional[Path]:
    """Create animated GIF of cross-sections across forecast hours.

    Args:
        grib_files: List of (grib_path, forecast_hour) tuples
        start_point: (lat, lon) start
        end_point: (lat, lon) end
        cycle: Model cycle string
        output_dir: Output directory
        n_points: Points along cross-section
        fps: Frames per second
        style: Shading style (wind_speed, rh, omega, vorticity, cloud)

    Returns:
        Path to animated GIF
    """
    try:
        import imageio.v2 as imageio
        from PIL import Image

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        frame_paths = []
        cached_surface_pressure = None  # Cache terrain (doesn't change between hours)

        for grib_file, fhr in grib_files:
            print(f"Processing F{fhr:02d}...")

            # Extract data (only fields needed for this style)
            data = extract_cross_section_multi_fields(
                grib_file, start_point, end_point, n_points, style=style
            )

            # Use cached surface pressure if current hour doesn't have it
            if data is not None:
                if 'surface_pressure' not in data and cached_surface_pressure is not None:
                    data['surface_pressure'] = cached_surface_pressure
                    print(f"  Using cached terrain from previous hour")
                elif 'surface_pressure' in data:
                    cached_surface_pressure = data['surface_pressure']

            if data is None:
                print(f"  Failed to extract data for F{fhr:02d}")
                continue

            # Create frame (fast_mode for animations - skip inset map)
            frame_path = create_production_cross_section(
                data=data,
                cycle=cycle,
                forecast_hour=fhr,
                output_dir=frames_dir,
                dpi=100,
                style=style,
                fast_mode=True,
            )

            if frame_path:
                frame_paths.append(frame_path)
                print(f"  Created frame: {frame_path.name}")

        if not frame_paths:
            print("No frames created")
            return None

        # Create GIF
        images = []
        for fp in frame_paths:
            img = Image.open(fp)
            images.append(img)

        style_short = {'wind_speed': 'wspd', 'rh': 'rh', 'omega': 'omega', 'vorticity': 'vort', 'cloud': 'cloud'}.get(style, style)
        output_path = output_dir / f"xsect_{style_short}_{cycle}.gif"
        duration = 1000 // fps  # milliseconds per frame

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
        )

        print(f"Created animation: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_distances(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Calculate cumulative distance along path in km."""
    R = 6371  # Earth radius

    distances = [0]
    for i in range(1, len(lats)):
        lat1, lon1 = np.radians(lats[i-1]), np.radians(lons[i-1])
        lat2, lon2 = np.radians(lats[i]), np.radians(lons[i])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        distances.append(distances[-1] + R * c)

    return np.array(distances)
