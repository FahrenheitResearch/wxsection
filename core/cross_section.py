"""Cross-section generator for vertical atmospheric slices.

Creates weathernerds.org-style cross-sections with filled contours,
contour lines with labels, professional meteorological styling,
and proper terrain masking.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import xarray as xr


def extract_surface_pressure(
    grib_file: str,
    path_lats: np.ndarray,
    path_lons: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract surface pressure along a cross-section path.

    Args:
        grib_file: Path to GRIB2 file
        path_lats: Latitudes along the path
        path_lons: Longitudes along the path

    Returns:
        1D array of surface pressure in hPa along the path, or None
    """
    try:
        import cfgrib
        import warnings
        from scipy.spatial import cKDTree

        # Try different surface pressure variable names
        sp_keys = ['sp', 'pres', 'surface_pressure']

        ds = None
        for sp_key in sp_keys:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ds = cfgrib.open_dataset(
                        grib_file,
                        filter_by_keys={
                            'typeOfLevel': 'surface',
                            'shortName': sp_key,
                        },
                        backend_kwargs={'indexpath': ''},
                    )
                if ds is not None and len(ds.data_vars) > 0:
                    break
            except:
                continue

        if ds is None or len(ds.data_vars) == 0:
            # Try without shortName filter
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ds = cfgrib.open_dataset(
                        grib_file,
                        filter_by_keys={
                            'typeOfLevel': 'surface',
                            'paramId': 134,  # Surface pressure paramId
                        },
                        backend_kwargs={'indexpath': ''},
                    )
            except:
                pass

        if ds is None or len(ds.data_vars) == 0:
            print("Could not load surface pressure")
            return None

        # Get the data
        var_name = list(ds.data_vars)[0]
        sp_data = ds[var_name]
        print(f"Loaded surface pressure: {var_name}, shape: {sp_data.shape}")

        # Get lat/lon grid
        if 'latitude' in sp_data.coords:
            lats_grid = sp_data.latitude.values
            lons_grid = sp_data.longitude.values
        else:
            lats_grid = sp_data.lat.values
            lons_grid = sp_data.lon.values

        # Convert lons to -180/180 if needed
        if lons_grid.max() > 180:
            lons_grid = np.where(lons_grid > 180, lons_grid - 360, lons_grid)

        sp_values = sp_data.values
        # Handle extra dimensions
        while sp_values.ndim > 2:
            sp_values = sp_values[0]

        # Interpolate to path using KDTree
        is_curvilinear = lats_grid.ndim == 2

        if is_curvilinear:
            src_pts = np.column_stack([lats_grid.ravel(), lons_grid.ravel()])
            tree = cKDTree(src_pts)
            tgt_pts = np.column_stack([path_lats, path_lons])
            _, indices = tree.query(tgt_pts, k=1)
            sp_path = sp_values.ravel()[indices]
        else:
            from scipy.interpolate import RegularGridInterpolator
            if lats_grid.ndim == 1:
                lats_1d, lons_1d = lats_grid, lons_grid
            else:
                lats_1d = lats_grid[:, 0]
                lons_1d = lons_grid[0, :]
            interp = RegularGridInterpolator(
                (lats_1d, lons_1d), sp_values,
                method='linear', bounds_error=False, fill_value=np.nan
            )
            sp_path = interp(np.column_stack([path_lats, path_lons]))

        ds.close()

        # Convert Pa to hPa if needed
        if sp_path.max() > 2000:  # Probably in Pa
            sp_path = sp_path / 100.0

        print(f"Surface pressure range: {sp_path.min():.1f} to {sp_path.max():.1f} hPa")
        return sp_path

    except Exception as e:
        print(f"Error extracting surface pressure: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_terrain_height(
    grib_file: str,
    path_lats: np.ndarray,
    path_lons: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract terrain/orography height along a cross-section path.

    Args:
        grib_file: Path to GRIB2 file
        path_lats: Latitudes along the path
        path_lons: Longitudes along the path

    Returns:
        1D array of terrain height in meters along the path, or None
    """
    try:
        import cfgrib
        import warnings
        from scipy.spatial import cKDTree

        # Try different orography variable names
        orog_keys = ['orog', 'z', 'hgt', 'gh']

        ds = None
        for orog_key in orog_keys:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ds = cfgrib.open_dataset(
                        grib_file,
                        filter_by_keys={
                            'typeOfLevel': 'surface',
                            'shortName': orog_key,
                        },
                        backend_kwargs={'indexpath': ''},
                    )
                if ds is not None and len(ds.data_vars) > 0:
                    break
            except:
                continue

        if ds is None or len(ds.data_vars) == 0:
            return None

        var_name = list(ds.data_vars)[0]
        orog_data = ds[var_name]

        # Get lat/lon grid
        if 'latitude' in orog_data.coords:
            lats_grid = orog_data.latitude.values
            lons_grid = orog_data.longitude.values
        else:
            lats_grid = orog_data.lat.values
            lons_grid = orog_data.lon.values

        if lons_grid.max() > 180:
            lons_grid = np.where(lons_grid > 180, lons_grid - 360, lons_grid)

        orog_values = orog_data.values
        while orog_values.ndim > 2:
            orog_values = orog_values[0]

        # Interpolate to path
        is_curvilinear = lats_grid.ndim == 2

        if is_curvilinear:
            src_pts = np.column_stack([lats_grid.ravel(), lons_grid.ravel()])
            tree = cKDTree(src_pts)
            tgt_pts = np.column_stack([path_lats, path_lons])
            _, indices = tree.query(tgt_pts, k=1)
            orog_path = orog_values.ravel()[indices]
        else:
            from scipy.interpolate import RegularGridInterpolator
            if lats_grid.ndim == 1:
                lats_1d, lons_1d = lats_grid, lons_grid
            else:
                lats_1d = lats_grid[:, 0]
                lons_1d = lons_grid[0, :]
            interp = RegularGridInterpolator(
                (lats_1d, lons_1d), orog_values,
                method='linear', bounds_error=False, fill_value=0
            )
            orog_path = interp(np.column_stack([path_lats, path_lons]))

        ds.close()

        # Convert geopotential to height if needed (divide by g=9.80665)
        if orog_path.max() > 10000:  # Probably geopotential, not height
            orog_path = orog_path / 9.80665

        print(f"Terrain height range: {orog_path.min():.0f} to {orog_path.max():.0f} m")
        return orog_path

    except Exception as e:
        print(f"Error extracting terrain height: {e}")
        return None


def apply_terrain_mask(
    data_3d: np.ndarray,
    pressure_levels: np.ndarray,
    surface_pressure: np.ndarray,
) -> np.ndarray:
    """Mask data values that are below ground (pressure > surface pressure).

    Args:
        data_3d: Shape (n_levels, n_points)
        pressure_levels: 1D array of pressure levels in hPa
        surface_pressure: 1D array of surface pressure in hPa at each point

    Returns:
        Masked data array with underground values set to NaN
    """
    data_masked = data_3d.copy()
    n_levels, n_points = data_3d.shape

    for i in range(n_points):
        sp = surface_pressure[i]
        for lev_idx, plev in enumerate(pressure_levels):
            if plev > sp:  # Underground (higher pressure than surface)
                data_masked[lev_idx, i] = np.nan

    return data_masked


def create_cross_section(
    data_3d: np.ndarray,  # shape: (n_levels, n_points)
    pressure_levels: np.ndarray,  # 1D array of pressure levels in hPa
    lats: np.ndarray,  # 1D array of latitudes along cross-section
    lons: np.ndarray,  # 1D array of longitudes along cross-section
    field_name: str,
    field_config: Dict[str, Any],
    cycle: str,
    forecast_hour: int,
    output_dir: Path,
    colormap: str = "viridis",
    n_contours: int = 20,
    surface_pressure: Optional[np.ndarray] = None,  # For terrain masking
    terrain_height: Optional[np.ndarray] = None,  # For terrain display (meters)
) -> Optional[Path]:
    """Create a weathernerds.org-style cross-section plot with terrain masking.

    Args:
        data_3d: 2D array of shape (n_levels, n_points) - vertical slice
        pressure_levels: Pressure levels in hPa (will be inverted so surface at bottom)
        lats: Latitudes along the cross-section path
        lons: Longitudes along the cross-section path
        field_name: Name of the field
        field_config: Field configuration dict
        cycle: Model cycle string
        forecast_hour: Forecast hour
        output_dir: Output directory
        colormap: Colormap name
        n_contours: Number of contour levels
        surface_pressure: Optional 1D array of surface pressure (hPa) for terrain masking
        terrain_height: Optional 1D array of terrain height (m) for display

    Returns:
        Path to generated HTML file, or None on failure
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Get value range
        levels = field_config.get('levels')
        if levels and len(levels) >= 2:
            vmin = float(min(levels))
            vmax = float(max(levels))
        else:
            vmin = field_config.get('vmin', float(np.nanmin(data_3d)))
            vmax = field_config.get('vmax', float(np.nanmax(data_3d)))

        units = field_config.get('units', '')
        title = field_config.get('title', field_name)

        # Create x-axis: distance along cross-section or index
        n_points = data_3d.shape[1]
        x_vals = np.arange(n_points)

        # Calculate distances along path for better x-axis
        distances = _calculate_distances(lats, lons)

        # Get plotly colormap
        plotly_cmap = _get_plotly_colormap(colormap, field_name)

        # Create contour levels
        contour_levels = np.linspace(vmin, vmax, n_contours)

        # Format coordinates for title
        start_label = f"{abs(lats[0]):.2f}°{'N' if lats[0] >= 0 else 'S'}, {abs(lons[0]):.2f}°{'W' if lons[0] < 0 else 'E'}"
        end_label = f"{abs(lats[-1]):.2f}°{'N' if lats[-1] >= 0 else 'S'}, {abs(lons[-1]):.2f}°{'W' if lons[-1] < 0 else 'E'}"

        # Apply terrain masking if surface pressure provided
        plot_data = data_3d.copy()
        if surface_pressure is not None:
            plot_data = apply_terrain_mask(plot_data, pressure_levels, surface_pressure)
            print(f"Applied terrain mask using surface pressure")

        # Create figure
        fig = go.Figure()

        # Add filled contour with smooth rendering
        fig.add_trace(go.Contour(
            z=plot_data,
            x=distances,
            y=pressure_levels,
            colorscale=plotly_cmap,
            zmin=vmin,
            zmax=vmax,
            contours=dict(
                start=vmin,
                end=vmax,
                size=(vmax - vmin) / n_contours,
                showlabels=True,
                labelfont=dict(size=9, color='black'),
            ),
            line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
            colorbar=dict(
                title=dict(text=units if units else title, side='right'),
                thickness=15,
                len=0.85,
                tickfont=dict(size=10),
            ),
            hovertemplate=(
                f'Distance: %{{x:.0f}} km<br>'
                f'Pressure: %{{y:.0f}} hPa<br>'
                f'{title}: %{{z:.1f}} {units}<extra></extra>'
            ),
        ))

        # Add terrain fill if surface pressure provided
        if surface_pressure is not None:
            # Create terrain polygon: surface pressure line with fill to bottom
            # Close the polygon by going to max pressure at edges
            max_pressure = max(pressure_levels)

            # Build terrain polygon vertices
            terrain_x = list(distances) + [distances[-1], distances[0]]
            terrain_y = list(surface_pressure) + [max_pressure + 50, max_pressure + 50]

            fig.add_trace(go.Scatter(
                x=terrain_x,
                y=terrain_y,
                fill='toself',
                fillcolor='rgba(139, 90, 43, 0.9)',  # Brown terrain color
                line=dict(color='rgba(80, 50, 20, 1)', width=2),
                mode='lines',
                name='Terrain',
                hoverinfo='skip',
                showlegend=False,
            ))

            # Add terrain surface line on top for cleaner look
            fig.add_trace(go.Scatter(
                x=distances,
                y=surface_pressure,
                mode='lines',
                line=dict(color='rgba(60, 40, 20, 1)', width=2.5),
                name='Surface',
                hovertemplate='Distance: %{x:.0f} km<br>Surface: %{y:.0f} hPa<extra></extra>',
                showlegend=False,
            ))

        # Invert y-axis (pressure decreases with height)
        y_max = max(pressure_levels)
        if surface_pressure is not None:
            y_max = max(y_max, np.max(surface_pressure) + 20)

        fig.update_yaxes(
            range=[y_max, min(pressure_levels)],  # Inverted: high pressure at bottom
            title=dict(text='Pressure (hPa)', font=dict(size=12)),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(size=10),
            dtick=100,  # Tick every 100 hPa
        )

        fig.update_xaxes(
            title=dict(text='Distance (km)', font=dict(size=12)),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(size=10),
        )

        fig.update_layout(
            title=dict(
                text=(
                    f'<b>HRRR: {title}</b><br>'
                    f'<span style="font-size:11px">{start_label} → {end_label}</span><br>'
                    f'<span style="font-size:10px">Valid: {cycle} F{forecast_hour:02d}</span>'
                ),
                x=0.5,
                xanchor='center',
                font=dict(size=14),
            ),
            width=1200,
            height=550,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=80, t=90, b=60),
        )

        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{field_name}_xsect_f{forecast_hour:02d}.html"
        fig.write_html(str(output_path))

        return output_path

    except Exception as e:
        print(f"Error creating cross-section: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_cross_section_multi(
    data_dict: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
    pressure_levels: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cycle: str,
    forecast_hour: int,
    output_dir: Path,
    primary_field: str,
    overlay_field: Optional[str] = None,
    n_contours: int = 20,
) -> Optional[Path]:
    """Create cross-section with optional contour overlay (like temp with RH contours).

    Args:
        data_dict: Dict mapping field_name -> (data_3d, field_config)
        pressure_levels: Pressure levels in hPa
        lats, lons: Coordinates along cross-section
        cycle, forecast_hour: Model timing
        output_dir: Output directory
        primary_field: Field name for filled contours
        overlay_field: Optional field name for contour lines overlay
        n_contours: Number of contour levels

    Returns:
        Path to generated HTML file
    """
    try:
        import plotly.graph_objects as go

        # Get primary data
        primary_data, primary_config = data_dict[primary_field]

        # Value range for primary
        levels = primary_config.get('levels')
        if levels and len(levels) >= 2:
            vmin = float(min(levels))
            vmax = float(max(levels))
        else:
            vmin = primary_config.get('vmin', float(np.nanmin(primary_data)))
            vmax = primary_config.get('vmax', float(np.nanmax(primary_data)))

        units = primary_config.get('units', '')
        title = primary_config.get('title', primary_field)
        colormap = primary_config.get('colormap', 'viridis')
        plotly_cmap = _get_plotly_colormap(colormap, primary_field)

        distances = _calculate_distances(lats, lons)

        fig = go.Figure()

        # Primary filled contour
        fig.add_trace(go.Contour(
            z=primary_data,
            x=distances,
            y=pressure_levels,
            colorscale=plotly_cmap,
            zmin=vmin,
            zmax=vmax,
            contours=dict(
                start=vmin,
                end=vmax,
                size=(vmax - vmin) / n_contours,
                showlabels=True,
                labelfont=dict(size=9, color='black'),
            ),
            line=dict(width=0.5, color='black'),
            colorbar=dict(
                title=dict(text=f'{title} ({units})', side='right'),
                thickness=15,
                len=0.9,
                x=1.02,
            ),
            name=title,
            hovertemplate=f'{title}: %{{z:.1f}} {units}<extra></extra>',
        ))

        # Overlay contour lines if specified
        if overlay_field and overlay_field in data_dict:
            overlay_data, overlay_config = data_dict[overlay_field]
            overlay_title = overlay_config.get('title', overlay_field)
            overlay_units = overlay_config.get('units', '')

            # Overlay range
            o_levels = overlay_config.get('levels')
            if o_levels and len(o_levels) >= 2:
                o_vmin = float(min(o_levels))
                o_vmax = float(max(o_levels))
            else:
                o_vmin = float(np.nanmin(overlay_data))
                o_vmax = float(np.nanmax(overlay_data))

            fig.add_trace(go.Contour(
                z=overlay_data,
                x=distances,
                y=pressure_levels,
                contours=dict(
                    start=o_vmin,
                    end=o_vmax,
                    size=(o_vmax - o_vmin) / 10,
                    showlabels=True,
                    labelfont=dict(size=10, color='white'),
                    coloring='none',  # Lines only
                ),
                line=dict(width=2, color='white'),
                showscale=False,
                name=overlay_title,
                hovertemplate=f'{overlay_title}: %{{z:.1f}} {overlay_units}<extra></extra>',
            ))

        # Configure axes
        fig.update_yaxes(
            autorange='reversed',
            title='Pressure (hPa)',
        )

        start_label = f"{lats[0]:.2f}°N, {lons[0]:.2f}°"
        end_label = f"{lats[-1]:.2f}°N, {lons[-1]:.2f}°"

        fig.update_xaxes(
            title=f'Distance (km)<br><span style="font-size:10px">{start_label} → {end_label}</span>',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
        )

        # Build title
        title_text = f'<b>HRRR: {title}</b>'
        if overlay_field:
            overlay_title = data_dict[overlay_field][1].get('title', overlay_field)
            title_text = f'<b>HRRR: {title} + {overlay_title}</b>'

        fig.update_layout(
            title=dict(
                text=f'{title_text}<br><span style="font-size:12px">Valid: {cycle} F{forecast_hour:02d}</span>',
                x=0.5,
                xanchor='center',
            ),
            width=1200,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.3)',
            ),
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{primary_field}"
        if overlay_field:
            fname += f"_{overlay_field}"
        output_path = output_dir / f"{fname}_xsect_f{forecast_hour:02d}.html"
        fig.write_html(str(output_path))

        return output_path

    except Exception as e:
        print(f"Error creating multi cross-section: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_cross_section_data(
    grib_file: str,
    field_name: str,
    field_config: Dict[str, Any],
    start_point: Tuple[float, float],  # (lat, lon)
    end_point: Tuple[float, float],
    n_points: int = 100,
    model: str = 'hrrr',
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract cross-section data from GRIB file along a path.

    Args:
        grib_file: Path to GRIB2 file
        field_name: Field name to extract
        field_config: Field configuration
        start_point: (lat, lon) of start
        end_point: (lat, lon) of end
        n_points: Number of points along cross-section
        model: Model name

    Returns:
        Tuple of (data_3d, pressure_levels, lats, lons) or None
    """
    try:
        import cfgrib
        import warnings
        from scipy.spatial import cKDTree

        grib_key = field_config.get('grib_key', field_name)

        # Load with cfgrib filtering to pressure levels
        ds = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = cfgrib.open_dataset(
                grib_file,
                filter_by_keys={
                    'typeOfLevel': 'isobaricInhPa',
                    'shortName': grib_key,
                },
                backend_kwargs={'indexpath': ''},
            )

        if ds is None:
            print(f"Could not load {grib_key} on pressure levels")
            return None

        # Get the data variable
        var_name = list(ds.data_vars)[0]
        data = ds[var_name]
        print(f"Loaded variable: {var_name}, shape: {data.shape}, dims: {data.dims}")

        # Get pressure levels
        if 'isobaricInhPa' in data.dims:
            pressure_levels = data.isobaricInhPa.values
        elif 'level' in data.dims:
            pressure_levels = data.level.values
        else:
            print(f"No pressure dimension found. Dims: {data.dims}")
            return None

        # Get lat/lon
        if 'latitude' in data.coords:
            lats_grid = data.latitude.values
            lons_grid = data.longitude.values
        else:
            lats_grid = data.lat.values
            lons_grid = data.lon.values

        # Convert lons to -180/180 if needed
        if lons_grid.max() > 180:
            lons_grid = np.where(lons_grid > 180, lons_grid - 360, lons_grid)

        # Create cross-section path
        path_lats = np.linspace(start_point[0], end_point[0], n_points)
        path_lons = np.linspace(start_point[1], end_point[1], n_points)

        # Get data values
        data_values = data.values  # shape: (n_levels, ny, nx)
        n_levels = len(pressure_levels)

        print(f"Data values shape: {data_values.shape}, n_levels: {n_levels}")

        # Handle curvilinear vs rectilinear grids
        is_curvilinear = lats_grid.ndim == 2

        if is_curvilinear:
            # Use KDTree for curvilinear grid interpolation
            src_pts = np.column_stack([lats_grid.ravel(), lons_grid.ravel()])
            tree = cKDTree(src_pts)

            # Query for path points
            tgt_pts = np.column_stack([path_lats, path_lons])
            _, indices = tree.query(tgt_pts, k=1)

            # Extract data along path for each level
            data_3d = np.zeros((n_levels, n_points))
            for lev_idx in range(n_levels):
                level_data = data_values[lev_idx].ravel()
                data_3d[lev_idx, :] = level_data[indices]
        else:
            # Regular grid - use scipy interpolator
            from scipy.interpolate import RegularGridInterpolator

            # Create 1D coordinate arrays
            if lats_grid.ndim == 1:
                lats_1d = lats_grid
                lons_1d = lons_grid
            else:
                lats_1d = lats_grid[:, 0]
                lons_1d = lons_grid[0, :]

            data_3d = np.zeros((n_levels, n_points))
            for lev_idx in range(n_levels):
                interp = RegularGridInterpolator(
                    (lats_1d, lons_1d),
                    data_values[lev_idx],
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan,
                )
                data_3d[lev_idx, :] = interp(np.column_stack([path_lats, path_lons]))

        ds.close()

        # Apply unit conversion if needed
        if 'convert' in field_config:
            conv = field_config['convert']
            if conv == 'K_to_F':
                data_3d = (data_3d - 273.15) * 9/5 + 32
            elif conv == 'K_to_C':
                data_3d = data_3d - 273.15
            elif conv == 'Pa_to_hPa':
                data_3d = data_3d / 100

        return data_3d, pressure_levels, path_lats, path_lons

    except Exception as e:
        print(f"Error extracting cross-section data: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_distances(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Calculate cumulative distance along path in km."""
    R = 6371  # Earth radius in km

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


def extract_surface_smoke_cross_section(
    grib_file: str,
    start_point: Tuple[float, float],  # (lat, lon)
    end_point: Tuple[float, float],
    n_points: int = 100,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract smoke concentration along a cross-section path at surface heights.

    HRRR smoke (MASSDEN) is only available at 1m, 2m, and 8m AGL, not on pressure
    levels like temperature/wind. This function extracts smoke at these heights
    along a horizontal path.

    Args:
        grib_file: Path to GRIB2 file (wrfsfc or wrfprs)
        start_point: (lat, lon) of start
        end_point: (lat, lon) of end
        n_points: Number of points along cross-section

    Returns:
        Tuple of (smoke_data, heights, lats, lons, distances) or None
        smoke_data shape: (n_heights, n_points)
    """
    try:
        import cfgrib
        import warnings
        from scipy.spatial import cKDTree

        # Create cross-section path first
        path_lats = np.linspace(start_point[0], end_point[0], n_points)
        path_lons = np.linspace(start_point[1], end_point[1], n_points)
        distances = _calculate_distances(path_lats, path_lons)

        # HRRR smoke (MASSDEN) is stored as 'unknown' at heights [1, 2, 8]m
        # We need to find the right dataset by enumerating all datasets
        data = None
        heights = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            datasets = cfgrib.open_datasets(
                grib_file,
                backend_kwargs={'indexpath': ''},
            )

            # Find the dataset with smoke (heights [1, 2, 8]m at heightAboveGround)
            for ds in datasets:
                for var_name in ds.data_vars:
                    var_data = ds[var_name]
                    # Look for the smoke signature: 3 heights at 1, 2, 8m
                    if 'heightAboveGround' in var_data.dims:
                        h = var_data.heightAboveGround.values
                        if len(h) == 3 and 1.0 in h and 2.0 in h and 8.0 in h:
                            # Check if values are in smoke range (small positive)
                            vals = var_data.values
                            if vals.max() < 100 and vals.max() > 0:  # Smoke in kg/m³
                                data = var_data
                                heights = h
                                print(f"Found smoke data: {var_name} at heights {h}m")
                                break
                if data is not None:
                    break

        if data is None:
            print("Could not find smoke data (looking for heights [1, 2, 8]m)")
            return None

        print(f"Loaded smoke variable, shape: {data.shape}, dims: {data.dims}")

        # Get heights
        if 'heightAboveGround' in data.dims:
            heights = data.heightAboveGround.values
        else:
            heights = np.array([8.0])  # Default to 8m if single level

        # Make heights 1D array
        if np.ndim(heights) == 0:
            heights = np.array([float(heights)])

        print(f"Smoke heights: {heights} m")

        # Get lat/lon grid
        if 'latitude' in data.coords:
            lats_grid = data.latitude.values
            lons_grid = data.longitude.values
        else:
            lats_grid = data.lat.values
            lons_grid = data.lon.values

        # Convert lons to -180/180
        if lons_grid.max() > 180:
            lons_grid = np.where(lons_grid > 180, lons_grid - 360, lons_grid)

        data_values = data.values

        # Handle curvilinear grid with KDTree
        is_curvilinear = lats_grid.ndim == 2

        if is_curvilinear:
            src_pts = np.column_stack([lats_grid.ravel(), lons_grid.ravel()])
            tree = cKDTree(src_pts)
            tgt_pts = np.column_stack([path_lats, path_lons])
            _, indices = tree.query(tgt_pts, k=1)

            # Extract along path for each height
            n_heights = len(heights)
            if data_values.ndim == 2:
                # Single height level
                smoke_path = data_values.ravel()[indices]
                smoke_data = smoke_path.reshape(1, -1)
            else:
                # Multiple height levels
                smoke_data = np.zeros((n_heights, n_points))
                for h_idx in range(n_heights):
                    level_data = data_values[h_idx].ravel()
                    smoke_data[h_idx, :] = level_data[indices]
        else:
            from scipy.interpolate import RegularGridInterpolator
            if lats_grid.ndim == 1:
                lats_1d, lons_1d = lats_grid, lons_grid
            else:
                lats_1d = lats_grid[:, 0]
                lons_1d = lons_grid[0, :]

            n_heights = len(heights)
            smoke_data = np.zeros((n_heights, n_points))

            for h_idx in range(n_heights):
                if data_values.ndim == 2:
                    level_data = data_values
                else:
                    level_data = data_values[h_idx]

                interp = RegularGridInterpolator(
                    (lats_1d, lons_1d), level_data,
                    method='linear', bounds_error=False, fill_value=np.nan
                )
                smoke_data[h_idx, :] = interp(np.column_stack([path_lats, path_lons]))

        ds.close()

        # Convert to more readable units (µg/m³) if very small values
        # MASSDEN is in kg/m³, convert to µg/m³
        smoke_data = smoke_data * 1e9  # kg/m³ -> µg/m³

        print(f"Smoke data shape: {smoke_data.shape}")
        print(f"Smoke range: {np.nanmin(smoke_data):.2f} to {np.nanmax(smoke_data):.2f} µg/m³")

        return smoke_data, heights, path_lats, path_lons, distances

    except Exception as e:
        print(f"Error extracting smoke cross-section: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_smoke_cross_section(
    smoke_data: np.ndarray,  # shape: (n_heights, n_points)
    heights: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    distances: np.ndarray,
    cycle: str,
    forecast_hour: int,
    output_dir: Path,
    surface_pressure: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """Create an interactive smoke cross-section plot.

    Since smoke is only at surface heights (1, 2, 8m), this creates a line plot
    or small heatmap showing smoke concentration along the path.

    Args:
        smoke_data: 2D array (n_heights, n_points) of smoke in µg/m³
        heights: Heights in meters
        lats, lons: Coordinates along path
        distances: Distance in km along path
        cycle: Model cycle
        forecast_hour: Forecast hour
        output_dir: Output directory
        surface_pressure: Optional surface pressure for terrain reference

    Returns:
        Path to HTML file
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_heights, n_points = smoke_data.shape

        # Format coordinates
        start_label = f"{abs(lats[0]):.2f}°{'N' if lats[0] >= 0 else 'S'}, {abs(lons[0]):.2f}°{'W' if lons[0] < 0 else 'E'}"
        end_label = f"{abs(lats[-1]):.2f}°{'N' if lats[-1] >= 0 else 'S'}, {abs(lons[-1]):.2f}°{'W' if lons[-1] < 0 else 'E'}"

        # Create figure with 2 subplots: line plot and heatmap
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            subplot_titles=['Surface Smoke Concentration (8m AGL)', 'Smoke by Height'],
            vertical_spacing=0.15,
        )

        # Color scale for smoke (brown/gray for smoke)
        smoke_colors = [
            [0.0, 'rgba(255, 255, 255, 0)'],
            [0.1, 'rgba(200, 200, 200, 0.5)'],
            [0.3, 'rgba(180, 160, 140, 0.7)'],
            [0.5, 'rgba(139, 90, 43, 0.8)'],
            [0.7, 'rgba(100, 60, 30, 0.9)'],
            [1.0, 'rgba(50, 30, 20, 1.0)'],
        ]

        # Main line plot: 8m smoke (or highest available)
        # Use the highest height level (usually 8m)
        main_height_idx = -1  # Last height = 8m
        main_smoke = smoke_data[main_height_idx, :]

        # Add filled area under the line
        fig.add_trace(
            go.Scatter(
                x=distances,
                y=main_smoke,
                fill='tozeroy',
                fillcolor='rgba(139, 90, 43, 0.4)',
                line=dict(color='rgba(100, 60, 30, 1)', width=2),
                mode='lines',
                name=f'Smoke @ {heights[main_height_idx]:.0f}m',
                hovertemplate='Distance: %{x:.0f} km<br>Smoke: %{y:.1f} µg/m³<extra></extra>',
            ),
            row=1, col=1
        )

        # Mark max smoke location
        max_idx = np.nanargmax(main_smoke)
        max_val = main_smoke[max_idx]
        max_dist = distances[max_idx]
        max_lat = lats[max_idx]
        max_lon = lons[max_idx]

        fig.add_trace(
            go.Scatter(
                x=[max_dist],
                y=[max_val],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='diamond'),
                text=[f'Max: {max_val:.1f}'],
                textposition='top center',
                textfont=dict(size=10, color='red'),
                name='Maximum',
                hovertemplate=(
                    f'Maximum Smoke<br>'
                    f'Distance: %{{x:.0f}} km<br>'
                    f'Value: %{{y:.1f}} µg/m³<br>'
                    f'Location: {max_lat:.2f}°N, {max_lon:.2f}°W<extra></extra>'
                ),
            ),
            row=1, col=1
        )

        # Heatmap of all height levels
        fig.add_trace(
            go.Heatmap(
                z=smoke_data,
                x=distances,
                y=heights,
                colorscale='YlOrBr',
                zmin=0,
                zmax=max(np.nanmax(smoke_data), 10),
                colorbar=dict(
                    title='µg/m³',
                    len=0.35,
                    y=0.2,
                ),
                hovertemplate='Distance: %{x:.0f} km<br>Height: %{y:.0f}m<br>Smoke: %{z:.1f} µg/m³<extra></extra>',
            ),
            row=2, col=1
        )

        # Update axes
        fig.update_xaxes(title_text='Distance (km)', row=1, col=1)
        fig.update_yaxes(title_text='Smoke (µg/m³)', row=1, col=1)
        fig.update_xaxes(title_text='Distance (km)', row=2, col=1)
        fig.update_yaxes(title_text='Height (m)', row=2, col=1)

        fig.update_layout(
            title=dict(
                text=(
                    f'<b>HRRR: Surface Smoke Cross-Section</b><br>'
                    f'<span style="font-size:11px">{start_label} → {end_label}</span><br>'
                    f'<span style="font-size:10px">Valid: {cycle} F{forecast_hour:02d}</span>'
                ),
                x=0.5,
                xanchor='center',
            ),
            width=1200,
            height=700,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        )

        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"smoke_xsect_f{forecast_hour:02d}.html"
        fig.write_html(str(output_path))

        return output_path

    except Exception as e:
        print(f"Error creating smoke cross-section: {e}")
        import traceback
        traceback.print_exc()
        return None


def _get_plotly_colormap(colormap: str, field_name: str) -> str:
    """Convert matplotlib colormap names to Plotly equivalents."""
    mappings = {
        'viridis': 'Viridis',
        'plasma': 'Plasma',
        'inferno': 'Inferno',
        'magma': 'Magma',
        'turbo': 'Turbo',
        'jet': 'Jet',
        'hot': 'Hot',
        'cool': 'ice',
        'coolwarm': 'RdBu',
        'RdBu_r': 'RdBu_r',
        'RdBu': 'RdBu',
        'BrBG': 'BrBG',
        'RdYlGn': 'RdYlGn',
        'Spectral': 'Spectral',
        'Blues': 'Blues',
        'Greens': 'Greens',
        'Reds': 'Reds',
        'YlOrRd': 'YlOrRd',
        'YlGnBu': 'YlGnBu',
    }

    if colormap in mappings:
        return mappings[colormap]

    # Field-specific defaults
    if 'temp' in field_name.lower() or 't_' in field_name.lower():
        return 'RdBu_r'
    if 'wind' in field_name.lower() or 'wspd' in field_name.lower():
        return 'Turbo'
    if 'rh' in field_name.lower() or 'humidity' in field_name.lower():
        return 'YlGnBu'
    if 'omega' in field_name.lower() or 'vvel' in field_name.lower():
        return 'RdBu_r'

    return 'Viridis'
