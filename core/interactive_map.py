"""Interactive map generator for weather data visualization.

Creates Leaflet-based interactive maps where users can hover/click
to see data values at any point - similar to PivotalWeather.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import folium
from folium import plugins


def create_interactive_map(
    data,  # xarray.DataArray
    field_name: str,
    field_config: Dict[str, Any],
    cycle: str,
    forecast_hour: int,
    output_dir: Path,
    colormap: str = "viridis",
) -> Optional[Path]:
    """Create an interactive HTML map with hover values.

    Args:
        data: xarray DataArray with lat/lon coordinates
        field_name: Name of the field
        field_config: Field configuration dict
        cycle: Model cycle string
        forecast_hour: Forecast hour
        output_dir: Output directory
        colormap: Matplotlib colormap name

    Returns:
        Path to generated HTML file, or None on failure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Get data values and coordinates
        values = data.values
        lats = data.latitude.values if 'latitude' in data.coords else data.lat.values
        lons = data.longitude.values if 'longitude' in data.coords else data.lon.values

        # Convert longitudes to -180 to 180 range if needed
        if lons.max() > 180:
            lons = np.where(lons > 180, lons - 360, lons)

        # Get data bounds
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())

        # Calculate center
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        # Get value range for colormap
        # Use levels if available, otherwise vmin/vmax, otherwise data range
        levels = field_config.get('levels')
        if levels and len(levels) >= 2:
            vmin = float(min(levels))
            vmax = float(max(levels))
        else:
            vmin = field_config.get('vmin', float(np.nanmin(values)))
            vmax = field_config.get('vmax', float(np.nanmax(values)))

        # Handle data - minimal subsampling for quality
        # Full HRRR grid is ~1059x1799, subsample to ~530x900 for good quality
        step = 2  # Only 2x subsample for the image

        values_sub = values[::step, ::step]
        lats_sub = lats[::step] if lats.ndim == 1 else lats[::step, ::step]
        lons_sub = lons[::step] if lons.ndim == 1 else lons[::step, ::step]

        # Create color-mapped image
        # Handle custom colormaps - fall back to standard ones
        try:
            cmap = plt.get_cmap(colormap)
        except ValueError:
            # Custom colormap not available, use appropriate fallback
            if 'reflectivity' in field_name.lower() or 'refl' in colormap.lower():
                cmap = plt.get_cmap('turbo')  # Good for radar
            else:
                cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Normalize values and apply colormap
        values_norm = norm(values_sub)
        rgba = cmap(values_norm)

        # Mask values below vmin (make transparent) - important for reflectivity
        mask = values_sub < vmin
        rgba[mask, 3] = 0  # Set alpha to 0 for masked values

        # Convert to uint8 for PNG
        rgba_uint8 = (rgba * 255).astype(np.uint8)

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=4,
            tiles='CartoDB positron',
        )

        # Add image overlay
        from PIL import Image
        import io
        import base64

        # Flip vertically for correct orientation
        img_array = np.flipud(rgba_uint8)
        img = Image.fromarray(img_array, 'RGBA')

        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode()

        # Add as image overlay
        bounds = [[lat_min, lon_min], [lat_max, lon_max]]
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_base64}",
            bounds=bounds,
            opacity=0.7,
            name=field_name,
        ).add_to(m)

        # Prepare data for JavaScript hover lookup
        # Subsample more aggressively for 2D grids to reduce file size and search time
        hover_step = 4 if lats_sub.ndim == 2 else 2
        values_js = values_sub[::hover_step, ::hover_step]

        if lats_sub.ndim == 1:
            lats_js = lats_sub[::hover_step]
            lons_js = lons_sub[::hover_step]
        else:
            lats_js = lats_sub[::hover_step, ::hover_step]
            lons_js = lons_sub[::hover_step, ::hover_step]

        # Convert to lists - show actual values (even negative for reflectivity)
        values_list = np.where(np.isnan(values_js), None, np.round(values_js, 1)).tolist()

        # Handle 2D curvilinear grids (like HRRR Lambert Conformal)
        if lats_js.ndim == 2:
            # Flatten to 1D for simpler JS lookup, store grid dimensions
            grid_shape = lats_js.shape
            lats_flat = lats_js.flatten().tolist()
            lons_flat = lons_js.flatten().tolist()
            values_flat = np.array(values_list).flatten().tolist()
            is_2d_grid = True
        else:
            lats_flat = lats_js.tolist()
            lons_flat = lons_js.tolist()
            values_flat = values_list
            grid_shape = (len(lats_js), len(lons_js))
            is_2d_grid = False

        # Get units
        units = field_config.get('units', '')
        title = field_config.get('title', field_name)

        # Add custom JavaScript for hover display and opacity control
        if is_2d_grid:
            # 2D grid - need full coordinate search
            hover_js = f"""
            <script>
            var weatherData = {{
                lats: {json.dumps(lats_flat)},
                lons: {json.dumps(lons_flat)},
                values: {json.dumps(values_flat)},
                units: "{units}",
                title: "{title}"
            }};

            function findNearestValue(lat, lon) {{
                var lats = weatherData.lats;
                var lons = weatherData.lons;
                var values = weatherData.values;

                // Convert lon to 0-360 if needed (HRRR uses 0-360)
                var lonAdj = lon < 0 ? lon + 360 : lon;

                // Find nearest point in flattened grid
                var minDist = Infinity;
                var bestIdx = -1;
                for (var i = 0; i < lats.length; i++) {{
                    var dLat = lats[i] - lat;
                    var dLon = lons[i] - lonAdj;
                    var dist = dLat*dLat + dLon*dLon;
                    if (dist < minDist) {{
                        minDist = dist;
                        bestIdx = i;
                    }}
                }}

                if (bestIdx >= 0 && bestIdx < values.length) {{
                    return values[bestIdx];
                }}
                return null;
            }}"""
        else:
            # 1D regular grid
            hover_js = f"""
            <script>
            var weatherData = {{
                lats: {json.dumps(lats_flat)},
                lons: {json.dumps(lons_flat)},
                values: {json.dumps(values_flat)},
                units: "{units}",
                title: "{title}"
            }};

            function findNearestValue(lat, lon) {{
                var lats = weatherData.lats;
                var lons = weatherData.lons;
                var values = weatherData.values;

                var latIdx = 0, lonIdx = 0;
                var minLatDiff = Math.abs(lats[0] - lat);
                var minLonDiff = Math.abs(lons[0] - lon);

                for (var i = 1; i < lats.length; i++) {{
                    if (Math.abs(lats[i] - lat) < minLatDiff) {{
                        minLatDiff = Math.abs(lats[i] - lat);
                        latIdx = i;
                    }}
                }}
                for (var i = 1; i < lons.length; i++) {{
                    if (Math.abs(lons[i] - lon) < minLonDiff) {{
                        minLonDiff = Math.abs(lons[i] - lon);
                        lonIdx = i;
                    }}
                }}

                if (latIdx < values.length && lonIdx < values[0].length) {{
                    return values[latIdx][lonIdx];
                }}
                return null;
            }}"""

        hover_js += f"""

        document.addEventListener('DOMContentLoaded', function() {{
            var map = document.querySelector('.folium-map');
            if (!map) return;

            // Create info display
            var info = document.createElement('div');
            info.id = 'weather-info';
            info.style.cssText = 'position:absolute;top:10px;right:10px;z-index:1000;background:white;padding:10px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,0.3);font-family:monospace;min-width:200px;';
            info.innerHTML = '<b>{title}</b><br>Hover over map for values';
            document.body.appendChild(info);

            // Create opacity slider
            var sliderDiv = document.createElement('div');
            sliderDiv.style.cssText = 'position:fixed;bottom:80px;left:10px;z-index:1000;background:white;padding:10px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,0.3);';
            sliderDiv.innerHTML = '<label style="font-weight:bold;">Opacity: <span id="opacity-val">70%</span></label><br>' +
                '<input type="range" id="opacity-slider" min="0" max="100" value="70" style="width:150px;">';
            document.body.appendChild(sliderDiv);

            // Add mousemove and opacity control
            setTimeout(function() {{
                var leafletMap = Object.values(window).find(v => v && v._leaflet_id);
                if (leafletMap && leafletMap.on) {{
                    leafletMap.on('mousemove', function(e) {{
                        var lat = e.latlng.lat;
                        var lon = e.latlng.lng;
                        var val = findNearestValue(lat, lon);
                        var valStr = (val === null) ? 'No data' : val + ' {units}';
                        info.innerHTML = '<b>{title}</b><br>' +
                            'Lat: ' + lat.toFixed(2) + '°<br>' +
                            'Lon: ' + lon.toFixed(2) + '°<br>' +
                            '<b>Value: ' + valStr + '</b>';
                    }});

                    // Find the image overlay and connect opacity slider
                    var slider = document.getElementById('opacity-slider');
                    var opacityVal = document.getElementById('opacity-val');
                    slider.addEventListener('input', function() {{
                        var opacity = this.value / 100;
                        opacityVal.textContent = this.value + '%';
                        // Find all image overlays
                        var overlays = document.querySelectorAll('.leaflet-image-layer');
                        overlays.forEach(function(overlay) {{
                            overlay.style.opacity = opacity;
                        }});
                    }});
                }}
            }}, 500);
        }});
        </script>
        """

        # Add the JavaScript
        m.get_root().html.add_child(folium.Element(hover_js))

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add colorbar legend
        colorbar_html = f"""
        <div style="position:fixed;bottom:30px;left:10px;z-index:1000;background:white;padding:10px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,0.3);">
            <div style="font-weight:bold;margin-bottom:5px;">{title}</div>
            <div style="display:flex;align-items:center;">
                <span>{vmin:.1f}</span>
                <div style="width:150px;height:15px;margin:0 5px;background:linear-gradient(to right,
                    {mcolors.to_hex(cmap(0.0))},
                    {mcolors.to_hex(cmap(0.25))},
                    {mcolors.to_hex(cmap(0.5))},
                    {mcolors.to_hex(cmap(0.75))},
                    {mcolors.to_hex(cmap(1.0))});"></div>
                <span>{vmax:.1f}</span>
                <span style="margin-left:5px;">{units}</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(colorbar_html))

        # Add title
        title_html = f"""
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:1000;background:white;padding:10px 20px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,0.3);font-size:16px;font-weight:bold;">
            {title} - {cycle} F{forecast_hour:02d}
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{field_name}_f{forecast_hour:02d}_interactive.html"
        m.save(str(output_path))

        return output_path

    except Exception as e:
        print(f"Error creating interactive map: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_create_interactive_maps(
    data_dict: Dict[str, Any],  # field_name -> (data, field_config)
    cycle: str,
    forecast_hour: int,
    output_dir: Path,
) -> Dict[str, Path]:
    """Create interactive maps for multiple fields.

    Returns dict mapping field_name -> output_path
    """
    results = {}
    for field_name, (data, field_config) in data_dict.items():
        colormap = field_config.get('colormap', 'viridis')
        path = create_interactive_map(
            data, field_name, field_config,
            cycle, forecast_hour, output_dir, colormap
        )
        if path:
            results[field_name] = path
    return results
