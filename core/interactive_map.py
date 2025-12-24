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
        cmap = plt.get_cmap(colormap)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Normalize values and apply colormap
        values_norm = norm(values_sub)
        rgba = cmap(values_norm)

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
        # Subsample for the data array - ~265x450 grid for hover (reasonable file size)
        data_step = 2
        values_js = values_sub[::data_step, ::data_step]

        if lats_sub.ndim == 1:
            lats_js = lats_sub[::data_step]
            lons_js = lons_sub[::data_step]
        else:
            lats_js = lats_sub[::data_step, ::data_step]
            lons_js = lons_sub[::data_step, ::data_step]

        # Convert to lists, handling NaN values
        values_list = np.where(np.isnan(values_js), None, np.round(values_js, 2)).tolist()

        if lats_js.ndim == 1:
            lats_list = lats_js.tolist()
            lons_list = lons_js.tolist()
        else:
            lats_list = lats_js[:, 0].tolist()  # Get 1D array for regular grid
            lons_list = lons_js[0, :].tolist()

        # Get units
        units = field_config.get('units', '')
        title = field_config.get('title', field_name)

        # Add custom JavaScript for hover display and opacity control
        hover_js = f"""
        <script>
        var weatherData = {{
            values: {json.dumps(values_list)},
            lats: {json.dumps(lats_list)},
            lons: {json.dumps(lons_list)},
            units: "{units}",
            title: "{title}"
        }};

        function findNearestValue(lat, lon) {{
            var lats = weatherData.lats;
            var lons = weatherData.lons;
            var values = weatherData.values;

            // Find nearest lat index
            var latIdx = 0;
            var minLatDiff = Math.abs(lats[0] - lat);
            for (var i = 1; i < lats.length; i++) {{
                var diff = Math.abs(lats[i] - lat);
                if (diff < minLatDiff) {{
                    minLatDiff = diff;
                    latIdx = i;
                }}
            }}

            // Find nearest lon index
            var lonIdx = 0;
            var minLonDiff = Math.abs(lons[0] - lon);
            for (var i = 1; i < lons.length; i++) {{
                var diff = Math.abs(lons[i] - lon);
                if (diff < minLonDiff) {{
                    minLonDiff = diff;
                    lonIdx = i;
                }}
            }}

            // Get value
            if (latIdx < values.length && lonIdx < values[0].length) {{
                return values[latIdx][lonIdx];
            }}
            return null;
        }}

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
                        if (val !== null) {{
                            info.innerHTML = '<b>{title}</b><br>' +
                                'Lat: ' + lat.toFixed(2) + '°<br>' +
                                'Lon: ' + lon.toFixed(2) + '°<br>' +
                                '<b>Value: ' + val + ' {units}</b>';
                        }}
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
