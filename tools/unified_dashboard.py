#!/usr/bin/env python3
"""
HRRR Cross-Section Dashboard

Interactive cross-section visualization on a Leaflet map.
Draw a line on the map to generate vertical cross-sections.

Usage:
    python tools/unified_dashboard.py --data-dir outputs/hrrr/20251227/09z
    python tools/unified_dashboard.py --auto-update
"""

import argparse
import json
import logging
import sys
import time
import io
import threading
from pathlib import Path
from datetime import datetime
from functools import wraps
from collections import defaultdict

from flask import Flask, jsonify, request, send_file, abort

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONUS_BOUNDS = {
    'south': 21.14, 'north': 52.62,
    'west': -134.10, 'east': -60.92,
}

XSECT_STYLES = [
    ('wind_speed', 'Wind Speed'),
    ('temp', 'Temperature'),
    ('theta_e', 'Theta-E'),
    ('rh', 'Relative Humidity'),
    ('q', 'Specific Humidity'),
    ('omega', 'Vertical Velocity'),
    ('vorticity', 'Vorticity'),
    ('shear', 'Wind Shear'),
    ('lapse_rate', 'Lapse Rate'),
    ('cloud', 'Cloud Water'),
    ('cloud_total', 'Total Condensate'),
    ('wetbulb', 'Wet-Bulb Temp'),
    ('icing', 'Icing Potential'),
]

# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    def __init__(self, rpm=60, burst=10):
        self.rpm, self.burst = rpm, burst
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, ip):
        now = time.time()
        with self.lock:
            self.requests[ip] = [t for t in self.requests[ip] if t > now - 60]
            if len(self.requests[ip]) >= self.rpm:
                return False
            if len([t for t in self.requests[ip] if t > now - 1]) >= self.burst:
                return False
            self.requests[ip].append(now)
            return True

rate_limiter = RateLimiter()

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if app.config.get('PRODUCTION'):
            if not rate_limiter.is_allowed(request.remote_addr):
                return jsonify({'error': 'Rate limit exceeded'}), 429
        return f(*args, **kwargs)
    return decorated

# =============================================================================
# DATA MANAGER
# =============================================================================

class CrossSectionManager:
    """Manages cross-section data from multiple cycles."""

    # Default forecast hours to load (6-hourly for efficiency)
    FORECAST_HOURS = [0, 6, 12, 18]

    def __init__(self):
        self.xsect = None
        self.cycles = {}  # {cycle_key: {'date': str, 'hour': str, 'fhrs': [int]}}
        self.valid_times = []  # Sorted list of (valid_dt, cycle_key, fhr) tuples
        self.base_dir = Path("outputs/hrrr")

    def find_latest_cycles(self, n_cycles: int = 2):
        """Find the N most recent available cycles."""
        from datetime import datetime

        cycles = []
        if not self.base_dir.exists():
            return cycles

        # Scan for date directories
        for date_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not date_dir.is_dir() or not date_dir.name.isdigit():
                continue
            if len(date_dir.name) != 8:
                continue

            # Scan for hour directories within date
            for hour_dir in sorted(date_dir.iterdir(), reverse=True):
                if not hour_dir.is_dir() or not hour_dir.name.endswith('z'):
                    continue

                hour = hour_dir.name.replace('z', '')
                if not hour.isdigit():
                    continue

                # Check what forecast hours are available
                available_fhrs = []
                for fhr in self.FORECAST_HOURS:
                    fhr_dir = hour_dir / f"F{fhr:02d}"
                    if fhr_dir.exists() and list(fhr_dir.glob("*wrfprs*.grib2")):
                        available_fhrs.append(fhr)

                if available_fhrs:
                    cycles.append({
                        'date': date_dir.name,
                        'hour': hour,
                        'path': str(hour_dir),
                        'available_fhrs': available_fhrs,
                    })

                if len(cycles) >= n_cycles:
                    return cycles

        return cycles

    def load_multi_cycle(self, n_cycles: int = 2, forecast_hours: list = None):
        """Load data from multiple cycles, only specified forecast hours."""
        from datetime import datetime, timedelta
        from core.cross_section_interactive import InteractiveCrossSection

        fhrs_to_load = forecast_hours or self.FORECAST_HOURS
        cycles = self.find_latest_cycles(n_cycles)

        if not cycles:
            logger.error("No HRRR cycles found in outputs/hrrr/")
            return 0

        logger.info(f"Found {len(cycles)} cycles to load")
        for c in cycles:
            logger.info(f"  {c['date']} {c['hour']}Z: F{c['available_fhrs']}")

        # Create cross-section engine
        self.xsect = InteractiveCrossSection(cache_dir='cache/dashboard/xsect')

        # Load each cycle's data
        total_loaded = 0
        self.valid_times = []

        for cycle in cycles:
            cycle_key = f"{cycle['date']}_{cycle['hour']}z"
            init_dt = datetime.strptime(f"{cycle['date']}{cycle['hour']}", "%Y%m%d%H")

            # Set metadata for this cycle
            self.xsect.init_date = cycle['date']
            self.xsect.init_hour = cycle['hour']

            # Load only the forecast hours we want
            run_path = Path(cycle['path'])
            for fhr in fhrs_to_load:
                if fhr not in cycle['available_fhrs']:
                    continue

                fhr_dir = run_path / f"F{fhr:02d}"
                prs_files = list(fhr_dir.glob("*wrfprs*.grib2"))
                if not prs_files:
                    continue

                # Load this forecast hour
                if self.xsect.load_forecast_hour(str(prs_files[0]), fhr):
                    valid_dt = init_dt + timedelta(hours=fhr)
                    self.valid_times.append({
                        'valid_dt': valid_dt,
                        'valid_str': valid_dt.strftime("%Y-%m-%d %HZ"),
                        'init_dt': init_dt,
                        'init_str': init_dt.strftime("%Y-%m-%d %HZ"),
                        'cycle_key': cycle_key,
                        'fhr': fhr,
                    })
                    total_loaded += 1

            self.cycles[cycle_key] = cycle

        # Sort by valid time
        self.valid_times.sort(key=lambda x: x['valid_dt'])

        logger.info(f"Loaded {total_loaded} forecast hours across {len(cycles)} cycles")
        logger.info(f"Valid times: {[v['valid_str'] for v in self.valid_times]}")

        return total_loaded

    def load_run(self, data_dir: str, max_hours: int = 6):
        """Load HRRR run data for cross-sections (legacy single-cycle mode)."""
        from core.cross_section_interactive import InteractiveCrossSection

        data_path = Path(data_dir).resolve()
        logger.info(f"Loading data from {data_path}...")

        # Parse cycle info from path
        import re
        path_str = str(data_path)
        date_match = re.search(r'/(\d{8})/(\d{2})z', path_str)
        if date_match:
            init_date = date_match.group(1)
            init_hour = date_match.group(2)
        else:
            init_date = None
            init_hour = None

        # Load cross-section engine
        self.xsect = InteractiveCrossSection(cache_dir='cache/dashboard/xsect')
        loaded = self.xsect.load_run(str(data_path), max_hours=max_hours)

        # Build valid_times list for compatibility
        from datetime import datetime, timedelta
        if init_date and init_hour:
            init_dt = datetime.strptime(f"{init_date}{init_hour}", "%Y%m%d%H")
            for fhr in self.xsect.get_loaded_hours():
                valid_dt = init_dt + timedelta(hours=fhr)
                self.valid_times.append({
                    'valid_dt': valid_dt,
                    'valid_str': valid_dt.strftime("%Y-%m-%d %HZ"),
                    'init_dt': init_dt,
                    'init_str': init_dt.strftime("%Y-%m-%d %HZ"),
                    'cycle_key': f"{init_date}_{init_hour}z",
                    'fhr': fhr,
                })
            self.valid_times.sort(key=lambda x: x['valid_dt'])

        logger.info(f"Loaded {loaded} hours for cross-sections")
        return loaded

    def get_available_times(self):
        """Return list of available valid times for the UI."""
        return [
            {
                'valid': v['valid_str'],
                'init': v['init_str'],
                'fhr': v['fhr'],
                'index': i,
            }
            for i, v in enumerate(self.valid_times)
        ]

    def generate_cross_section(self, start, end, time_index, style):
        """Generate a cross-section image for a given valid time index."""
        if not self.xsect:
            return None

        # Get the forecast hour and metadata for this valid time
        if self.valid_times and isinstance(time_index, int) and 0 <= time_index < len(self.valid_times):
            vt = self.valid_times[time_index]
            fhr = vt['fhr']
            # Update xsect metadata to match this cycle
            cycle_parts = vt['cycle_key'].split('_')
            self.xsect.init_date = cycle_parts[0]
            self.xsect.init_hour = cycle_parts[1].replace('z', '')
        else:
            # Legacy: treat as direct forecast hour
            fhr = int(time_index) if time_index else 0

        try:
            png_bytes = self.xsect.get_cross_section(
                start_point=start,
                end_point=end,
                forecast_hour=fhr,
                style=style,
                return_image=True,
                dpi=100
            )
            if png_bytes is None:
                return None

            return io.BytesIO(png_bytes)
        except Exception as e:
            logger.error(f"Cross-section error: {e}")
            return None

data_manager = CrossSectionManager()

# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HRRR Cross-Section Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
            --bg: #0f172a;
            --panel: #1e293b;
            --card: #334155;
            --text: #f1f5f9;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --border: #475569;
        }
        body {
            font-family: system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
        }
        #map-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        #map { flex: 1; }
        #controls {
            background: var(--panel);
            padding: 12px;
            display: flex;
            gap: 16px;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }
        .control-group { display: flex; align-items: center; gap: 8px; }
        label { color: var(--muted); font-size: 13px; }
        select, button {
            background: var(--card);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover { background: var(--accent); color: #000; }
        #cycle-info {
            margin-left: auto;
            color: var(--muted);
            font-size: 13px;
        }
        #sidebar {
            width: 55%;
            min-width: 500px;
            background: var(--panel);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        #xsect-header {
            padding: 12px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
        }
        #xsect-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px;
            overflow: hidden;
        }
        #xsect-img {
            max-width: 100%;
            max-height: 100%;
            border-radius: 4px;
        }
        #instructions {
            color: var(--muted);
            text-align: center;
            padding: 20px;
        }
        .loading {
            color: var(--accent);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div id="map-container">
        <div id="controls">
            <div class="control-group">
                <label>Time:</label>
                <select id="hour-select"></select>
            </div>
            <div class="control-group">
                <label>Style:</label>
                <select id="style-select"></select>
            </div>
            <button id="clear-btn">Clear Line</button>
            <div id="cycle-info"></div>
        </div>
        <div id="map"></div>
    </div>
    <div id="sidebar">
        <div id="xsect-header">Cross-Section</div>
        <div id="xsect-container">
            <div id="instructions">
                Click two points on the map to draw a cross-section line
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const bounds = ''' + json.dumps(CONUS_BOUNDS) + ''';
        const styles = ''' + json.dumps(XSECT_STYLES) + ''';

        // Initialize map
        const map = L.map('map', {
            center: [39, -98],
            zoom: 5,
            minZoom: 4,
            maxZoom: 10
        });

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap, &copy; CARTO'
        }).addTo(map);

        // State
        let startMarker = null, endMarker = null, line = null;
        let availableHours = [];

        // Populate style selector
        const styleSelect = document.getElementById('style-select');
        styles.forEach(([val, label]) => {
            const opt = document.createElement('option');
            opt.value = val;
            opt.textContent = label;
            styleSelect.appendChild(opt);
        });

        // Load initial data
        let availableTimes = [];
        fetch('/api/info')
            .then(r => r.json())
            .then(data => {
                availableTimes = data.times || [];
                const hourSelect = document.getElementById('hour-select');
                hourSelect.innerHTML = '';

                if (availableTimes.length > 0) {
                    // New multi-cycle mode: show valid times
                    availableTimes.forEach((t, idx) => {
                        const opt = document.createElement('option');
                        opt.value = idx;  // Use index as value
                        // Show: "Valid 2025-12-28 18Z (F06 from 12Z)"
                        const validShort = t.valid.split(' ')[1];  // Just the time part
                        const initShort = t.init.split(' ')[1];
                        opt.textContent = `${validShort} (F${String(t.fhr).padStart(2,'0')} ${initShort})`;
                        hourSelect.appendChild(opt);
                    });
                } else {
                    // Legacy mode: use hours directly
                    availableHours = data.hours || [];
                    availableHours.forEach(h => {
                        const opt = document.createElement('option');
                        opt.value = h;
                        opt.textContent = 'F' + String(h).padStart(2, '0');
                        hourSelect.appendChild(opt);
                    });
                }

                // Show cycle info
                if (data.cycles && data.cycles.length > 0) {
                    document.getElementById('cycle-info').textContent =
                        `Cycles: ${data.cycles.join(', ')}`;
                } else if (data.cycle) {
                    document.getElementById('cycle-info').textContent =
                        `${data.cycle.model.toUpperCase()} ${data.cycle.date} ${data.cycle.hour}Z`;
                }
            });

        // Map click handler
        map.on('click', e => {
            const {lat, lng} = e.latlng;

            if (!startMarker) {
                startMarker = L.marker([lat, lng], {
                    draggable: true,
                    icon: L.divIcon({
                        className: 'marker-start',
                        html: '<div style="width:16px;height:16px;background:#38bdf8;border-radius:50%;border:2px solid white;"></div>',
                        iconSize: [16, 16],
                        iconAnchor: [8, 8]
                    })
                }).addTo(map);
                startMarker.on('drag', updateLine);
                startMarker.on('dragend', generateCrossSection);
            } else if (!endMarker) {
                endMarker = L.marker([lat, lng], {
                    draggable: true,
                    icon: L.divIcon({
                        className: 'marker-end',
                        html: '<div style="width:16px;height:16px;background:#f87171;border-radius:50%;border:2px solid white;"></div>',
                        iconSize: [16, 16],
                        iconAnchor: [8, 8]
                    })
                }).addTo(map);
                endMarker.on('drag', updateLine);
                endMarker.on('dragend', generateCrossSection);

                line = L.polyline([startMarker.getLatLng(), endMarker.getLatLng()], {
                    color: '#fbbf24', weight: 3, dashArray: '10, 5'
                }).addTo(map);

                generateCrossSection();
            }
        });

        function updateLine() {
            if (startMarker && endMarker && line) {
                line.setLatLngs([startMarker.getLatLng(), endMarker.getLatLng()]);
            }
        }

        // Generate cross-section
        function generateCrossSection() {
            if (!startMarker || !endMarker) return;

            const container = document.getElementById('xsect-container');
            container.innerHTML = '<div class="loading">Generating cross-section...</div>';

            const start = startMarker.getLatLng();
            const end = endMarker.getLatLng();
            const timeIndex = document.getElementById('hour-select').value;
            const style = document.getElementById('style-select').value;

            const url = `/api/xsect?start_lat=${start.lat}&start_lon=${start.lng}` +
                `&end_lat=${end.lat}&end_lon=${end.lng}&time_index=${timeIndex}&style=${style}`;

            fetch(url)
                .then(r => {
                    if (!r.ok) throw new Error('Failed to generate');
                    return r.blob();
                })
                .then(blob => {
                    const img = document.createElement('img');
                    img.id = 'xsect-img';
                    img.src = URL.createObjectURL(blob);
                    container.innerHTML = '';
                    container.appendChild(img);
                })
                .catch(err => {
                    container.innerHTML = `<div style="color:#f87171">${err.message}</div>`;
                });
        }

        // Clear button
        document.getElementById('clear-btn').onclick = () => {
            if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
            if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
            if (line) { map.removeLayer(line); line = null; }
            document.getElementById('xsect-container').innerHTML =
                '<div id="instructions">Click two points on the map to draw a cross-section line</div>';
        };

        // Regenerate on option change
        document.getElementById('hour-select').onchange = generateCrossSection;
        document.getElementById('style-select').onchange = generateCrossSection;
    </script>
</body>
</html>'''

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/api/info')
def api_info():
    # Return valid times for the slider
    times = data_manager.get_available_times()
    return jsonify({
        'times': times,  # New: list of {valid, init, fhr, index}
        'hours': [t['fhr'] for t in times],  # Legacy compatibility
        'cycles': list(data_manager.cycles.keys()),
        'styles': XSECT_STYLES,
    })

@app.route('/api/xsect')
@rate_limit
def api_xsect():
    try:
        start = (float(request.args['start_lat']), float(request.args['start_lon']))
        end = (float(request.args['end_lat']), float(request.args['end_lon']))
        # Support both time_index (new) and hour (legacy)
        time_index = request.args.get('time_index', request.args.get('hour', 0))
        time_index = int(time_index)
        style = request.args.get('style', 'wind_speed')
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    buf = data_manager.generate_cross_section(start, end, time_index, style)
    if buf is None:
        return jsonify({'error': 'Failed to generate cross-section'}), 500

    return send_file(buf, mimetype='image/png')

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HRRR Cross-Section Dashboard')
    parser.add_argument('--data-dir', type=str, help='Path to HRRR run data (single cycle)')
    parser.add_argument('--auto-update', action='store_true', help='Auto-download latest')
    parser.add_argument('--multi-cycle', action='store_true',
                       help='Load F00,F06,F12,F18 from latest 2 cycles (default mode)')
    parser.add_argument('--n-cycles', type=int, default=2, help='Number of cycles to load')
    parser.add_argument('--max-hours', type=int, default=18, help='Max forecast hour to load')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--production', action='store_true', help='Enable rate limiting')

    args = parser.parse_args()
    app.config['PRODUCTION'] = args.production

    # Determine loading mode
    if args.auto_update:
        # Download latest data, then use multi-cycle loading
        from smart_hrrr.orchestrator import download_latest_cycle

        logger.info("Downloading latest HRRR data...")
        # Download F00, F06, F12, F18 only
        fhrs_to_download = [0, 6, 12, 18]
        fhrs_to_download = [f for f in fhrs_to_download if f <= args.max_hours]

        date_str, hour, results = download_latest_cycle(
            max_hours=max(fhrs_to_download),
            forecast_hours=fhrs_to_download
        )
        if not date_str:
            logger.error("Failed to download data")
            sys.exit(1)

        # Now load with multi-cycle mode
        args.multi_cycle = True

    if args.multi_cycle or not args.data_dir:
        # Multi-cycle mode: load F00,F06,F12,F18 from latest N cycles
        logger.info(f"Loading {args.n_cycles} cycles (F00, F06, F12, F18 each)...")
        fhrs = [f for f in [0, 6, 12, 18] if f <= args.max_hours]
        loaded = data_manager.load_multi_cycle(
            n_cycles=args.n_cycles,
            forecast_hours=fhrs
        )
        if loaded == 0:
            logger.error("No data loaded. Run with --auto-update to download.")
            sys.exit(1)
    else:
        # Single cycle mode (legacy)
        data_manager.load_run(args.data_dir, max_hours=args.max_hours)

    logger.info("=" * 60)
    logger.info("HRRR Cross-Section Dashboard")
    logger.info(f"Open: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()
