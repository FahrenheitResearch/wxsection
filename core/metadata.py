"""Metadata generation and handling module"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path


def save_metadata_json(field_name, field_config, cycle_dt, valid_dt, forecast_hour, data, output_dir, current_region, model_name="hrrr"):
    """Save comprehensive metadata as JSON file"""
    # Create metadata directory parallel to image directory
    metadata_dir = output_dir.parent / 'metadata' / output_dir.name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Build metadata dictionary
    metadata = {
        "model": {
            "name": "High-Resolution Rapid Refresh (HRRR)" if model_name == "hrrr" else model_name.upper(),
            "version": "v4",
            "resolution": "3km"
        },
        "timing": {
            "init_time": cycle_dt.strftime('%Y-%m-%d %H:%M UTC'),
            "valid_time": valid_dt.strftime('%Y-%m-%d %H:%M UTC'),
            "forecast_hour": forecast_hour,
            "forecast_hour_str": f"F{forecast_hour:02d}"
        },
        "parameter": {
            "name": field_name,
            "title": field_config.get('title', 'Unknown'),
            "units": field_config.get('units', 'dimensionless'),
            "category": field_config.get('category', 'general'),
            "type": "derived" if field_config.get('derived') else "direct",
            "description": field_config.get('description', '')
        },
        "visualization": {
            "colormap": field_config.get('cmap', 'default'),
            "levels": field_config.get('levels', []),
            "extend": field_config.get('extend', 'neither'),
            "plot_style": field_config.get('plot_style', 'filled')
        },
        "processing": {
            "generated_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "processor_version": "2.0",
            "region": "conus"
        }
    }
    
    # Add derived parameter details if applicable
    if field_config.get('derived'):
        metadata["parameter"]["derived_details"] = {
            "function": field_config.get('function', 'unknown'),
            "inputs": field_config.get('inputs', []),
            "formula": _extract_formula(field_config.get('function'))
        }
    
    # Add data statistics if available
    if data is not None:
        try:
            valid_data = data.values[~np.isnan(data.values)]
            if len(valid_data) > 0:
                metadata["data_statistics"] = {
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "mean": float(valid_data.mean()),
                    "std": float(valid_data.std()),
                    "grid_shape": list(data.shape),
                    "valid_points": int(len(valid_data)),
                    "total_points": int(data.size)
                }
        except:
            pass
    
    # Save metadata JSON
    metadata_file = metadata_dir / f"{field_name}_f{forecast_hour:02d}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def create_info_panel(field_config, cycle_dt, valid_dt, forecast_hour, data):
    """Create comprehensive information panel text (kept for backward compatibility)"""
    
    # Basic timing info
    info_lines = [
        "═══ HRRR METADATA ═══",
        f"Model: High-Resolution Rapid Refresh",
        f"Init: {cycle_dt.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Valid: {valid_dt.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Forecast Hour: F{forecast_hour:02d}",
        "",
        "═══ PARAMETER INFO ═══",
        f"Name: {field_config.get('title', 'Unknown')}",
        f"Units: {field_config.get('units', 'dimensionless')}",
        f"Category: {field_config.get('category', 'general').title()}",
    ]
    
    # Add derived parameter info if applicable
    if field_config.get('derived'):
        info_lines.extend([
            f"Type: Derived Parameter",
            f"Function: {field_config.get('function', 'unknown')}",
        ])
        if field_config.get('inputs'):
            inputs_str = ', '.join(field_config['inputs'])
            info_lines.append(f"Inputs: {inputs_str}")
        
        # Add actual calculation formula
        formula = _extract_formula(field_config.get('function'))
        if formula:
            info_lines.extend([
                "",
                "═══ CALCULATION ═══",
                f"{formula}"
            ])
    else:
        info_lines.append("Type: Direct GRIB Field")
    
    # Add description if available
    if field_config.get('description'):
        desc = field_config['description']
        # Wrap long descriptions
        if len(desc) > 40:
            desc = desc[:37] + "..."
        info_lines.extend(["", f"Description:", f"{desc}"])
    
    # Data statistics
    if data is not None:
        try:
            valid_data = data.values[~np.isnan(data.values)]
            if len(valid_data) > 0:
                info_lines.extend([
                    "",
                    "═══ DATA STATS ═══",
                    f"Min: {valid_data.min():.2f}",
                    f"Max: {valid_data.max():.2f}",
                    f"Mean: {valid_data.mean():.2f}",
                    f"Grid: {data.shape[0]}×{data.shape[1]}",
                ])
        except:
            pass
    
    # Color scale info
    if field_config.get('levels'):
        levels = field_config['levels']
        info_lines.extend([
            "",
            "═══ COLOR SCALE ═══",
            f"Levels: {len(levels)} intervals",
            f"Range: {levels[0]} to {levels[-1]}",
            f"Colormap: {field_config.get('cmap', 'default')}",
        ])
    
    # Add processing timestamp
    info_lines.extend([
        "",
        "═══ PROCESSING ═══",
        f"Generated: {datetime.utcnow().strftime('%H:%M:%S UTC')}",
        f"System: HRRR Processor v2",
    ])
    
    return '\n'.join(info_lines)


def _extract_formula(function_name):
    """Extract the main calculation formula from a derived parameter function"""
    if not function_name:
        return None
        
    try:
        from derived_params import DerivedParameters
        import inspect
        
        # Get the function
        if hasattr(DerivedParameters, function_name):
            func = getattr(DerivedParameters, function_name)
            source = inspect.getsource(func)
            
            # Extract key calculation lines
            formula = _parse_calculation_from_source(source, function_name)
            return formula
            
    except Exception as e:
        return f"Formula extraction error: {str(e)}"
    
    return None


def _parse_calculation_from_source(source, function_name):
    """Parse source code to extract the main calculation"""
    import re
    
    # Known formula patterns for common calculations
    formulas = {
        'wbgt_shade': 'WBGT = 0.7 × WB + 0.3 × DB',
        'wbgt_estimated_outdoor': 'WBGT = 0.7 × WB + 0.2 × BG + 0.1 × DB\nBG = DB + solar - wind_cooling',
        'wbgt_simplified_outdoor': 'WBGT = 0.7 × WB + 0.2 × BG + 0.1 × DB\nBG = DB + 2°C - wind_cooling',
        'wet_bulb_temperature': 'Wet Bulb = f(T, Td, P)\nBisection method with fallback approximation',
        'wet_bulb_temperature_metpy': 'Wet Bulb = f(T, Td, P)\nBisection method with fallback approximation',
        'wind_speed_10m': 'Speed = √(u² + v²)',
        'mixing_ratio_2m': 'MR = 0.622 × es / (P - es)\nes = 6.1094 × exp(17.625×Td/(Td+243.04))',
        'supercell_composite_parameter': 'SCP = (muCAPE/1000) × (ESRH/50) × clip((EBWD-10)/10, 0, 1) × CIN_term\nCIN_term = 1 if muCIN > -40, else -40/muCIN',
        'significant_tornado_parameter': 'STP = (MLCAPE/1500) × (2000-LCL)/1000\n    × (SRH/150) × (Shear/20)',
        'energy_helicity_index': 'EHI = (CAPE × SRH) / 160000',
        'bulk_richardson_number': 'BRN = CAPE / (0.5 × Shear²)',
        'crosswind_component': 'Crosswind = u × sin(θ) + v × cos(θ)\nwhere θ = reference_direction',
        'fire_weather_index': 'FWI = f(T, RH, WindSpeed)\nCombines temperature, humidity, wind',
        'wind_shear_magnitude': 'Shear = √(u_shear² + v_shear²)',
        'ventilation_rate_from_components': 'VR = WindSpeed × PBL_Height\nWindSpeed = √(u² + v²)',
        'effective_srh': 'Effective SRH = SRH × (CAPE/2500)\nwith LCL and CIN adjustments',
        'craven_brooks_composite': 'CBC = √((CAPE/2500) × (SRH/150) × (Shear/20))',
        'modified_stp_effective': 'Modified STP = (MLCAPE/1500) × (2000-LCL)/1000\n× (SRH/150) × (Shear/20) × CIN_factor',
        'surface_richardson_number': 'Ri = (g/T) × (dT/dz) / (du/dz)²\nStability parameter',
        'cross_totals': 'CT = Dewpoint_850 - Temperature_500\nInstability index',
        'violent_tornado_parameter': 'VTP = (MLCAPE/1500) × (EBWD/20) × (ESRH/150) × ((2000-MLLCL)/1000)\n    × ((200+MLCIN)/150) × (0-3km CAPE/50) × (0-3km Lapse/6.5)',
        'significant_tornado_parameter_cin': 'STP-CIN = (MLCAPE/1500) × (ESRH/150) × (EBWD/12)\n    × ((2000-MLLCL)/1000) × ((MLCIN+200)/150)'
    }
    
    # Return known formula if available
    if function_name in formulas:
        return formulas[function_name]
    
    # Try to extract from source code patterns
    lines = source.split('\n')
    calculation_lines = []
    
    # Look for key calculation patterns
    for line in lines:
        line = line.strip()
        
        # Skip comments and docstrings
        if line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
            continue
            
        # Look for return statements with calculations
        if 'return ' in line and any(op in line for op in ['+', '-', '*', '/', '**', 'np.']):
            # Clean up the return statement
            formula_line = line.replace('return ', '').strip()
            # Remove numpy prefixes for readability
            formula_line = re.sub(r'np\.', '', formula_line)
            calculation_lines.append(f"= {formula_line}")
            
        # Look for key assignment lines with mathematical operations
        elif any(op in line for op in ['=', '+', '-', '*', '/', '**']) and any(keyword in line.lower() for keyword in ['temp', 'cape', 'shear', 'wind', 'wbgt', 'wet_bulb', 'ratio']):
            # Skip simple assignments like variable declarations
            if '=' in line and any(op in line for op in ['+', '-', '*', '/', '**']):
                # Clean up the line
                clean_line = re.sub(r'^\s*\w+\s*=\s*', '', line)
                clean_line = re.sub(r'np\.', '', clean_line)
                if len(clean_line) < 100:  # Only include reasonably short lines
                    calculation_lines.append(clean_line)
    
    # Return the most relevant calculation lines
    if calculation_lines:
        return '\n'.join(calculation_lines[:3])  # Limit to 3 lines max
    
    # Fallback: return a generic description
    return f"Complex calculation in {function_name}()\nSee source code for details"