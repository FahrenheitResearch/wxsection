# derived_params/__init__.py
"""
HRRR Derived Parameters Package

This package provides computation functions for meteorological derived parameters
used in severe weather analysis and atmospheric research.
"""

from typing import Dict, Any
import numpy as np

# Import all the calculation functions
from .common import *
from .supercell_composite_parameter import supercell_composite_parameter
from .supercell_composite_parameter_modified import supercell_composite_parameter_modified
from .significant_tornado_parameter import significant_tornado_parameter
from .significant_tornado_parameter_fixed import significant_tornado_parameter_fixed
from .significant_tornado_parameter_fixed_modified import significant_tornado_parameter_fixed_modified
from .significant_tornado_parameter_fixed_no_cin import significant_tornado_parameter_fixed_no_cin
from .significant_tornado_parameter_effective import significant_tornado_parameter_effective
from .energy_helicity_index import energy_helicity_index
from .energy_helicity_index_display import energy_helicity_index_display
from .energy_helicity_index_01km import energy_helicity_index_01km
from .wind_shear_magnitude import wind_shear_magnitude
from .updraft_helicity_threshold import updraft_helicity_threshold
from .composite_severe_index import composite_severe_index
from .wet_bulb_temperature import wet_bulb_temperature
from .wet_bulb_temperature_metpy import wet_bulb_temperature_metpy
from ._wet_bulb_approximation import _wet_bulb_approximation
from .wbgt_shade import wbgt_shade
from .wbgt_estimated_outdoor import wbgt_estimated_outdoor
from .wbgt_simplified_outdoor import wbgt_simplified_outdoor
from .mixing_ratio_2m import mixing_ratio_2m
from ._mixing_ratio_approximation import _mixing_ratio_approximation
from .wind_speed_10m import wind_speed_10m
from .crosswind_component import crosswind_component
from .wind_shear_vector_01km import wind_shear_vector_01km
from .wind_shear_vector_06km import wind_shear_vector_06km
from .shear_vector_magnitude_ratio import shear_vector_magnitude_ratio
from .sweat_index import sweat_index
from .cross_totals import cross_totals
from .effective_srh import effective_srh
from .bulk_richardson_number import bulk_richardson_number
from .craven_brooks_composite import craven_brooks_composite
from .modified_stp_effective import modified_stp_effective
from .right_mover_supercell_composite import right_mover_supercell_composite
from .supercell_strength_index import supercell_strength_index
from .mesocyclone_strength_parameter import mesocyclone_strength_parameter
from .surface_richardson_number import surface_richardson_number
from .monin_obukhov_length import monin_obukhov_length
from .convective_velocity_scale import convective_velocity_scale
from .turbulent_kinetic_energy_estimate import turbulent_kinetic_energy_estimate
from .haines_index import haines_index
from .ventilation_rate import ventilation_rate
from .enhanced_smoke_dispersion_index import enhanced_smoke_dispersion_index
from .ventilation_rate_from_components import ventilation_rate_from_components, ventilation_rate_from_surface_winds
from .effective_layer_detection import detect_effective_inflow_layer, compute_effective_layer_wind_shear, compute_effective_layer_srh
from .enhanced_smoke_dispersion_index_simplified import enhanced_smoke_dispersion_index_simplified
from .shear_vector_magnitude_ratio_from_components import shear_vector_magnitude_ratio_from_components
from .enhanced_smoke_dispersion_index_from_components import enhanced_smoke_dispersion_index_from_components
from .crude_lcl_estimate import crude_lcl_estimate
from .absolute_vorticity_500_estimate import absolute_vorticity_500_estimate
from ._calculate_saturation_vapor_pressure import _calculate_saturation_vapor_pressure
from ._calculate_virtual_temperature import _calculate_virtual_temperature
from ._find_lcl_bolton import _find_lcl_bolton
from ._moist_adiabatic_temperature import _moist_adiabatic_temperature
from .surface_based_cape_and_cin import surface_based_cape_and_cin
from .mixed_layer_cape_and_cin import mixed_layer_cape_and_cin
from .most_unstable_cape_and_cin import most_unstable_cape_and_cin
from .calculate_surface_based_cape import calculate_surface_based_cape
from .calculate_surface_based_cin import calculate_surface_based_cin
from .calculate_mixed_layer_cape import calculate_mixed_layer_cape
from .calculate_mixed_layer_cin import calculate_mixed_layer_cin
from .calculate_most_unstable_cape import calculate_most_unstable_cape
from .lifted_index import lifted_index
from .showalter_index import showalter_index
from .significant_hail_parameter import significant_hail_parameter
from .craven_significant_severe import craven_significant_severe
from .vorticity_generation_parameter import vorticity_generation_parameter
from .supercell_composite_parameter_effective import supercell_composite_parameter_effective
from .cape_03km import cape_03km
from .lapse_rate_03km import lapse_rate_03km
from .effective_shear import effective_shear
from .significant_tornado_parameter_cin import significant_tornado_parameter_cin
from .violent_tornado_parameter import violent_tornado_parameter
from .calculate_lapse_rate_700_500 import calculate_lapse_rate_700_500

# Diurnal temperature functions
from .diurnal_temperature import (
    diurnal_temperature_range,
    diurnal_max_temperature,
    diurnal_min_temperature,
    diurnal_mean_temperature,
    day_night_temperature_difference,
    temperature_departure_from_mean,
    heating_rate,
    cooling_rate,
    hour_of_maximum_temperature,
    hour_of_minimum_temperature,
    diurnal_temperature_amplitude,
    temperature_at_hour,
    compute_all_diurnal_products
)

# Create a dispatch table of all available functions
_DERIVED_FUNCTIONS = {
    'supercell_composite_parameter': supercell_composite_parameter,
    'supercell_composite_parameter_modified': supercell_composite_parameter_modified,
    'significant_tornado_parameter': significant_tornado_parameter,
    'significant_tornado_parameter_fixed': significant_tornado_parameter_fixed,
    'significant_tornado_parameter_fixed_modified': significant_tornado_parameter_fixed_modified,
    'significant_tornado_parameter_fixed_no_cin': significant_tornado_parameter_fixed_no_cin,
    'significant_tornado_parameter_effective': significant_tornado_parameter_effective,
    'energy_helicity_index': energy_helicity_index,
    'energy_helicity_index_display': energy_helicity_index_display,
    'energy_helicity_index_01km': energy_helicity_index_01km,
    'wind_shear_magnitude': wind_shear_magnitude,
    'updraft_helicity_threshold': updraft_helicity_threshold,
    'composite_severe_index': composite_severe_index,
    'wet_bulb_temperature': wet_bulb_temperature,
    'wet_bulb_temperature_metpy': wet_bulb_temperature,  # legacy alias
    '_wet_bulb_approximation': _wet_bulb_approximation,
    'wbgt_shade': wbgt_shade,
    'wbgt_estimated_outdoor': wbgt_estimated_outdoor,
    'wbgt_simplified_outdoor': wbgt_simplified_outdoor,
    'mixing_ratio_2m': mixing_ratio_2m,
    '_mixing_ratio_approximation': _mixing_ratio_approximation,
    'wind_speed_10m': wind_speed_10m,
    'crosswind_component': crosswind_component,
    'wind_shear_vector_01km': wind_shear_vector_01km,
    'wind_shear_vector_06km': wind_shear_vector_06km,
    'shear_vector_magnitude_ratio': shear_vector_magnitude_ratio,
    'sweat_index': sweat_index,
    'cross_totals': cross_totals,
    'effective_srh': effective_srh,
    'bulk_richardson_number': bulk_richardson_number,
    'craven_brooks_composite': craven_brooks_composite,
    'modified_stp_effective': modified_stp_effective,
    'right_mover_supercell_composite': right_mover_supercell_composite,
    'supercell_strength_index': supercell_strength_index,
    'mesocyclone_strength_parameter': mesocyclone_strength_parameter,
    'surface_richardson_number': surface_richardson_number,
    'monin_obukhov_length': monin_obukhov_length,
    'convective_velocity_scale': convective_velocity_scale,
    'turbulent_kinetic_energy_estimate': turbulent_kinetic_energy_estimate,
    'haines_index': haines_index,
    'ventilation_rate': ventilation_rate,
    'enhanced_smoke_dispersion_index': enhanced_smoke_dispersion_index,
    'ventilation_rate_from_components': ventilation_rate_from_components,
    'ventilation_rate_from_surface_winds': ventilation_rate_from_surface_winds,
    'detect_effective_inflow_layer': detect_effective_inflow_layer,
    'compute_effective_layer_wind_shear': compute_effective_layer_wind_shear,
    'compute_effective_layer_srh': compute_effective_layer_srh,
    'enhanced_smoke_dispersion_index_simplified': enhanced_smoke_dispersion_index_simplified,
    'shear_vector_magnitude_ratio_from_components': shear_vector_magnitude_ratio_from_components,
    'enhanced_smoke_dispersion_index_from_components': enhanced_smoke_dispersion_index_from_components,
    'crude_lcl_estimate': crude_lcl_estimate,
    'absolute_vorticity_500_estimate': absolute_vorticity_500_estimate,
    '_calculate_saturation_vapor_pressure': _calculate_saturation_vapor_pressure,
    '_calculate_virtual_temperature': _calculate_virtual_temperature,
    '_find_lcl_bolton': _find_lcl_bolton,
    '_moist_adiabatic_temperature': _moist_adiabatic_temperature,
    'surface_based_cape_and_cin': surface_based_cape_and_cin,
    'mixed_layer_cape_and_cin': mixed_layer_cape_and_cin,
    'most_unstable_cape_and_cin': most_unstable_cape_and_cin,
    'calculate_surface_based_cape': calculate_surface_based_cape,
    'calculate_surface_based_cin': calculate_surface_based_cin,
    'calculate_mixed_layer_cape': calculate_mixed_layer_cape,
    'calculate_mixed_layer_cin': calculate_mixed_layer_cin,
    'calculate_most_unstable_cape': calculate_most_unstable_cape,
    'lifted_index': lifted_index,
    'showalter_index': showalter_index,
    'significant_hail_parameter': significant_hail_parameter,
    'craven_significant_severe': craven_significant_severe,
    'vorticity_generation_parameter': vorticity_generation_parameter,
    'supercell_composite_parameter_effective': supercell_composite_parameter_effective,
    'cape_03km': cape_03km,
    'lapse_rate_03km': lapse_rate_03km,
    'effective_shear': effective_shear,
    'significant_tornado_parameter_cin': significant_tornado_parameter_cin,
    'violent_tornado_parameter': violent_tornado_parameter,
    'calculate_lapse_rate_700_500': calculate_lapse_rate_700_500,
    'identity': identity,
    # Diurnal temperature functions
    'diurnal_temperature_range': diurnal_temperature_range,
    'diurnal_max_temperature': diurnal_max_temperature,
    'diurnal_min_temperature': diurnal_min_temperature,
    'diurnal_mean_temperature': diurnal_mean_temperature,
    'day_night_temperature_difference': day_night_temperature_difference,
    'temperature_departure_from_mean': temperature_departure_from_mean,
    'heating_rate': heating_rate,
    'cooling_rate': cooling_rate,
    'hour_of_maximum_temperature': hour_of_maximum_temperature,
    'hour_of_minimum_temperature': hour_of_minimum_temperature,
    'diurnal_temperature_amplitude': diurnal_temperature_amplitude,
    'temperature_at_hour': temperature_at_hour,
    'compute_all_diurnal_products': compute_all_diurnal_products,
}


def compute_derived_parameter(param_name: str, input_data: Dict[str, np.ndarray], 
                            config: Dict[str, Any]) -> np.ndarray:
    """
    Compute a specific derived parameter using a modern dispatch approach.
    
    Args:
        param_name: Name of the parameter to compute
        input_data: Dictionary of input arrays
        config: Parameter configuration
        
    Returns:
        Computed parameter array
        
    Raises:
        ValueError: If function not found or required input data missing
    """
    function_name = config['function']
    inputs = config['inputs']
    kwargs = config.get('kwargs', {})
    
    # Get the function from dispatch table
    if function_name not in _DERIVED_FUNCTIONS:
        raise ValueError(f"Function {function_name} not found in derived parameters")
    
    func = _DERIVED_FUNCTIONS[function_name]
    
    # Special handling for SCP - make mucin optional
    if param_name == 'scp':
        # Build args conditionally for SCP
        required_inputs = ['mucape', 'effective_srh', 'effective_shear']
        args = []
        
        # Add required inputs
        for input_name in required_inputs:
            if input_name in input_data:
                args.append(input_data[input_name])
            else:
                raise ValueError(f"Missing required input data for SCP: {input_name}")
        
        # Conditionally add mucin if available
        if 'mucin' in input_data:
            args.append(input_data['mucin'])
            print("🔍 SCP: Using full SPC recipe with MUCIN")
        else:
            print("Warning: SCP - MUCIN not available, using fallback recipe (no CIN penalty)")
        
        # Compute the derived parameter
        result = func(*args, **kwargs)
        
        # Quick sanity check for SCP diagnostics
        if param_name == 'scp':
            # SCP returns a tuple (scp_raw, scp_plot)
            scp_raw, scp_plot = result
            print("🔎  SCP diagnostics — raw max:", np.nanmax(scp_raw),
                  "clip max:", np.nanmax(scp_plot))
            # Expect raw somewhere 15-25 on big days, plot ALWAYS <= 10
            result = scp_raw  # Return the unclipped version - clipping happens in plotting layer
    else:
        # Standard behavior for all other parameters
        args = []
        for input_name in inputs:
            if input_name in input_data:
                args.append(input_data[input_name])
            else:
                raise ValueError(f"Missing input data for {input_name}")
        
        # Compute the derived parameter
        result = func(*args, **kwargs)
    
    return result


def create_derived_config() -> Dict[str, Dict[str, Any]]:
    """
    Create configuration for derived parameters
    
    Returns:
        Dictionary of derived parameter configurations
    """
    config = {
        'scp': {
            'title': 'Supercell Composite Parameter (SPC Standard)',
            'units': 'dimensionless',
            'cmap': 'SCP',
            'levels': [0, 0.5, 1, 2, 4, 8, 12, 16, 20],
            'extend': 'max',
            'category': 'severe',
            'derived': True,
            'status': '🟢 SPC-Operational',
            'inputs': ['mucape', 'effective_srh', 'effective_shear'],
            'function': 'supercell_composite_parameter',
            'description': 'SPC standard SCP (no CIN term). SCP > 1 indicates supercell potential.'
        },
        
        'scp_modified': {
            'title': 'Supercell Composite Parameter (Modified with CIN)',
            'units': 'dimensionless',
            'cmap': 'SCP',
            'levels': [0, 0.5, 1, 2, 4, 8, 12, 16, 20],
            'extend': 'max',
            'category': 'severe',
            'derived': True,
            'status': '🟡 Modified',
            'inputs': ['mucape', 'effective_srh', 'effective_shear', 'mucin'],
            'function': 'supercell_composite_parameter_modified',
            'description': 'Modified SCP with CIN weighting for operational applications.'
        },
        
        'stp_fixed': {
            'title': 'Significant Tornado Parameter (Fixed Layer - SPC)',
            'units': 'dimensionless', 
            'cmap': 'STP',
            'levels': [0, 0.5, 1, 2, 3, 4, 5, 8, 10],
            'extend': 'max',
            'category': 'severe',
            'derived': True,
            'status': '🟢 SPC-Operational',
            'inputs': ['mlcape', 'mlcin', 'srh_01km', 'wind_shear_06km', 'lcl_height'],
            'function': 'significant_tornado_parameter_fixed',
            'description': 'SPC canonical fixed-layer STP with CIN term. STP > 1 indicates heightened EF2+ tornado risk.'
        },
        
        'stp_effective': {
            'title': 'Significant Tornado Parameter (Effective Layer - SPC)',
            'units': 'dimensionless', 
            'cmap': 'STP',
            'levels': [0, 0.5, 1, 2, 3, 4, 5, 8, 10],
            'extend': 'max',
            'category': 'severe',
            'derived': True,
            'status': '🟢 SPC-Operational',
            'inputs': ['mlcape', 'mlcin', 'effective_srh', 'effective_shear', 'lcl_height'],
            'function': 'significant_tornado_parameter_effective',
            'description': 'SPC canonical effective-layer STP with CIN term. Uses ESRH and EBWD for improved accuracy.'
        },
        
        'stp_fixed_no_cin': {
            'title': 'Significant Tornado Parameter (Fixed Layer - No CIN)',
            'units': 'dimensionless', 
            'cmap': 'STP',
            'levels': [0, 0.5, 1, 2, 3, 4, 5, 8, 10],
            'extend': 'max',
            'category': 'severe',
            'derived': True,
            'status': '🟡 Modified',
            'inputs': ['mlcape', 'srh_01km', 'wind_shear_06km', 'lcl_height'],
            'function': 'significant_tornado_parameter_fixed_no_cin',
            'description': 'Modified STP without CIN term for comparison studies.'
        },
        
        'stp': {
            'title': 'Significant Tornado Parameter (Legacy)',
            'units': 'dimensionless', 
            'cmap': 'STP',
            'levels': [0, 0.5, 1, 2, 3, 4, 5, 8, 10],
            'extend': 'max',
            'category': 'severe',
            'derived': True,
            'status': '🟡 Legacy',
            'inputs': ['mlcape', 'mlcin', 'srh_01km', 'wind_shear_06km', 'lcl_height'],
            'function': 'significant_tornado_parameter',
            'description': 'Legacy STP for backward compatibility. Prefer stp_fixed or stp_effective.'
        },
        
        'ehi_spc': {
            'title': 'Energy-Helicity Index (SPC Canonical)',
            'units': 'dimensionless',
            'cmap': 'EHI',
            'levels': [-2, -1, 0, 1, 2, 4, 6, 8, 10],
            'extend': 'both',
            'category': 'severe',
            'derived': True,
            'status': '🟢 SPC-Operational',
            'inputs': ['sbcape', 'srh_03km'],
            'function': 'energy_helicity_index',
            'description': 'SPC canonical EHI: (CAPE/1000) × (SRH/100). EHI > 2 indicates significant tornado potential.'
        },
        
        'ehi_display': {
            'title': 'Energy-Helicity Index (Display Scaled)',
            'units': 'dimensionless',
            'cmap': 'EHI',
            'levels': [-1, -0.5, 0, 0.5, 1, 2, 3, 5, 7],
            'extend': 'both',
            'category': 'severe',
            'derived': True,
            'status': '🟡 Modified',
            'inputs': ['sbcape', 'srh_03km'],
            'function': 'energy_helicity_index_display',
            'description': 'Display-scaled EHI with damping for visualization. Adjusted thresholds: >0.6, >1.25, >2.5.'
        },
        
        'brn': {
            'title': 'Bulk Richardson Number',
            'units': 'dimensionless',
            'cmap': 'BRN',
            'levels': [0, 5, 10, 20, 30, 45, 60, 80, 100],
            'extend': 'max',
            'category': 'severe',
            'derived': True,
            'inputs': ['sbcape', 'wind_shear_06km'],
            'function': 'bulk_richardson_number',
            'description': 'BRN 10-45 optimal for supercells. <10 = extreme shear, >50 = weak shear (pulse storms).'
        },
        
        'uh_tornado_risk': {
            'title': 'Updraft Helicity > 75 (Tornado Risk)',
            'units': 'binary',
            'cmap': 'Reds',
            'levels': [0.5, 1],
            'extend': 'neither',
            'category': 'severe',
            'derived': True,
            'inputs': ['updraft_helicity'],
            'function': 'updraft_helicity_threshold',
            'kwargs': {'threshold': 75.0}
        },
        
        'composite_severe': {
            'title': 'Composite Severe Weather Index',
            'units': 'index',
            'cmap': 'YlOrRd',
            'levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'extend': 'max',
            'category': 'severe',
            'derived': True,
            'inputs': ['scp', 'stp', 'updraft_helicity'],
            'function': 'composite_severe_index'
        }
    }
    
    return config


# Legacy class for backward compatibility (can be removed in future versions)
class DerivedParameters:
    """Legacy wrapper class for backward compatibility."""
    
    @staticmethod
    def get_function(name: str):
        """Get a function by name."""
        return _DERIVED_FUNCTIONS.get(name)
    
    # Add static methods for all functions for backward compatibility
    supercell_composite_parameter = staticmethod(supercell_composite_parameter)
    supercell_composite_parameter_modified = staticmethod(supercell_composite_parameter_modified)
    significant_tornado_parameter = staticmethod(significant_tornado_parameter)
    significant_tornado_parameter_fixed = staticmethod(significant_tornado_parameter_fixed)
    significant_tornado_parameter_fixed_modified = staticmethod(significant_tornado_parameter_fixed_modified)
    significant_tornado_parameter_fixed_no_cin = staticmethod(significant_tornado_parameter_fixed_no_cin)
    significant_tornado_parameter_effective = staticmethod(significant_tornado_parameter_effective)
    energy_helicity_index = staticmethod(energy_helicity_index)
    energy_helicity_index_display = staticmethod(energy_helicity_index_display)
    energy_helicity_index_01km = staticmethod(energy_helicity_index_01km)
    wind_shear_magnitude = staticmethod(wind_shear_magnitude)
    updraft_helicity_threshold = staticmethod(updraft_helicity_threshold)
    composite_severe_index = staticmethod(composite_severe_index)
    wet_bulb_temperature = staticmethod(wet_bulb_temperature)
    wet_bulb_temperature_metpy = staticmethod(wet_bulb_temperature)  # legacy alias
    _wet_bulb_approximation = staticmethod(_wet_bulb_approximation)
    wbgt_shade = staticmethod(wbgt_shade)
    wbgt_estimated_outdoor = staticmethod(wbgt_estimated_outdoor)
    wbgt_simplified_outdoor = staticmethod(wbgt_simplified_outdoor)
    mixing_ratio_2m = staticmethod(mixing_ratio_2m)
    _mixing_ratio_approximation = staticmethod(_mixing_ratio_approximation)
    wind_speed_10m = staticmethod(wind_speed_10m)
    crosswind_component = staticmethod(crosswind_component)
    wind_shear_vector_01km = staticmethod(wind_shear_vector_01km)
    wind_shear_vector_06km = staticmethod(wind_shear_vector_06km)
    shear_vector_magnitude_ratio = staticmethod(shear_vector_magnitude_ratio)
    sweat_index = staticmethod(sweat_index)
    cross_totals = staticmethod(cross_totals)
    effective_srh = staticmethod(effective_srh)
    bulk_richardson_number = staticmethod(bulk_richardson_number)
    craven_brooks_composite = staticmethod(craven_brooks_composite)
    modified_stp_effective = staticmethod(modified_stp_effective)
    right_mover_supercell_composite = staticmethod(right_mover_supercell_composite)
    supercell_strength_index = staticmethod(supercell_strength_index)
    mesocyclone_strength_parameter = staticmethod(mesocyclone_strength_parameter)
    surface_richardson_number = staticmethod(surface_richardson_number)
    monin_obukhov_length = staticmethod(monin_obukhov_length)
    convective_velocity_scale = staticmethod(convective_velocity_scale)
    turbulent_kinetic_energy_estimate = staticmethod(turbulent_kinetic_energy_estimate)
    haines_index = staticmethod(haines_index)
    ventilation_rate = staticmethod(ventilation_rate)
    enhanced_smoke_dispersion_index = staticmethod(enhanced_smoke_dispersion_index)
    ventilation_rate_from_components = staticmethod(ventilation_rate_from_components)
    ventilation_rate_from_surface_winds = staticmethod(ventilation_rate_from_surface_winds)
    detect_effective_inflow_layer = staticmethod(detect_effective_inflow_layer)
    compute_effective_layer_wind_shear = staticmethod(compute_effective_layer_wind_shear)
    compute_effective_layer_srh = staticmethod(compute_effective_layer_srh)
    enhanced_smoke_dispersion_index_simplified = staticmethod(enhanced_smoke_dispersion_index_simplified)
    shear_vector_magnitude_ratio_from_components = staticmethod(shear_vector_magnitude_ratio_from_components)
    enhanced_smoke_dispersion_index_from_components = staticmethod(enhanced_smoke_dispersion_index_from_components)
    crude_lcl_estimate = staticmethod(crude_lcl_estimate)
    absolute_vorticity_500_estimate = staticmethod(absolute_vorticity_500_estimate)
    _calculate_saturation_vapor_pressure = staticmethod(_calculate_saturation_vapor_pressure)
    _calculate_virtual_temperature = staticmethod(_calculate_virtual_temperature)
    _find_lcl_bolton = staticmethod(_find_lcl_bolton)
    _moist_adiabatic_temperature = staticmethod(_moist_adiabatic_temperature)
    surface_based_cape_and_cin = staticmethod(surface_based_cape_and_cin)
    mixed_layer_cape_and_cin = staticmethod(mixed_layer_cape_and_cin)
    most_unstable_cape_and_cin = staticmethod(most_unstable_cape_and_cin)
    calculate_surface_based_cape = staticmethod(calculate_surface_based_cape)
    calculate_surface_based_cin = staticmethod(calculate_surface_based_cin)
    calculate_mixed_layer_cape = staticmethod(calculate_mixed_layer_cape)
    calculate_mixed_layer_cin = staticmethod(calculate_mixed_layer_cin)
    calculate_most_unstable_cape = staticmethod(calculate_most_unstable_cape)
    lifted_index = staticmethod(lifted_index)
    showalter_index = staticmethod(showalter_index)
    significant_hail_parameter = staticmethod(significant_hail_parameter)
    craven_significant_severe = staticmethod(craven_significant_severe)
    vorticity_generation_parameter = staticmethod(vorticity_generation_parameter)
    supercell_composite_parameter_effective = staticmethod(supercell_composite_parameter_effective)
    cape_03km = staticmethod(cape_03km)
    lapse_rate_03km = staticmethod(lapse_rate_03km)
    effective_shear = staticmethod(effective_shear)
    significant_tornado_parameter_cin = staticmethod(significant_tornado_parameter_cin)
    violent_tornado_parameter = staticmethod(violent_tornado_parameter)
    calculate_lapse_rate_700_500 = staticmethod(calculate_lapse_rate_700_500)


if __name__ == '__main__':
    # Example usage
    print("HRRR Derived Parameters Package")
    print(f"Available functions: {len(_DERIVED_FUNCTIONS)}")
    for name in sorted(_DERIVED_FUNCTIONS.keys()):
        print(f"  - {name}")
    
    # Show derived parameter configurations
    configs = create_derived_config()
    print(f"\nDerived parameter configurations: {len(configs)}")
    for name, config in configs.items():
        print(f"  - {name}: {config['title']}")