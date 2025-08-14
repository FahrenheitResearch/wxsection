# derived_params/wet_bulb_temperature_metpy.py
import warnings
from .wet_bulb_temperature import wet_bulb_temperature as _wet_bulb_temperature

def wet_bulb_temperature_metpy(*args, **kwargs):
    warnings.warn(
        "wet_bulb_temperature_metpy is deprecated; use wet_bulb_temperature",
        DeprecationWarning,
        stacklevel=2,
    )
    return _wet_bulb_temperature(*args, **kwargs)
