# derived_params/wet_bulb_temperature.py
import numpy as np
from .common import _dbg
from ._wet_bulb_approximation import _wet_bulb_approximation
from ._psychrometrics import wet_bulb_bisection, _to_pa

def wet_bulb_temperature(temp_2m, dewpoint_2m, pressure):
    """
    Compute wet-bulb temperature via robust bisection (°C).
    Pressure may be Pa or hPa (auto-detected).
    """
    try:
        Tw = wet_bulb_bisection(temp_2m, dewpoint_2m, pressure)
        frac_nan = np.mean(~np.isfinite(Tw))
        if np.isnan(frac_nan) or frac_nan > 0.2:
            raise RuntimeError("excess NaNs in wet_bulb_bisection")

        # Match incoming precision
        out_dtype = np.result_type(temp_2m, dewpoint_2m, pressure, np.float32)
        Tw = Tw.astype(out_dtype, copy=False)
        return Tw
    except Exception as e:
        _dbg(f"WB iterative failed ({e}); using fallback approximation.")
        return _wet_bulb_approximation(temp_2m, dewpoint_2m, _to_pa(pressure))