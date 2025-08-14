from .common import _dbg
from ._mixing_ratio_approximation import _mixing_ratio_approximation
from ._psychrometrics import e_from_dewpoint_pa, mixing_ratio_from_e, _to_pa
import numpy as np

def mixing_ratio_2m(dewpoint_2m, pressure):
    """
    Compute 2m mixing ratio (g/kg). Pressure may be Pa or hPa (auto-detected).
    """
    try:
        P_pa = _to_pa(pressure)
        e = e_from_dewpoint_pa(dewpoint_2m)
        w = mixing_ratio_from_e(P_pa, e)           # kg/kg
        mr_gkg = 1000.0 * w
        mr_gkg = np.maximum(mr_gkg, 0.0)           # non-negative
        bad = ~np.isfinite(mr_gkg)
        if np.any(bad) and (np.mean(bad) > 0.1):
            raise RuntimeError("excess NaNs in mixing_ratio_2m")
        out_dtype = np.result_type(dewpoint_2m, pressure, np.float32)
        return mr_gkg.astype(out_dtype, copy=False)
    except Exception as e:
        _dbg(f"Mixing ratio exact failed ({e}); using fallback approximation.")
        return _mixing_ratio_approximation(dewpoint_2m, pressure)
