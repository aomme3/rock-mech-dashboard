import numpy as np

def resolve_to_plane_components(Fx_kN, Fy_kN, alpha_deg):
    """
    Komponenter relativt til plan:
      T: + ned langs plan
      N: + inn i planet
    """
    a = np.radians(alpha_deg)
    t = np.array([np.cos(a), -np.sin(a)])
    n = np.array([np.sin(a),  np.cos(a)])
    F = np.array([Fx_kN, Fy_kN])
    return float(F @ t), float(F @ n)

def moment_about_point(Fx, Fy, x, y, x0=0.0, y0=0.0):
    """M = (x-x0)*Fy - (y-y0)*Fx"""
    return float((x - x0) * Fy - (y - y0) * Fx)

def kpa_from_head_m(head_m: float, gamma_w_kN_m3: float = 9.81) -> float:
    """1 m vannsøyle = gamma_w kPa"""
    return float(gamma_w_kN_m3 * head_m)