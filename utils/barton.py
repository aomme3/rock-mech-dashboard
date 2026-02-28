import numpy as np

def scale_jrc_jcs(JRC0: float, JCS0_mpa: float, L_m: float, L0_m: float) -> tuple[float, float]:
    """
    Vanlig brukt Barton-skalering:
      JRC(L) = JRC0 * (L/L0)^(-0.02*JRC0)
      JCS(L) = JCS0 * (L/L0)^(-0.03*JRC0)
    """
    ratio = max(L_m / max(L0_m, 1e-12), 1e-12)
    JRC = float(JRC0 * (ratio ** (-0.02 * JRC0)))
    JCS = float(JCS0_mpa * (ratio ** (-0.03 * JRC0)))
    return JRC, JCS

def tau_barton_bandis_mpa(sigma_eff_mpa: float, JRC: float, JCS_mpa: float, phi_b_deg: float) -> float:
    """
    τ = σn' * tan( φb + JRC*log10(JCS/σn') )
    """
    sn = max(float(sigma_eff_mpa), 1e-6)
    angle = phi_b_deg + JRC * np.log10(max(JCS_mpa, 1e-6) / sn)
    return float(sn * np.tan(np.radians(angle)))

def phi_active_deg_from_tau_sigma(tau_mpa: float, sigma_mpa: float) -> float:
    return float(np.degrees(np.arctan(max(tau_mpa, 0.0) / max(sigma_mpa, 1e-6))))