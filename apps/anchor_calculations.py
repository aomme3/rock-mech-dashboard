"""
Beregningsmodul for passive forankringer i berg.
Kilde: NGI-rapport 20210114-01-R (Sikringshåndboken, NVE)

Alle krafter i kN, lengder i mm, spenninger i MPa.
Ingen Streamlit-avhengigheter i denne modulen.
"""

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tverrsnitt og stålkapasitet
# ---------------------------------------------------------------------------

def stress_area(d_major: float, pitch: float) -> float:
    """
    Spenningsareal for metriske grovgjenger (ISO 262).

    A_s = π/4 · (d_major - 0.9382·pitch)²

    Args:
        d_major: Nominell ytre diameter [mm]
        pitch:   Gjengestigning [mm]

    Returns:
        A_s: Spenningsareal [mm²]
    """
    if d_major <= 0 or pitch <= 0:
        raise ValueError("d_major og pitch må være positive.")
    d_s = d_major - 0.9382 * pitch
    return math.pi / 4 * d_s ** 2


def r_itd(A_s: float, f_yk: float, gamma_s: float = 1.35) -> float:
    """
    Dimensjonerende strekkapasitet fra stålkvalitet.

    R_itd = A_s · f_yk / gamma_s

    Args:
        A_s:     Spenningsareal [mm²]
        f_yk:    Karakteristisk flytespenning [MPa]
        gamma_s: Materialfaktor for stål [-]

    Returns:
        R_itd: Dimensjonerende strekkapasitet [kN]
    """
    if A_s <= 0 or f_yk <= 0 or gamma_s <= 0:
        raise ValueError("A_s, f_yk og gamma_s må være positive.")
    return A_s * f_yk / gamma_s / 1000  # N → kN


def r_vd(A_s: float, f_yk: float, gamma_s: float = 1.35) -> float:
    """
    Dimensjonerende skjærkapasitet (von Mises, koeff √3).

    R_Vd = A_s · f_yk / (sqrt(3) · gamma_s)

    Args:
        A_s:     Spenningsareal [mm²]
        f_yk:    Karakteristisk flytespenning [MPa]
        gamma_s: Materialfaktor for stål [-]

    Returns:
        R_Vd: Dimensjonerende skjærkapasitet [kN]
    """
    if A_s <= 0 or f_yk <= 0 or gamma_s <= 0:
        raise ValueError("A_s, f_yk og gamma_s må være positive.")
    return A_s * f_yk / (math.sqrt(3) * gamma_s) / 1000  # N → kN


def r_itd_supplier(F_tk: float, gamma_s: float = 1.35) -> float:
    """
    Dimensjonerende strekkapasitet fra leverandørdata.

    R_itd = F_tk / gamma_s

    Args:
        F_tk:    Karakteristisk strekkraft fra leverandør [kN]
        gamma_s: Materialfaktor [-]

    Returns:
        R_itd: Dimensjonerende strekkapasitet [kN]
    """
    if F_tk <= 0 or gamma_s <= 0:
        raise ValueError("F_tk og gamma_s må være positive.")
    return F_tk / gamma_s


def r_vd_supplier(F_vk: float, gamma_s: float = 1.35) -> float:
    """
    Dimensjonerende skjærkapasitet fra leverandørdata.

    R_Vd = F_vk / gamma_s

    Args:
        F_vk:    Karakteristisk skjærkraft fra leverandør [kN]
        gamma_s: Materialfaktor [-]

    Returns:
        R_Vd: Dimensjonerende skjærkapasitet [kN]
    """
    if F_vk <= 0 or gamma_s <= 0:
        raise ValueError("F_vk og gamma_s må være positive.")
    return F_vk / gamma_s


# ---------------------------------------------------------------------------
# Kraftdekomponering
# ---------------------------------------------------------------------------

def decompose(F_Ed: float, beta_deg: float) -> tuple[float, float]:
    """
    Dekomponerer resultantkraft F_Ed i aksial- og skjærkomponent.

    N_Ed = F_Ed · cos(β)   (langs stagaksen)
    V_Ed = F_Ed · sin(β)   (vinkelrett på stagaksen)

    Args:
        F_Ed:     Dimensjonerende resultantkraft [kN]
        beta_deg: Vinkel mellom kraft og stagaksen i vertikalplanet [grader, 0–90]

    Returns:
        (N_Ed, V_Ed): Aksialkraft [kN], skjærkraft [kN]
    """
    if not (0.0 <= beta_deg <= 90.0):
        raise ValueError(f"beta_deg={beta_deg} er utenfor 0–90°.")
    beta = math.radians(beta_deg)
    return F_Ed * math.cos(beta), F_Ed * math.sin(beta)


def apply_load_factors(
    N_k: float, V_k: float,
    gamma_FN: float = 1.5, gamma_FV: float = 1.5
) -> tuple[float, float]:
    """
    Påfører lastfaktorer på karakteristiske kraftkomponenter.

    N_Ed = gamma_FN · N_k
    V_Ed = gamma_FV · V_k

    Args:
        N_k:      Karakteristisk aksialkraft [kN]
        V_k:      Karakteristisk skjærkraft [kN]
        gamma_FN: Lastfaktor for aksialkraft [-]
        gamma_FV: Lastfaktor for skjærkraft [-]

    Returns:
        (N_Ed, V_Ed): Dimensjonerende kraftkomponenter [kN]
    """
    return gamma_FN * N_k, gamma_FV * V_k


# ---------------------------------------------------------------------------
# Kapasitetskontroll
# ---------------------------------------------------------------------------

def utilisation_von_mises(N_Ed: float, V_Ed: float, R_itd: float) -> float:
    """
    Utnyttelsesgrad etter von Mises-kriterium (kap. 4.3.2).

    U = sqrt(N_Ed² + 3·V_Ed²) / R_itd

    Args:
        N_Ed:  Dimensjonerende aksialkraft [kN]
        V_Ed:  Dimensjonerende skjærkraft [kN]
        R_itd: Dimensjonerende strekkapasitet [kN]

    Returns:
        U: Utnyttelsesgrad [-]
    """
    if R_itd <= 0:
        raise ValueError("R_itd må være positiv.")
    return math.sqrt(N_Ed ** 2 + 3 * V_Ed ** 2) / R_itd


def utilisation_elliptic(
    N_Ed: float, V_Ed: float, R_itd: float, R_Vd: float
) -> float:
    """
    Utnyttelsesgrad etter elliptisk interaksjonsformel.

    U = sqrt((N_Ed/R_itd)² + (V_Ed/R_Vd)²)

    Args:
        N_Ed:  Dimensjonerende aksialkraft [kN]
        V_Ed:  Dimensjonerende skjærkraft [kN]
        R_itd: Dimensjonerende strekkapasitet [kN]
        R_Vd:  Dimensjonerende skjærkapasitet [kN]

    Returns:
        U: Utnyttelsesgrad [-]
    """
    if R_itd <= 0 or R_Vd <= 0:
        raise ValueError("R_itd og R_Vd må være positive.")
    return math.sqrt((N_Ed / R_itd) ** 2 + (V_Ed / R_Vd) ** 2)


def critical_angle(
    F_Ed: float, R_itd: float, R_Vd: float, method: str = "elliptic"
) -> float:
    """
    Grensevinkel β der utnyttelsen U = 1.

    Args:
        F_Ed:   Dimensjonerende resultantkraft [kN]
        R_itd:  Dimensjonerende strekkapasitet [kN]
        R_Vd:   Dimensjonerende skjærkapasitet [kN]
        method: 'von_mises' eller 'elliptic'

    Returns:
        beta_crit: Grensevinkel [grader], eller 90 hvis aldri nådd
    """
    for beta in np.linspace(0, 90, 9001):
        N, V = decompose(F_Ed, beta)
        if method == "von_mises":
            u = utilisation_von_mises(N, V, R_itd)
        else:
            u = utilisation_elliptic(N, V, R_itd, R_Vd)
        if u >= 1.0:
            return float(beta)
    return 90.0


# ---------------------------------------------------------------------------
# Innfestingskapasitet (kap. 4.5)
# ---------------------------------------------------------------------------

def f_ck(f_ccube: float) -> float:
    """
    Sylinderfasthet fra terningfasthet.

    f_ck = 0.8 · f_ccube

    Args:
        f_ccube: Karakteristisk terningfasthet [MPa]

    Returns:
        f_ck: Sylinderfasthet [MPa]
    """
    if f_ccube <= 0:
        raise ValueError("f_ccube må være positiv.")
    return 0.8 * f_ccube


def f_ctk(f_ck_val: float) -> float:
    """
    Karakteristisk strekkfasthet for injeksjonsmasse.

    f_ctk = 0.7 · 0.3 · f_ck^(2/3)

    Args:
        f_ck_val: Sylinderfasthet [MPa]

    Returns:
        f_ctk: Karakteristisk strekkfasthet [MPa]
    """
    if f_ck_val <= 0:
        raise ValueError("f_ck må være positiv.")
    return 0.7 * 0.3 * f_ck_val ** (2 / 3)


def f_ctd(f_ctk_val: float, gamma_c: float = 1.5) -> float:
    """
    Dimensjonerende strekkfasthet for injeksjonsmasse.

    f_ctd = f_ctk / gamma_c

    Args:
        f_ctk_val: Karakteristisk strekkfasthet [MPa]
        gamma_c:   Materialfaktor for betong/injeksjonsmasse [-]

    Returns:
        f_ctd: Dimensjonerende strekkfasthet [MPa]
    """
    if f_ctk_val <= 0 or gamma_c <= 0:
        raise ValueError("f_ctk og gamma_c må være positive.")
    return f_ctk_val / gamma_c


def f_bd(f_ctd_val: float) -> float:
    """
    Dimensjonerende heftfasthet (innstøpt bolt i injeksjonsmasse).

    f_bd = 2.25 · f_ctd

    Args:
        f_ctd_val: Dimensjonerende strekkfasthet [MPa]

    Returns:
        f_bd: Dimensjonerende heftfasthet [MPa]
    """
    if f_ctd_val <= 0:
        raise ValueError("f_ctd må være positiv.")
    return 2.25 * f_ctd_val


def R_tg(f_bd_val: float, d: float, l_b: float) -> float:
    """
    Innfestingskapasitet stål–injeksjonsmasse (periferilast).

    R_tg = π · d · l_b · f_bd / 1000  [kN]

    Args:
        f_bd_val: Dimensjonerende heftfasthet stål–injeksjonsmasse [MPa]
        d:        Boltediameter [mm]
        l_b:      Innfestingslengde [mm]

    Returns:
        R_tg: Kapasitet stål–injeksjonsmasse [kN]
    """
    if d <= 0 or l_b <= 0 or f_bd_val <= 0:
        raise ValueError("d, l_b og f_bd må være positive.")
    return math.pi * d * l_b * f_bd_val / 1000


def R_gg(tau_d: float, d_bh: float, l_b: float) -> float:
    """
    Innfestingskapasitet injeksjonsmasse–berg (periferilast ved borehullsveggen).

    R_gg = π · d_bh · l_b · tau_d / 1000  [kN]

    Args:
        tau_d: Dimensjonerende heftfasthet injeksjonsmasse–berg [MPa]
        d_bh:  Borehullsdiameter [mm]
        l_b:   Innfestingslengde [mm]

    Returns:
        R_gg: Kapasitet injeksjonsmasse–berg [kN]
    """
    if d_bh <= 0 or l_b <= 0 or tau_d <= 0:
        raise ValueError("d_bh, l_b og tau_d må være positive.")
    return math.pi * d_bh * l_b * tau_d / 1000


# ---------------------------------------------------------------------------
# Ytre kapasitet – kjeglemodell (kap. 4.6)
# ---------------------------------------------------------------------------

def lambda_required(
    gamma_M: float, P_p: float, tau_k: float, psi_deg: float
) -> float:
    """
    Nødvendig forankringslengde fra kjeglemodell (kap. 4.6).

    λ = gamma_M · P_p / (π · tau_k · tan(ψ))  [mm]

    Args:
        gamma_M: Materialfaktor berg [-]
        P_p:     Dimensjonerende uttrekkskraft [kN]
        tau_k:   Karakteristisk skjærstyrke berg langs kjegleflate [kPa]
        psi_deg: Halvåpningsvinkel kjegle [grader]

    Returns:
        lambda_req: Nødvendig forankringslengde [mm]
    """
    if gamma_M <= 0 or P_p <= 0 or tau_k <= 0:
        raise ValueError("gamma_M, P_p og tau_k må være positive.")
    if not (0 < psi_deg < 90):
        raise ValueError("psi_deg må være mellom 0 og 90 grader.")
    tau_MPa = tau_k / 1000  # kPa → MPa
    psi = math.radians(psi_deg)
    # Kraft [kN], tau [MPa], l i mm:  l² · pi · tau · tan(psi) = gamma_M · P_p [kN]
    # l [mm]: løs l² = gamma_M * P_p * 1000 / (pi * tau_MPa * tan(psi))
    l2 = gamma_M * P_p * 1000 / (math.pi * tau_MPa * math.tan(psi))
    return math.sqrt(max(l2, 0.0))


# ---------------------------------------------------------------------------
# Sensitivitetsanalyse
# ---------------------------------------------------------------------------

def sensitivity_beta(
    F_Ed: float,
    R_itd: float,
    R_Vd: float,
    method: str = "elliptic",
    beta_range: tuple[float, float] = (0.0, 90.0),
    n_pts: int = 181,
) -> pd.DataFrame:
    """
    Sensitivitetsanalyse: utnyttelse som funksjon av vinkel β.

    Args:
        F_Ed:       Dimensjonerende resultantkraft [kN]
        R_itd:      Dimensjonerende strekkapasitet [kN]
        R_Vd:       Dimensjonerende skjærkapasitet [kN]
        method:     'von_mises' eller 'elliptic'
        beta_range: (beta_min, beta_max) [grader]
        n_pts:      Antall beregningspunkter

    Returns:
        DataFrame med kolonner: beta_deg, N_Ed, V_Ed, U, status
    """
    betas = np.linspace(beta_range[0], beta_range[1], n_pts)
    rows = []
    for b in betas:
        N, V = decompose(F_Ed, float(b))
        if method == "von_mises":
            u = utilisation_von_mises(N, V, R_itd)
        else:
            u = utilisation_elliptic(N, V, R_itd, R_Vd)
        rows.append({
            "beta_deg": round(float(b), 2),
            "N_Ed": round(N, 3),
            "V_Ed": round(V, 3),
            "U": round(u, 4),
            "status": "OK" if u <= 1.0 else "IKKE OK",
        })
    return pd.DataFrame(rows)


def sensitivity_parameter(
    param_name: str,
    param_range: tuple[float, float],
    fixed_inputs: dict,
    method: str = "elliptic",
    n_pts: int = 50,
) -> pd.DataFrame:
    """
    Sensitivitetsanalyse: utnyttelse som funksjon av én valgt parameter.

    Args:
        param_name:   Navn på parameteren som varieres
                      ('F_k', 'gamma_s', 'l_b', 'd', 'f_yk', 'tau_k_berg')
        param_range:  (min, max) for parameteren
        fixed_inputs: Dict med alle øvrige parametre:
                      {F_k, gamma_s, beta_deg, gamma_FN, gamma_FV,
                       A_s, f_yk, l_b, d, d_bh, tau_k_berg, gamma_tau,
                       F_tk, F_vk, mode}
                      mode: 'steel' | 'supplier'
        method:       'von_mises' eller 'elliptic'
        n_pts:        Antall punkter

    Returns:
        DataFrame med kolonner: param_value, U, status
    """
    values = np.linspace(param_range[0], param_range[1], n_pts)
    rows = []
    for v in values:
        inp = fixed_inputs.copy()
        inp[param_name] = float(v)

        mode = inp.get("mode", "steel")
        gamma_s   = inp.get("gamma_s", 1.35)
        beta_deg  = inp.get("beta_deg", 0.0)
        gamma_FN  = inp.get("gamma_FN", 1.5)
        gamma_FV  = inp.get("gamma_FV", 1.5)
        F_k       = inp.get("F_k", 100.0)

        N_k, V_k = decompose(F_k, beta_deg)
        N_Ed, V_Ed = apply_load_factors(N_k, V_k, gamma_FN, gamma_FV)

        if mode == "supplier":
            rit = r_itd_supplier(inp.get("F_tk", 525), gamma_s)
            rvd = r_vd_supplier(inp.get("F_vk", 262), gamma_s)
        else:
            A_s  = inp.get("A_s", 561.0)
            f_yk = inp.get("f_yk", 500.0)
            rit = r_itd(A_s, f_yk, gamma_s)
            rvd = r_vd(A_s, f_yk, gamma_s)

        if method == "von_mises":
            u = utilisation_von_mises(N_Ed, V_Ed, rit)
        else:
            u = utilisation_elliptic(N_Ed, V_Ed, rit, rvd)

        rows.append({
            "param_value": round(float(v), 4),
            "U": round(u, 4),
            "status": "OK" if u <= 1.0 else "IKKE OK",
        })
    return pd.DataFrame(rows)
