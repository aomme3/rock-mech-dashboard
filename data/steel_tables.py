"""
Stål- og produkttabeller for passive forankringer i berg.
Kilde: NGI-rapport 20210114-01-R (Sikringshåndboken, NVE)
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Tabell 4 – Stålkvaliteter
# ---------------------------------------------------------------------------
STEEL_GRADES = pd.DataFrame([
    {"Ståltype": "B500NC",      "f_yk": 500,  "f_uk": 550,  "Merknad": "Armering, duktil (NC)"},
    {"Ståltype": "St555/700",   "f_yk": 555,  "f_uk": 700,  "Merknad": "Bergsikringsbolter (SB)"},
    {"Ståltype": "St670/800",   "f_yk": 670,  "f_uk": 800,  "Merknad": "Bergsikringsbolter, høyfast"},
    {"Ståltype": "St835/1030",  "f_yk": 835,  "f_uk": 1030, "Merknad": "Selvborende stag, høyfast"},
    {"Ståltype": "S355",        "f_yk": 355,  "f_uk": 490,  "Merknad": "Konstruksjonsstål (EN 10025)"},
    {"Ståltype": "S460NH",      "f_yk": 460,  "f_uk": 550,  "Merknad": "Konstruksjonsstål, hulprofil"},
])

# ---------------------------------------------------------------------------
# Tabell 9 – Spenningsareal for metriske grovgjenger (ISO 262)
# ---------------------------------------------------------------------------
STRESS_AREAS = pd.DataFrame([
    {"Betegnelse": "M20", "d_major": 20.0, "Stigning": 2.5,  "A_s": 245},
    {"Betegnelse": "M22", "d_major": 22.0, "Stigning": 2.5,  "A_s": 303},
    {"Betegnelse": "M24", "d_major": 24.0, "Stigning": 3.0,  "A_s": 353},
    {"Betegnelse": "M27", "d_major": 27.0, "Stigning": 3.0,  "A_s": 459},
    {"Betegnelse": "M30", "d_major": 30.0, "Stigning": 3.5,  "A_s": 561},
    {"Betegnelse": "M33", "d_major": 33.0, "Stigning": 3.5,  "A_s": 694},
    {"Betegnelse": "M36", "d_major": 36.0, "Stigning": 4.0,  "A_s": 817},
    {"Betegnelse": "M39", "d_major": 39.0, "Stigning": 4.0,  "A_s": 976},
])

# For lookup by nom diameter (closest match)
STRESS_AREA_BY_NAME = {row["Betegnelse"]: row for _, row in STRESS_AREAS.iterrows()}

# ---------------------------------------------------------------------------
# Tabell 11.6.4.5-1 – Karakteristisk heftfasthet mørtel–berg
# Kilde: Internrapport nr. 2374 [226]
# Tyngdetetthet: midtpunkt av oppgitt område [kN/m³]
# Trykkfasthet:  midtpunkt av oppgitt område [MPa]
# Heftfasthet:   karakteristisk verdi τ_k [MPa]  (γ_τ = 1,25 anbefalt)
# ---------------------------------------------------------------------------
ROCK_BOND = pd.DataFrame([
    {"Bergart": "Granitt",    "Tyngdetetthet_kNm3": "25–28", "Trykkfasthet_MPa": " 90–170", "tau_k_MPa": 2.0},
    {"Bergart": "Gabbro",     "Tyngdetetthet_kNm3": "27–31", "Trykkfasthet_MPa": " 18–250", "tau_k_MPa": 2.5},
    {"Bergart": "Gneis",      "Tyngdetetthet_kNm3": "25–28", "Trykkfasthet_MPa": " 90–130", "tau_k_MPa": 1.5},
    {"Bergart": "Kvartsitt",  "Tyngdetetthet_kNm3": "21–25", "Trykkfasthet_MPa": "150–170", "tau_k_MPa": 2.5},
    {"Bergart": "Sandstein",  "Tyngdetetthet_kNm3": "20–26", "Trykkfasthet_MPa": "100–140", "tau_k_MPa": 1.2},
    {"Bergart": "Kalkstein",  "Tyngdetetthet_kNm3": "25–28", "Trykkfasthet_MPa": " 70–100", "tau_k_MPa": 2.0},
    {"Bergart": "Leirskifer", "Tyngdetetthet_kNm3": "20–27", "Trykkfasthet_MPa": " 25–60",  "tau_k_MPa": 0.5},
])

# ---------------------------------------------------------------------------
# Tabell 11.6.4.5-2 – Bruddvinkel og heftfasthet på bruddplan
# Kilde: NGI-rapport 20210114-01-R, tabell 11.6.4.5-2
# tau_k i kPa (karakteristisk), psi_max i grader
# ---------------------------------------------------------------------------
CONE_PARAMS = pd.DataFrame([
    {
        "Bergkvalitet": "Meget godt berg — ett sprekkesett m/sporadiske sprekker; UCS > 50 MPa",
        "tau_k_min": 100, "tau_k_max": 200, "psi_max": 45,
    },
    {
        "Bergkvalitet": "To sprekkesett m/sporadiske sprekker; UCS 15–50 MPa",
        "tau_k_min": 50,  "tau_k_max": 100, "psi_max": 40,
    },
    {
        "Bergkvalitet": "Tre sprekkesett m/sporadiske sprekker (<20 pr. m²); UCS < 15 MPa",
        "tau_k_min": 50,  "tau_k_max": 50,  "psi_max": 30,
    },
])

# ---------------------------------------------------------------------------
# Leverandørdata – selvborende stag (eksempeldata T-Pretec)
# ---------------------------------------------------------------------------
SELF_DRILLING_ANCHORS = pd.DataFrame([
    {"Produkt": "T40/16 Pretec",  "Ytre_diam": 40,  "Indre_diam": 16, "F_tk": 525,  "F_vk": 262,  "Vekt": 7.3},
    {"Produkt": "T52/26 Pretec",  "Ytre_diam": 52,  "Indre_diam": 26, "F_tk": 800,  "F_vk": 400,  "Vekt": 9.2},
    {"Produkt": "T76/46 Pretec",  "Ytre_diam": 76,  "Indre_diam": 46, "F_tk": 1200, "F_vk": 600,  "Vekt": 14.5},
    {"Produkt": "R25 IBO",        "Ytre_diam": 25,  "Indre_diam": 9,  "F_tk": 250,  "F_vk": 125,  "Vekt": 3.0},
    {"Produkt": "R32 IBO",        "Ytre_diam": 32,  "Indre_diam": 11, "F_tk": 400,  "F_vk": 200,  "Vekt": 4.4},
    {"Produkt": "R51 IBO",        "Ytre_diam": 51,  "Indre_diam": 20, "F_tk": 800,  "F_vk": 400,  "Vekt": 8.6},
])
