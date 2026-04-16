"""
PDF-rapport for kapasitetsberegning av passive forankringer i berg.
Kilde: NGI-rapport 20210114-01-R (Sikringshåndboken, NVE)

Ingen Streamlit-avhengigheter. Returnerer ferdig PDF som bytes.
"""

import io
import math
import datetime
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, KeepTogether,
)

from apps.anchor_calculations import (
    decompose, apply_load_factors,
    utilisation_von_mises, utilisation_elliptic,
    f_ck, f_ctk, f_ctd, f_bd, R_tg, R_gg, lambda_required,
)

# ---------------------------------------------------------------------------
# Farger
# ---------------------------------------------------------------------------
GREEN  = colors.HexColor("#1a7a4a")
RED    = colors.HexColor("#b81c1c")
BLUE   = colors.HexColor("#1a3a6b")
LGREY  = colors.HexColor("#f2f2f2")
MGREY  = colors.HexColor("#cccccc")
WHITE  = colors.white
BLACK  = colors.black

# ---------------------------------------------------------------------------
# Stilsett
# ---------------------------------------------------------------------------

def _styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("title", fontSize=16, leading=20,
                                 textColor=BLUE, spaceAfter=4,
                                 fontName="Helvetica-Bold", alignment=TA_LEFT)
    s["subtitle"] = ParagraphStyle("subtitle", fontSize=10, leading=13,
                                    textColor=colors.HexColor("#444444"),
                                    spaceAfter=2, fontName="Helvetica")
    s["h1"] = ParagraphStyle("h1", fontSize=12, leading=15,
                              textColor=BLUE, spaceBefore=10, spaceAfter=4,
                              fontName="Helvetica-Bold")
    s["h2"] = ParagraphStyle("h2", fontSize=10, leading=13,
                              textColor=BLUE, spaceBefore=6, spaceAfter=3,
                              fontName="Helvetica-Bold")
    s["body"] = ParagraphStyle("body", fontSize=9, leading=12,
                                fontName="Helvetica")
    s["formula"] = ParagraphStyle("formula", fontSize=9, leading=12,
                                   fontName="Courier", leftIndent=10,
                                   textColor=colors.HexColor("#333333"))
    s["ok"]  = ParagraphStyle("ok",  fontSize=9, leading=12,
                               fontName="Helvetica-Bold", textColor=GREEN)
    s["nok"] = ParagraphStyle("nok", fontSize=9, leading=12,
                               fontName="Helvetica-Bold", textColor=RED)
    s["footer"] = ParagraphStyle("footer", fontSize=7, leading=9,
                                  fontName="Helvetica",
                                  textColor=colors.HexColor("#888888"),
                                  alignment=TA_CENTER)
    return s


# ---------------------------------------------------------------------------
# Hjelpere
# ---------------------------------------------------------------------------

def _tbl(data, col_widths, style_cmds=None):
    """Lager en reportlab-tabell med standard utseende."""
    base = [
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("LEADING",     (0, 0), (-1, -1), 11),
        ("BACKGROUND",  (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LGREY]),
        ("GRID",        (0, 0), (-1, -1), 0.3, MGREY),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",(0, 0), (-1, -1), 5),
    ]
    if style_cmds:
        base.extend(style_cmds)
    return Table(data, colWidths=col_widths, style=TableStyle(base))


def _status_para(ok: bool, text: str, styles):
    key = "ok" if ok else "nok"
    prefix = "✓" if ok else "✗"
    return Paragraph(f"{prefix}  {text}", styles[key])


def _section_rule(styles):
    return HRFlowable(width="100%", thickness=0.5, color=MGREY, spaceAfter=2)


# ---------------------------------------------------------------------------
# Skisse (matplotlib → PNG i minne)
# ---------------------------------------------------------------------------

def _make_sketch_image(inp: dict, width_mm: float = 140) -> Image:
    F_k      = inp["F_k"]
    beta_deg = inp["beta"]
    gamma_FN = inp["gamma_FN"]
    gamma_FV = inp["gamma_FV"]
    N_k, V_k = decompose(F_k, beta_deg)
    N_Ed, V_Ed = apply_load_factors(N_k, V_k, gamma_FN, gamma_FV)

    beta_r = math.radians(beta_deg)
    fmax   = max(F_k, 1.0)
    sc     = 1.8 / fmax

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(-2.8, 3.2)
    ax.set_ylim(-0.8, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Stagakse
    ax.annotate("", xy=(2.6, 0), xytext=(-2.6, 0),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.5))
    ax.text(2.75, 0, "Stagakse", va="center", fontsize=8, color="#444444")

    # Resultantkraft
    fx = F_k * math.cos(beta_r) * sc
    fy = F_k * math.sin(beta_r) * sc
    ax.annotate("", xy=(fx, fy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#E63946", lw=2))
    ax.text(fx + 0.08, fy + 0.08,
            f"F_k = {F_k:.0f} kN", color="#E63946", fontsize=8)

    # N-komponent
    nx = N_k * sc
    ax.annotate("", xy=(nx, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1d5fa8", lw=1.5,
                                linestyle="dashed"))
    ax.text(nx * 0.5, -0.22,
            f"N_k = {N_k:.1f} kN", color="#1d5fa8", fontsize=7.5,
            ha="center")

    # V-komponent
    if abs(V_k) > 0.01:
        vy = V_k * sc
        ax.annotate("", xy=(0, vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#2A9D8F", lw=1.5,
                                    linestyle="dashed"))
        ax.text(0.12, vy * 0.5,
                f"V_k = {V_k:.1f} kN", color="#2A9D8F", fontsize=7.5)
        # Stiplet fra spiss til komponenter
        ax.plot([fx, nx], [fy, 0], color="#aaaaaa", lw=0.7, ls=":")
        ax.plot([fx, 0],  [fy, vy], color="#aaaaaa", lw=0.7, ls=":")

    # Vinkelbue
    if beta_deg > 0.5:
        arc_r = 0.45
        theta = np.linspace(0, beta_r, 60)
        ax.plot(arc_r * np.cos(theta), arc_r * np.sin(theta),
                color="#E63946", lw=0.8, ls="--")
        mid = beta_r / 2
        ax.text(arc_r * 1.35 * math.cos(mid), arc_r * 1.35 * math.sin(mid),
                f"β = {beta_deg:.0f}°", color="#E63946", fontsize=7.5)

    # Dimensjonerende
    ax.text(-2.6, 2.2,
            f"N_Ed = {N_Ed:.1f} kN  |  V_Ed = {V_Ed:.1f} kN  (etter lastfaktorer)",
            fontsize=7.5, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc"))

    # Forklaring
    handles = [
        mpatches.Patch(color="#E63946", label=f"F_k = {F_k:.0f} kN (resultant)"),
        mpatches.Patch(color="#1d5fa8", label=f"N_k = {N_k:.1f} kN (langs stag)"),
        mpatches.Patch(color="#2A9D8F", label=f"V_k = {V_k:.1f} kN (⊥ stag)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              framealpha=0.85, edgecolor="#cccccc")

    ax.set_title(f"Kraftdekomponering  |  β = {beta_deg:.0f}°", fontsize=9,
                 color="#333333", pad=6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)

    w_pt = width_mm * mm
    # Behold proporsjoner
    img = Image(buf, width=w_pt, height=w_pt * 3.5 / 6)
    return img


# ---------------------------------------------------------------------------
# Bygger rapport-innhold
# ---------------------------------------------------------------------------

def _build_story(inp: dict, results: dict, project_info: dict, styles: dict):
    W = 160 * mm   # tekstbredde (A4 med 25 mm marginer)
    story = []

    # ---- Tittelblokk --------------------------------------------------------
    story.append(Paragraph("Beregningsnotat", styles["title"]))
    story.append(Paragraph(
        "Kapasitetsberegning – Passive forankringer i berg", styles["subtitle"]))
    story.append(Paragraph(
        "Kilde: NGI-rapport 20210114-01-R (Sikringshåndboken, NVE)", styles["subtitle"]))
    story.append(Spacer(1, 4 * mm))

    meta = [
        ["Prosjekt",   project_info.get("prosjekt", "–"),
         "Dato",       datetime.date.today().isoformat()],
        ["Utført av",  project_info.get("utfort_av", "–"),
         "Kontrollert av", project_info.get("kontrollert_av", "–")],
        ["Rev.",       project_info.get("revisjon", "0"),
         "Forankringstype", inp.get("type_label", "–")],
    ]
    story.append(_tbl(meta,
                      [35*mm, 45*mm, 35*mm, 45*mm],
                      [("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                       ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold")]))
    story.append(Spacer(1, 6 * mm))

    # ---- Inngangsdata -------------------------------------------------------
    story.append(Paragraph("1  Inngangsdata", styles["h1"]))
    story.append(_section_rule(styles))

    inp_rows = [["Parameter", "Symbol", "Verdi", "Enhet"]]
    inp_rows += [
        ["Karakteristisk kraft",             "F_k",      f"{inp['F_k']:.1f}",     "kN"],
        ["Vinkel (kraft vs stagakse)",        "β",        f"{inp['beta']:.1f}",    "°"],
        ["Lastfaktor aksialkraft",            "γ_F,N",    f"{inp['gamma_FN']:.2f}","–"],
        ["Lastfaktor skjærkraft",             "γ_F,V",    f"{inp['gamma_FV']:.2f}","–"],
        ["Materialfaktor stål",               "γ_s",      f"{inp['gamma_s']:.2f}", "–"],
        ["Materialfaktor heft",               "γ_τ",      f"{inp['gamma_tau']:.2f}","–"],
        ["Materialfaktor berg",               "γ_M",      f"{inp['gamma_M']:.2f}", "–"],
    ]
    if inp.get("mode") == "supplier":
        inp_rows += [
            ["Leverandør strekkraft",         "F_t,k",    f"{inp['F_tk']:.0f}",   "kN"],
            ["Leverandør skjærkraft",         "F_v,k",    f"{inp['F_vk']:.0f}",   "kN"],
        ]
    else:
        inp_rows += [
            ["Stålkvalitet",                  "–",        inp.get("steel_name","–"), "–"],
            ["Dimensjon",                     "–",        inp.get("dim_name","–"),   "–"],
            ["Spenningsareal",                "A_s",      f"{inp.get('A_s',0):.0f}","mm²"],
            ["Flytegrense",                   "f_yk",     f"{inp.get('f_yk',0):.0f}","MPa"],
        ]
    inp_rows += [
        ["Borehullsdiameter",                 "d_bh",     f"{inp.get('d_bh',0):.0f}","mm"],
        ["Innfestingslengde",                 "l_b",      f"{inp.get('l_b',0):.0f}","mm"],
        ["Terningfasthet injeksjonsmasse",    "f_ccube",  f"{inp.get('f_ccube',0):.0f}","MPa"],
        ["Heftfasthet berg (karakt.)",        "τ_k,berg", f"{inp.get('tau_k_berg',0):.3f}","MPa"],
        ["Kjegle skjærstyrke (karakt.)",     "τ_k,kjegle",f"{inp.get('tau_k_cone',0):.0f}","kPa"],
        ["Kjegle halvåpningsvinkel",          "ψ",        f"{inp.get('psi_deg',0):.0f}","°"],
    ]
    story.append(_tbl(inp_rows,
                      [65*mm, 22*mm, 30*mm, 20*mm],
                      [("FONTNAME", (0,0),(0,-1), "Helvetica-Bold")]))
    story.append(Spacer(1, 5 * mm))

    # ---- Kraftdekomponering -------------------------------------------------
    story.append(Paragraph("2  Kraftdekomponering", styles["h1"]))
    story.append(_section_rule(styles))

    N_k, V_k = decompose(inp["F_k"], inp["beta"])
    N_Ed, V_Ed = apply_load_factors(N_k, V_k, inp["gamma_FN"], inp["gamma_FV"])

    story.append(Paragraph(
        "Karakteristiske komponenter (langs / vinkelrett på stagaksen):", styles["body"]))
    story.append(Paragraph(
        f"N_k = F_k · cos β = {inp['F_k']:.1f} · cos({inp['beta']:.1f}°) = {N_k:.2f} kN",
        styles["formula"]))
    story.append(Paragraph(
        f"V_k = F_k · sin β = {inp['F_k']:.1f} · sin({inp['beta']:.1f}°) = {V_k:.2f} kN",
        styles["formula"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Dimensjonerende komponenter (etter lastfaktorer):", styles["body"]))
    story.append(Paragraph(
        f"N_Ed = γ_F,N · N_k = {inp['gamma_FN']:.2f} · {N_k:.2f} = {N_Ed:.2f} kN",
        styles["formula"]))
    story.append(Paragraph(
        f"V_Ed = γ_F,V · V_k = {inp['gamma_FV']:.2f} · {V_k:.2f} = {V_Ed:.2f} kN",
        styles["formula"]))
    story.append(Spacer(1, 4 * mm))

    # ---- Skisse -------------------------------------------------------------
    story.append(Paragraph("3  Kraftskisse", styles["h1"]))
    story.append(_section_rule(styles))
    try:
        sketch = _make_sketch_image(inp, width_mm=145)
        story.append(sketch)
    except Exception as e:
        story.append(Paragraph(f"(Skisse ikke tilgjengelig: {e})", styles["body"]))
    story.append(Spacer(1, 4 * mm))

    # ---- Indre kapasitet ----------------------------------------------------
    story.append(Paragraph("4  Indre kapasitet (kap. 4.3)", styles["h1"]))
    story.append(_section_rule(styles))

    R_itd_v = results["R_itd"]
    R_Vd_v  = results["R_Vd"]
    u_vm    = utilisation_von_mises(N_Ed, V_Ed, R_itd_v)
    u_ell   = utilisation_elliptic(N_Ed, V_Ed, R_itd_v, R_Vd_v)

    if inp.get("mode") == "supplier":
        story.append(Paragraph(
            f"R_itd = F_t,k / γ_s = {inp['F_tk']:.0f} / {inp['gamma_s']:.2f} = {R_itd_v:.2f} kN",
            styles["formula"]))
        story.append(Paragraph(
            f"R_Vd  = F_v,k / γ_s = {inp['F_vk']:.0f} / {inp['gamma_s']:.2f} = {R_Vd_v:.2f} kN",
            styles["formula"]))
    else:
        A_s  = inp.get("A_s", 0)
        f_yk = inp.get("f_yk", 0)
        story.append(Paragraph(
            f"R_itd = A_s · f_yk / γ_s = {A_s:.0f} · {f_yk:.0f} / {inp['gamma_s']:.2f} = {R_itd_v:.2f} kN",
            styles["formula"]))
        story.append(Paragraph(
            f"R_Vd  = A_s · f_yk / (√3 · γ_s) = {A_s:.0f} · {f_yk:.0f} / (1.732 · {inp['gamma_s']:.2f}) = {R_Vd_v:.2f} kN",
            styles["formula"]))

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"U (von Mises)  = √(N_Ed² + 3·V_Ed²) / R_itd "
        f"= √({N_Ed:.2f}² + 3·{V_Ed:.2f}²) / {R_itd_v:.2f} = {u_vm:.3f}",
        styles["formula"]))
    story.append(Paragraph(
        f"U (elliptisk)  = √((N/R_itd)² + (V/R_Vd)²) "
        f"= √(({N_Ed:.2f}/{R_itd_v:.2f})² + ({V_Ed:.2f}/{R_Vd_v:.2f})²) = {u_ell:.3f}",
        styles["formula"]))
    story.append(Spacer(1, 2 * mm))

    cap_rows = [
        ["Kontroll", "Verdi", "Grense", "Status"],
        ["U (von Mises) ≤ 1,0",  f"{u_vm:.3f}",  "1,000",
         "OK" if u_vm  <= 1.0 else "IKKE OK"],
        ["U (elliptisk) ≤ 1,0",  f"{u_ell:.3f}", "1,000",
         "OK" if u_ell <= 1.0 else "IKKE OK"],
    ]
    ok_cmds = []
    for i, row in enumerate(cap_rows[1:], 1):
        c = GREEN if row[3] == "OK" else RED
        ok_cmds.append(("TEXTCOLOR", (3, i), (3, i), c))
        ok_cmds.append(("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"))
    story.append(_tbl(cap_rows, [80*mm, 25*mm, 25*mm, 25*mm], ok_cmds))
    story.append(Spacer(1, 5 * mm))

    # ---- Innfestingskapasitet -----------------------------------------------
    story.append(Paragraph("5  Innfestingskapasitet (kap. 4.5)", styles["h1"]))
    story.append(_section_rule(styles))

    try:
        fck_v  = f_ck(inp["f_ccube"])
        fctk_v = f_ctk(fck_v)
        fctd_v = f_ctd(fctk_v)
        fbd_v  = f_bd(fctd_v)
        d_bolt = inp.get("d", inp.get("d_bh", 32))
        Rtg_v  = R_tg(fbd_v, d_bolt, inp["l_b"])
        tau_d  = inp["tau_k_berg"] / inp["gamma_tau"]
        Rgg_v  = R_gg(tau_d, inp["d_bh"], inp["l_b"])
        R_inn  = min(Rtg_v, Rgg_v)

        story.append(Paragraph(
            f"f_ck   = 0,8 · f_ccube = 0,8 · {inp['f_ccube']:.0f} = {fck_v:.2f} MPa",
            styles["formula"]))
        story.append(Paragraph(
            f"f_ctk  = 0,7 · 0,3 · f_ck^(2/3) = {fctk_v:.3f} MPa",
            styles["formula"]))
        story.append(Paragraph(
            f"f_ctd  = f_ctk / γ_c = {fctk_v:.3f} / 1,5 = {fctd_v:.3f} MPa",
            styles["formula"]))
        story.append(Paragraph(
            f"f_bd   = 2,25 · f_ctd = 2,25 · {fctd_v:.3f} = {fbd_v:.3f} MPa",
            styles["formula"]))
        story.append(Paragraph(
            f"R_tg   = π · d · l_b · f_bd = π · {d_bolt:.0f} · {inp['l_b']:.0f} · {fbd_v:.3f} / 1000 = {Rtg_v:.2f} kN",
            styles["formula"]))
        story.append(Paragraph(
            f"τ_d    = τ_k,berg / γ_τ = {inp['tau_k_berg']:.3f} / {inp['gamma_tau']:.2f} = {tau_d:.4f} MPa",
            styles["formula"]))
        story.append(Paragraph(
            f"R_gg   = π · d_bh · l_b · τ_d = π · {inp['d_bh']:.0f} · {inp['l_b']:.0f} · {tau_d:.4f} / 1000 = {Rgg_v:.2f} kN",
            styles["formula"]))
        story.append(Spacer(1, 2 * mm))

        inn_rows = [
            ["Kontroll", "Kapasitet [kN]", "Last [kN]", "Status"],
            ["R_tg ≥ N_Ed", f"{Rtg_v:.2f}", f"{N_Ed:.2f}",
             "OK" if Rtg_v >= N_Ed else "IKKE OK"],
            ["R_gg ≥ N_Ed", f"{Rgg_v:.2f}", f"{N_Ed:.2f}",
             "OK" if Rgg_v >= N_Ed else "IKKE OK"],
            ["R_inn,min ≥ N_Ed", f"{R_inn:.2f}", f"{N_Ed:.2f}",
             "OK" if R_inn >= N_Ed else "IKKE OK"],
        ]
        ok_cmds2 = []
        for i, row in enumerate(inn_rows[1:], 1):
            c = GREEN if row[3] == "OK" else RED
            ok_cmds2.append(("TEXTCOLOR", (3, i), (3, i), c))
            ok_cmds2.append(("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"))
        story.append(_tbl(inn_rows, [70*mm, 30*mm, 30*mm, 25*mm], ok_cmds2))
    except Exception as e:
        story.append(Paragraph(f"Innfesting: {e}", styles["body"]))
        Rtg_v = Rgg_v = None

    story.append(Spacer(1, 5 * mm))

    # ---- Ytre kapasitet -----------------------------------------------------
    story.append(Paragraph("6  Ytre kapasitet – kjeglemodell (kap. 4.6)", styles["h1"]))
    story.append(_section_rule(styles))

    try:
        lam = lambda_required(inp["gamma_M"], N_Ed, inp["tau_k_cone"], inp["psi_deg"])
        ok_cone = inp["l_b"] >= lam
        story.append(Paragraph(
            f"λ_req = √(γ_M · N_Ed · 1000 / (π · τ_k,kjegle/1000 · tan ψ))",
            styles["formula"]))
        story.append(Paragraph(
            f"      = √({inp['gamma_M']:.2f} · {N_Ed:.2f} · 1000 / (π · {inp['tau_k_cone']:.0f}/1000 · tan {inp['psi_deg']:.0f}°))",
            styles["formula"]))
        story.append(Paragraph(
            f"      = {lam:.0f} mm",
            styles["formula"]))
        story.append(Spacer(1, 2 * mm))
        cone_rows = [
            ["Kontroll", "Krevd [mm]", "Tilgjengelig [mm]", "Status"],
            ["l_b ≥ λ_req", f"{lam:.0f}", f"{inp['l_b']:.0f}",
             "OK" if ok_cone else "IKKE OK"],
        ]
        ok_cmds3 = [
            ("TEXTCOLOR", (3, 1), (3, 1), GREEN if ok_cone else RED),
            ("FONTNAME",  (3, 1), (3, 1), "Helvetica-Bold"),
        ]
        story.append(_tbl(cone_rows, [70*mm, 30*mm, 35*mm, 22*mm], ok_cmds3))
    except Exception as e:
        story.append(Paragraph(f"Kjegle: {e}", styles["body"]))
        lam = None

    story.append(Spacer(1, 5 * mm))

    # ---- Kjedekontroll ------------------------------------------------------
    story.append(Paragraph("7  Kjedekontroll", styles["h1"]))
    story.append(_section_rule(styles))
    story.append(Paragraph(
        "Kapasitetene skal øke fra stålkjerne til omstøpning til berg "
        "slik at brudd skjer i staget (R_itd ≤ R_tg ≤ R_gg).",
        styles["body"]))
    story.append(Spacer(1, 2 * mm))

    if Rtg_v is not None and Rgg_v is not None:
        ok_chain = (R_itd_v <= Rtg_v) and (Rtg_v <= Rgg_v)
        chain_rows = [
            ["Ledd", "Kapasitet [kN]", "Krav", "Status"],
            ["R_itd (stål)", f"{R_itd_v:.2f}", f"≤ R_tg = {Rtg_v:.2f}",
             "OK" if R_itd_v <= Rtg_v else "IKKE OK"],
            ["R_tg (stål–injeksjon)", f"{Rtg_v:.2f}", f"≤ R_gg = {Rgg_v:.2f}",
             "OK" if Rtg_v <= Rgg_v else "IKKE OK"],
            ["R_gg (injeksjon–berg)", f"{Rgg_v:.2f}", "≥ R_tg", "OK"],
        ]
        ok_cmds4 = []
        for i, row in enumerate(chain_rows[1:], 1):
            c = GREEN if row[3] == "OK" else RED
            ok_cmds4.append(("TEXTCOLOR", (3, i), (3, i), c))
            ok_cmds4.append(("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"))
        story.append(_tbl(chain_rows, [65*mm, 30*mm, 40*mm, 22*mm], ok_cmds4))

    story.append(Spacer(1, 5 * mm))

    # ---- Oppsummering -------------------------------------------------------
    story.append(Paragraph("8  Oppsummering", styles["h1"]))
    story.append(_section_rule(styles))

    all_checks = []
    all_checks.append(("U (von Mises) ≤ 1,0",   u_vm  <= 1.0, f"U = {u_vm:.3f}"))
    all_checks.append(("U (elliptisk) ≤ 1,0",   u_ell <= 1.0, f"U = {u_ell:.3f}"))
    if Rtg_v is not None:
        all_checks.append(("R_tg ≥ N_Ed",         Rtg_v >= N_Ed, f"{Rtg_v:.1f} ≥ {N_Ed:.1f} kN"))
        all_checks.append(("R_gg ≥ N_Ed",         Rgg_v >= N_Ed, f"{Rgg_v:.1f} ≥ {N_Ed:.1f} kN"))
        all_checks.append(("Kjedekontroll",        ok_chain,      f"R_itd={R_itd_v:.1f} ≤ R_tg={Rtg_v:.1f} ≤ R_gg={Rgg_v:.1f} kN"))
    if lam is not None:
        all_checks.append(("l_b ≥ λ_req (kjegle)",ok_cone,       f"{inp['l_b']:.0f} ≥ {lam:.0f} mm"))

    sum_rows = [["Kontroll", "Verdi", "Status"]]
    sum_cmds = []
    for i, (label, ok, val) in enumerate(all_checks, 1):
        sum_rows.append([label, val, "OK" if ok else "IKKE OK"])
        c = GREEN if ok else RED
        sum_cmds.append(("TEXTCOLOR", (2, i), (2, i), c))
        sum_cmds.append(("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"))

    story.append(KeepTogether([
        _tbl(sum_rows, [80*mm, 55*mm, 22*mm], sum_cmds)
    ]))

    all_ok = all(ok for _, ok, _ in all_checks)
    story.append(Spacer(1, 3 * mm))
    if all_ok:
        story.append(Paragraph(
            "Alle kontroller er tilfredsstilt. Forankringen er tilstrekkelig dimensjonert.",
            styles["ok"]))
    else:
        story.append(Paragraph(
            "En eller flere kontroller er IKKE tilfredsstilt. Revidering av dimensjoner er nødvendig.",
            styles["nok"]))

    return story


# ---------------------------------------------------------------------------
# Hoved-funksjon
# ---------------------------------------------------------------------------

def generate_pdf(inp: dict, results: dict, project_info: dict = None) -> bytes:
    """
    Genererer et PDF-beregningsnotat.

    Args:
        inp:          Dict med alle inngangsdata (fra sidebar).
        results:      Dict med {'R_itd': float, 'R_Vd': float}.
        project_info: Valgfri dict med {'prosjekt', 'utfort_av',
                      'kontrollert_av', 'revisjon'}.

    Returns:
        PDF som bytes.
    """
    if project_info is None:
        project_info = {}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=25*mm, rightMargin=25*mm,
        topMargin=22*mm, bottomMargin=22*mm,
        title="Beregningsnotat – Passive forankringer i berg",
        author=project_info.get("utfort_av", ""),
    )

    styles = _styles()
    story  = _build_story(inp, results, project_info, styles)
    doc.build(story)
    return buf.getvalue()
