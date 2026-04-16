"""
Streamlit-modul for kapasitetsberegning av passive forankringer i berg.
Kilde: NGI-rapport 20210114-01-R (Sikringshåndboken, NVE)
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.pdf_report import generate_pdf
from apps.anchor_calculations import (
    stress_area, r_itd, r_vd, r_itd_supplier, r_vd_supplier,
    decompose, apply_load_factors,
    utilisation_von_mises, utilisation_elliptic, critical_angle,
    f_ck, f_ctk, f_ctd, f_bd, R_tg, R_gg,
    lambda_required,
    sensitivity_beta, sensitivity_parameter,
)
from data.steel_tables import (
    STEEL_GRADES, STRESS_AREAS, ROCK_BOND, CONE_PARAMS,
    SELF_DRILLING_ANCHORS,
)

# ---------------------------------------------------------------------------
# Hjelpere
# ---------------------------------------------------------------------------

def _status_box(label: str, value: str, ok: bool):
    if ok:
        st.success(f"**{label}:** {value} ✓")
    else:
        st.error(f"**{label}:** {value} ✗")


def _metric_ok(label: str, value: str, ok: bool):
    delta = "OK" if ok else "IKKE OK"
    delta_color = "normal" if ok else "inverse"
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ---------------------------------------------------------------------------
# Sidebar-inputs (returnerer dict med alle inputs)
# ---------------------------------------------------------------------------

def _sidebar_common(prefix: str) -> dict:
    """Partialfaktorer og last — felles for alle faner."""
    st.subheader("Partialfaktorer")
    gamma_s  = st.number_input("γ_s (stål)",       0.5, 3.0, 1.35, 0.01,  key=f"{prefix}_gs")
    gamma_tau= st.number_input("γ_τ (heft)",        0.5, 5.0, 1.25, 0.05,  key=f"{prefix}_gtau")
    gamma_M  = st.number_input("γ_M (berg)",        0.5, 5.0, 2.0,  0.1,   key=f"{prefix}_gM")
    gamma_FN = st.number_input("γ_F,N (aksialkraft)",0.5, 3.0, 1.5,  0.05,  key=f"{prefix}_gFN")
    gamma_FV = st.number_input("γ_F,V (skjærkraft)", 0.5, 3.0, 1.5,  0.05,  key=f"{prefix}_gFV")

    st.subheader("Last")
    st.caption("β = vinkel mellom kraft og stagaksen i vertikalplanet (0° = rent strekk, 90° = rent skjær)")
    F_k     = st.number_input("F_k [kN]",    1.0, 5000.0, 150.0, 10.0,  key=f"{prefix}_Fk")
    beta    = st.number_input("β [grader]",  0.0,   90.0,  45.0,  1.0,  key=f"{prefix}_beta")

    return dict(gamma_s=gamma_s, gamma_tau=gamma_tau, gamma_M=gamma_M,
                gamma_FN=gamma_FN, gamma_FV=gamma_FV, F_k=F_k, beta=beta)


def _sidebar_steel_section(prefix: str, default_beta: float = 45.0) -> dict:
    """Stål- og dimensjonsvalg for fane 1 og 3."""
    # Stålkvalitet
    steel_names = STEEL_GRADES["Ståltype"].tolist()
    steel_idx   = st.selectbox("Stålkvalitet", range(len(steel_names)),
                               format_func=lambda i: steel_names[i],
                               key=f"{prefix}_steel")
    row_steel   = STEEL_GRADES.iloc[steel_idx]
    f_yk_val    = float(row_steel["f_yk"])
    f_uk_val    = float(row_steel["f_uk"])
    st.caption(f"f_yk = {f_yk_val} MPa  |  f_uk = {f_uk_val} MPa")

    # Dimensjon
    dim_names = STRESS_AREAS["Betegnelse"].tolist()
    dim_idx   = st.selectbox("Bolte-/stagdimensjon", range(len(dim_names)),
                             format_func=lambda i: dim_names[i],
                             index=dim_names.index("M30") if "M30" in dim_names else 0,
                             key=f"{prefix}_dim")
    row_dim   = STRESS_AREAS.iloc[dim_idx]
    d_val     = float(row_dim["d_major"])
    A_s_val   = float(row_dim["A_s"])
    st.caption(f"d = {d_val} mm  |  A_s = {A_s_val} mm²")

    return dict(f_yk=f_yk_val, f_uk=f_uk_val, d=d_val, A_s=A_s_val,
                steel_name=row_steel["Ståltype"], dim_name=row_dim["Betegnelse"])


def _sidebar_grouting(prefix: str) -> dict:
    """Innfestings- og bergparametere."""
    st.subheader("Innfesting")
    f_ccube = st.number_input("f_ccube injeksjonsmasse [MPa]", 10.0, 100.0, 35.0, 5.0, key=f"{prefix}_fcc")
    d_bh    = st.number_input("Borehullsdiameter d_bh [mm]",  20.0, 200.0, 46.0, 1.0, key=f"{prefix}_dbh")
    l_b     = st.number_input("Innfestingslengde l_b [mm]",   100.0, 5000.0, 1000.0, 50.0, key=f"{prefix}_lb")

    st.subheader("Bergparametere")
    st.caption("Velg fra tabell — verdier kan overstyres i feltene under.")

    # -- Heftfasthet mørtel–berg --
    rock_names = ROCK_BOND["Bergart"].tolist()
    rock_idx   = st.selectbox("Bergart (heftfasthet)", range(len(rock_names)),
                              format_func=lambda i: rock_names[i],
                              key=f"{prefix}_rock")
    tau_k_berg_tbl = float(ROCK_BOND.iloc[rock_idx]["tau_k_MPa"])
    tau_k_berg = st.number_input(
        "τ_k,berg [MPa]  ← override",
        min_value=0.01, max_value=10.0,
        value=tau_k_berg_tbl, step=0.1,
        key=f"{prefix}_rock_override_{rock_idx}",
        help="Karakteristisk heftfasthet mørtel–berg (Internrapport 2374). "
             "Endres automatisk ved nytt bergartsvalg, men kan overstyres manuelt.",
    )

    st.divider()

    # -- Kjeglemodell --
    cone_names = CONE_PARAMS["Bergkvalitet"].tolist()
    cone_idx   = st.selectbox("Bergkvalitet (kjeglemodell)", range(len(cone_names)),
                              format_func=lambda i: cone_names[i],
                              key=f"{prefix}_cone")
    row_cone      = CONE_PARAMS.iloc[cone_idx]
    tau_k_min_tbl = float(row_cone["tau_k_min"])
    tau_k_max_tbl = float(row_cone["tau_k_max"])
    psi_max_tbl   = float(row_cone["psi_max"])

    st.caption(
        f"Tabellanbefalte verdier:  "
        f"τ_k = {tau_k_min_tbl}–{tau_k_max_tbl} kPa  |  ψ_maks = {psi_max_tbl}°"
    )
    c1, c2 = st.columns(2)
    tau_k_cone = c1.number_input(
        "τ_k,kjegle [kPa]  ← override",
        min_value=1.0, max_value=2000.0,
        value=tau_k_min_tbl, step=10.0,
        key=f"{prefix}_cone_tau_{cone_idx}",
        help="Karakteristisk skjærstyrke langs kjegleflaten. "
             "Standard: konservativ minimumsverdi fra tabell.",
    )
    psi_deg = c2.number_input(
        "ψ [grader]  ← override",
        min_value=1.0, max_value=89.0,
        value=psi_max_tbl, step=1.0,
        key=f"{prefix}_cone_psi_{cone_idx}",
        help="Halvåpningsvinkel for uttrekkskjeglen. "
             "Skal ikke overstige tabellanbefalte ψ_maks.",
    )
    if psi_deg > psi_max_tbl:
        st.warning(
            f"ψ = {psi_deg:.0f}° overskrider tabellanbefalingen ψ_maks = {psi_max_tbl:.0f}° "
            f"for valgt bergkvalitet."
        )

    return dict(f_ccube=f_ccube, d_bh=d_bh, l_b=l_b,
                tau_k_berg=tau_k_berg, tau_k_cone=tau_k_cone, psi_deg=psi_deg)


# ---------------------------------------------------------------------------
# Tab: Kapasitet
# ---------------------------------------------------------------------------

def _tab_capacity(inp: dict, R_itd_val: float, R_Vd_val: float, mode: str):
    N_k, V_k = decompose(inp["F_k"], inp["beta"])
    N_Ed, V_Ed = apply_load_factors(N_k, V_k, inp["gamma_FN"], inp["gamma_FV"])

    u_vm  = utilisation_von_mises(N_Ed, V_Ed, R_itd_val)
    u_ell = utilisation_elliptic(N_Ed, V_Ed, R_itd_val, R_Vd_val)

    col1, col2, col3 = st.columns(3)

    # -- Indre kapasitet --
    with col1:
        st.markdown("### Indre kapasitet")
        st.metric("R_itd [kN]", f"{R_itd_val:.1f}")
        st.metric("R_Vd [kN]",  f"{R_Vd_val:.1f}")
        st.metric("N_Ed [kN]",  f"{N_Ed:.1f}")
        st.metric("V_Ed [kN]",  f"{V_Ed:.1f}")
        st.divider()
        _metric_ok("U (von Mises)", f"{u_vm:.3f}", u_vm <= 1.0)
        _metric_ok("U (elliptisk)", f"{u_ell:.3f}", u_ell <= 1.0)

    # -- Innfestingskapasitet --
    with col2:
        st.markdown("### Innfestingskapasitet")
        try:
            fck_v   = f_ck(inp["f_ccube"])
            fctk_v  = f_ctk(fck_v)
            fctd_v  = f_ctd(fctk_v)
            fbd_v   = f_bd(fctd_v)
            d_bolt  = inp.get("d", inp.get("d_bh", 32.0))
            R_tg_v  = R_tg(fbd_v, d_bolt, inp["l_b"])
            tau_d   = inp["tau_k_berg"] / inp["gamma_tau"]
            R_gg_v  = R_gg(tau_d, inp["d_bh"], inp["l_b"])
            R_inn   = min(R_tg_v, R_gg_v)
            st.metric("f_bd [MPa]",  f"{fbd_v:.2f}")
            st.metric("R_tg [kN]",   f"{R_tg_v:.1f}")
            st.metric("τ_d [MPa]",   f"{tau_d:.3f}")
            st.metric("R_gg [kN]",   f"{R_gg_v:.1f}")
            st.divider()
            _metric_ok("R_inn,min [kN]", f"{R_inn:.1f}", R_inn >= N_Ed)
        except Exception as e:
            st.warning(f"Innfesting: {e}")
            R_tg_v = R_gg_v = R_inn = None

    # -- Ytre kapasitet --
    with col3:
        st.markdown("### Ytre kapasitet (kjegle)")
        try:
            lam_req = lambda_required(
                inp["gamma_M"], N_Ed, inp["tau_k_cone"], inp["psi_deg"]
            )
            ok_cone = inp["l_b"] >= lam_req
            st.metric("λ_req [mm]", f"{lam_req:.0f}")
            st.metric("l_b [mm]",   f"{inp['l_b']:.0f}")
            _metric_ok("Kjegle OK?", "Ja" if ok_cone else "Nei", ok_cone)
        except Exception as e:
            st.warning(f"Kjegle: {e}")
            R_tg_v = R_gg_v = None

    # -- Kjedekontroll --
    st.divider()
    st.markdown("#### Kjedekontroll")
    if R_tg_v is not None and R_gg_v is not None:
        ok_chain = (R_itd_val <= R_tg_v) and (R_tg_v <= R_gg_v)
        msg = (
            f"R_itd ({R_itd_val:.1f} kN) ≤ R_tg ({R_tg_v:.1f} kN) "
            f"≤ R_gg ({R_gg_v:.1f} kN)"
        )
        if ok_chain:
            st.success(f"✓ Kjedekontroll OK:  {msg}")
        else:
            st.error(f"✗ Kjedekontroll IKKE OK:  {msg}")


# ---------------------------------------------------------------------------
# Tab: Skisse
# ---------------------------------------------------------------------------

def _tab_sketch(inp: dict):
    F_k   = inp["F_k"]
    beta  = inp["beta"]
    N_k, V_k = decompose(F_k, beta)
    N_Ed, V_Ed = apply_load_factors(N_k, V_k, inp["gamma_FN"], inp["gamma_FV"])

    beta_r = math.radians(beta)
    scale  = 1.0

    # Stagaksen langs x
    ax_len = 2.5
    # Kraft-pil (fra origo, vinkel beta over horisontalen)
    Fx =  F_k * math.cos(beta_r) * scale
    Fy =  F_k * math.sin(beta_r) * scale
    # Normaliser
    fmax = max(abs(Fx), abs(Fy), F_k, 1.0)
    sc   = 2.0 / fmax

    fig = go.Figure()

    # Stagakse
    fig.add_shape(type="line", x0=-ax_len, y0=0, x1=ax_len, y1=0,
                  line=dict(color="#555", width=3))
    fig.add_annotation(x=ax_len + 0.2, y=0, text="Stagakse", showarrow=False,
                       font=dict(size=12))

    # Resultantkraft F
    fx = F_k * math.cos(beta_r) * sc
    fy = F_k * math.sin(beta_r) * sc
    fig.add_annotation(x=fx, y=fy, ax=0, ay=0,
                       xref="x", yref="y", axref="x", ayref="y",
                       arrowhead=3, arrowwidth=2, arrowcolor="#E63946",
                       text=f"F_k={F_k:.0f} kN", font=dict(color="#E63946", size=12),
                       showarrow=True)

    # N-komponent (langs x)
    nx = N_k * sc
    fig.add_annotation(x=nx, y=0, ax=0, ay=0,
                       xref="x", yref="y", axref="x", ayref="y",
                       arrowhead=3, arrowwidth=2, arrowcolor="#457B9D",
                       text=f"N_k={N_k:.1f} kN", font=dict(color="#457B9D", size=11),
                       showarrow=True)

    # V-komponent (langs y)
    vy = V_k * sc
    fig.add_annotation(x=0, y=vy, ax=0, ay=0,
                       xref="x", yref="y", axref="x", ayref="y",
                       arrowhead=3, arrowwidth=2, arrowcolor="#2A9D8F",
                       text=f"V_k={V_k:.1f} kN", font=dict(color="#2A9D8F", size=11),
                       showarrow=True)

    # Stiplet linjer til komponenter
    fig.add_shape(type="line", x0=fx, y0=fy, x1=nx, y1=0,
                  line=dict(dash="dot", color="#aaa", width=1))
    fig.add_shape(type="line", x0=fx, y0=fy, x1=0, y1=vy,
                  line=dict(dash="dot", color="#aaa", width=1))

    # Vinkelbue
    if beta > 0.5:
        arc_r = 0.5
        theta = np.linspace(0, beta_r, 60)
        fig.add_trace(go.Scatter(
            x=arc_r * np.cos(theta), y=arc_r * np.sin(theta),
            mode="lines", line=dict(color="#E63946", dash="dot", width=1),
            showlegend=False
        ))
        mid = beta_r / 2
        fig.add_annotation(
            x=arc_r * 1.4 * math.cos(mid), y=arc_r * 1.4 * math.sin(mid),
            text=f"β={beta:.0f}°", showarrow=False,
            font=dict(color="#E63946", size=12)
        )

    fig.update_layout(
        xaxis=dict(range=[-ax_len - 0.5, ax_len + 1.5], zeroline=False, showgrid=False, visible=False),
        yaxis=dict(range=[-1.0, ax_len], zeroline=False, showgrid=False, visible=False, scaleanchor="x"),
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        title=f"Kraftdekomponering  |  β={beta:.0f}°  |  F_k={F_k:.0f} kN",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rødt = resultant F_k, blått = aksialkraft N_k (langs stag), grønt = skjærkraft V_k (⊥ stag)")


# ---------------------------------------------------------------------------
# Tab: Sensitivitet
# ---------------------------------------------------------------------------

def _tab_sensitivity(inp: dict, R_itd_val: float, R_Vd_val: float, mode: str):
    sub1, sub2 = st.tabs(["Vinkelanalyse (β)", "Parameteranalyse"])

    # -- Vinkelanalyse --
    with sub1:
        df_vm  = sensitivity_beta(inp["F_k"] * inp["gamma_FN"], R_itd_val, R_Vd_val,
                                  method="von_mises")
        df_ell = sensitivity_beta(inp["F_k"] * inp["gamma_FN"], R_itd_val, R_Vd_val,
                                  method="elliptic")

        beta_crit_vm  = critical_angle(inp["F_k"] * inp["gamma_FN"], R_itd_val, R_Vd_val, "von_mises")
        beta_crit_ell = critical_angle(inp["F_k"] * inp["gamma_FN"], R_itd_val, R_Vd_val, "elliptic")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vm["beta_deg"],  y=df_vm["U"],
                                 name="von Mises", line=dict(color="#E63946")))
        fig.add_trace(go.Scatter(x=df_ell["beta_deg"], y=df_ell["U"],
                                 name="Elliptisk", line=dict(color="#457B9D")))
        fig.add_hline(y=1.0, line_color="red", line_dash="dash",
                      annotation_text="U = 1.0", annotation_position="right")
        fig.add_vline(x=inp["beta"], line_color="#2A9D8F", line_dash="dot",
                      annotation_text=f"β={inp['beta']:.0f}°")
        if beta_crit_ell < 90:
            fig.add_annotation(x=beta_crit_ell, y=1.05,
                               text=f"β_crit={beta_crit_ell:.1f}°",
                               showarrow=True, arrowhead=2, font=dict(color="#457B9D"))
        fig.update_layout(
            xaxis_title="β [grader]", yaxis_title="Utnyttelse U [-]",
            height=420, margin=dict(l=10, r=10, t=40, b=10),
            title="Utnyttelse vs vinkel β"
        )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("β_crit (von Mises)",  f"{beta_crit_vm:.1f}°")
        c2.metric("β_crit (elliptisk)",  f"{beta_crit_ell:.1f}°")

    # -- Parameteranalyse --
    with sub2:
        param_opts = {
            "F_k [kN]":       "F_k",
            "γ_s [-]":        "gamma_s",
            "l_b [mm]":       "l_b",
            "f_yk [MPa]":     "f_yk",
            "τ_k,berg [MPa]": "tau_k_berg",
        }
        if mode == "supplier":
            param_opts["F_tk [kN]"] = "F_tk"
            param_opts["F_vk [kN]"] = "F_vk"

        param_label = st.selectbox("Parameter å variere", list(param_opts.keys()), key=f"{mode}_sens_param")
        param_key   = param_opts[param_label]

        cur_val = inp.get(param_key, 1.0)
        lo_def  = max(cur_val * 0.5, 0.01)
        hi_def  = cur_val * 2.0
        c1, c2, c3 = st.columns(3)
        p_min  = c1.number_input("Min", value=float(lo_def), key=f"{mode}_pmin")
        p_max  = c2.number_input("Maks", value=float(hi_def), key=f"{mode}_pmax")
        n_pts  = c3.number_input("Punkter", 10, 200, 50, 10, key=f"{mode}_npts")

        fixed = {**inp, "mode": mode,
                 "A_s": inp.get("A_s", 561.0),
                 "f_yk": inp.get("f_yk", 500.0),
                 "F_tk": inp.get("F_tk", 525.0),
                 "F_vk": inp.get("F_vk", 262.0)}
        try:
            df_p = sensitivity_parameter(param_key, (p_min, p_max), fixed,
                                         method="elliptic", n_pts=int(n_pts))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_p["param_value"], y=df_p["U"],
                                      mode="lines", name="U"))
            fig2.add_hline(y=1.0, line_color="red", line_dash="dash")
            fig2.add_vline(x=cur_val, line_color="#2A9D8F", line_dash="dot",
                           annotation_text="Nåværende")
            fig2.update_layout(
                xaxis_title=param_label, yaxis_title="U [-]",
                height=380, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Sensitivitet: {e}")


# ---------------------------------------------------------------------------
# Tab: Tabeller
# ---------------------------------------------------------------------------

def _pdf_download_button(inp: dict, R_itd_v: float, R_Vd_v: float, key: str):
    """Ekspandérbar prosjektinfo + nedlastningsknapp for PDF-rapport."""
    st.divider()
    with st.expander("Last ned beregningsnotat (PDF)", expanded=False):
        c1, c2 = st.columns(2)
        prosjekt  = c1.text_input("Prosjektnavn",     key=f"{key}_proj")
        utfort_av = c2.text_input("Utført av",        key=f"{key}_by")
        kontroll  = c1.text_input("Kontrollert av",   key=f"{key}_ctrl")
        revisjon  = c2.text_input("Revisjon", value="0", key=f"{key}_rev")
        if st.button("Generer PDF", key=f"{key}_gen"):
            project_info = dict(prosjekt=prosjekt, utfort_av=utfort_av,
                                kontrollert_av=kontroll, revisjon=revisjon)
            try:
                pdf_bytes = generate_pdf(
                    inp=inp,
                    results={"R_itd": R_itd_v, "R_Vd": R_Vd_v},
                    project_info=project_info,
                )
                st.download_button(
                    label="Last ned PDF",
                    data=pdf_bytes,
                    file_name="beregningsnotat_forankring.pdf",
                    mime="application/pdf",
                    key=f"{key}_dl",
                )
            except Exception as e:
                st.error(f"PDF-feil: {e}")


def _tab_tables():
    t1, t2, t3, t4, t5 = st.tabs([
        "Stålkvaliteter", "Spenningsareal", "Heftfasthet berg",
        "Kjeglemodell", "Selvborende stag"
    ])
    with t1:
        st.dataframe(STEEL_GRADES, use_container_width=True)
    with t2:
        st.dataframe(STRESS_AREAS, use_container_width=True)
    with t3:
        st.dataframe(ROCK_BOND, use_container_width=True)
    with t4:
        st.dataframe(CONE_PARAMS, use_container_width=True)
    with t5:
        st.dataframe(SELF_DRILLING_ANCHORS, use_container_width=True)


# ---------------------------------------------------------------------------
# Parameterforklaring
# ---------------------------------------------------------------------------

_PARAM_GLOSSARY = [
    ("Laster og geometri", [
        ("F_k",      "kN",      "Karakteristisk resultantkraft i forankringen (ufalktorert)."),
        ("F_Ed",     "kN",      "Dimensjonerende resultantkraft = γ_F · F_k."),
        ("β",        "grader",  "Vinkel mellom kraftretningen og stagaksen i vertikalplanet. β = 0° gir rent strekk, β = 90° gir rent skjær."),
        ("N_k / N_Ed", "kN",   "Karakteristisk / dimensjonerende aksialkraft langs stagaksen: N = F · cos β."),
        ("V_k / V_Ed", "kN",   "Karakteristisk / dimensjonerende skjærkraft vinkelrett på stagaksen: V = F · sin β."),
        ("l_b",      "mm",      "Innfestingslengde — lengden av staget som er omstøpt/injisert i berg."),
        ("d",        "mm",      "Nominell ytre diameter på bolt eller stag."),
        ("d_bh",     "mm",      "Borehullsdiameter."),
    ]),
    ("Partialfaktorer", [
        ("γ_s",      "–",       "Materialfaktor for stål. Standardverdi 1,35 etter NS-EN 1993."),
        ("γ_τ",      "–",       "Materialfaktor for heftfasthet mellom injeksjonsmasse og berg. Standardverdi 1,25 (Internrapport 2374)."),
        ("γ_M",      "–",       "Materialfaktor for berg i kjeglemodellen. Standardverdi 2,0."),
        ("γ_F,N",    "–",       "Lastfaktor for aksialkraftkomponenten N. Standardverdi 1,5."),
        ("γ_F,V",    "–",       "Lastfaktor for skjærkraftkomponenten V. Standardverdi 1,5."),
    ]),
    ("Stål og tverrsnitt", [
        ("f_yk",     "MPa",     "Karakteristisk flytegrense for stål (nedre grense)."),
        ("f_uk",     "MPa",     "Karakteristisk bruddgrense for stål."),
        ("A_s",      "mm²",     "Spenningsareal for metrisk grovgjenge etter ISO 262: A_s = π/4 · (d − 0,9382 · p)²."),
        ("F_t,k",    "kN",      "Karakteristisk strekkraft oppgitt av leverandør for selvborende stag."),
        ("F_v,k",    "kN",      "Karakteristisk skjærkraft oppgitt av leverandør for selvborende stag."),
    ]),
    ("Indre kapasitet (kap. 4.3)", [
        ("R_itd",    "kN",      "Dimensjonerende strekkapasitet for staget: R_itd = A_s · f_yk / γ_s."),
        ("R_Vd",     "kN",      "Dimensjonerende skjærkapasitet etter von Mises: R_Vd = A_s · f_yk / (√3 · γ_s)."),
        ("U (von Mises)", "–",  "Utnyttelsesgrad etter von Mises-kriteriet: U = √(N² + 3V²) / R_itd. Kap. 4.3.2."),
        ("U (elliptisk)", "–",  "Utnyttelsesgrad etter elliptisk interaksjonsformel: U = √((N/R_itd)² + (V/R_Vd)²)."),
        ("β_crit",   "grader",  "Grensevinkel der utnyttelsen U = 1,0. Vinkler over denne gir brudd."),
    ]),
    ("Innfestingskapasitet (kap. 4.5)", [
        ("f_ccube",  "MPa",     "Karakteristisk terningfasthet for injeksjonsmassen (28-dagers)."),
        ("f_ck",     "MPa",     "Karakteristisk sylinderfasthet: f_ck = 0,8 · f_ccube."),
        ("f_ctk",    "MPa",     "Karakteristisk strekkfasthet for injeksjonsmassen: f_ctk = 0,7 · 0,3 · f_ck^(2/3)."),
        ("f_ctd",    "MPa",     "Dimensjonerende strekkfasthet: f_ctd = f_ctk / γ_c (γ_c = 1,5)."),
        ("f_bd",     "MPa",     "Dimensjonerende heftfasthet stål–injeksjonsmasse: f_bd = 2,25 · f_ctd."),
        ("R_tg",     "kN",      "Kapasitet for heftbrudd ved stål–injeksjonsmasse-grenseflaten: R_tg = π · d · l_b · f_bd."),
        ("τ_k,berg", "MPa",     "Karakteristisk heftfasthet mellom injeksjonsmasse og berg (fra bergartstabellen)."),
        ("τ_d",      "MPa",     "Dimensjonerende heftfasthet berg: τ_d = τ_k,berg / γ_τ."),
        ("R_gg",     "kN",      "Kapasitet for heftbrudd ved injeksjonsmasse–berg-grenseflaten: R_gg = π · d_bh · l_b · τ_d."),
        ("R_inn,min","kN",      "Styrende innfestingskapasitet = min(R_tg, R_gg). Skal være ≥ N_Ed."),
    ]),
    ("Ytre kapasitet — kjeglemodell (kap. 4.6)", [
        ("τ_k,kjegle","kPa",   "Karakteristisk skjærstyrke langs kjegleflaten i berg. Avhenger av bergkvalitet (RQD)."),
        ("ψ",        "grader",  "Halvåpningsvinkel for uttrekkskjeglen. Typisk 30–45° avhengig av bergkvalitet."),
        ("λ_req",    "mm",      "Nødvendig forankringslengde fra kjeglemodellen: λ = √(γ_M · P_p · 1000 / (π · τ · tan ψ)). Skal være ≤ l_b."),
    ]),
    ("Kjedekontroll", [
        ("Kjedekontroll", "–",  "Kontrollerer at kapasitetene øker fra stål til omstøpning til berg: R_itd ≤ R_tg ≤ R_gg. Dette sikrer at bruddet skjer i staget og ikke i heftsonene."),
    ]),
]


def _param_glossary():
    st.divider()
    with st.expander("Parameterforklaring", expanded=False):
        for group_name, params in _PARAM_GLOSSARY:
            st.markdown(f"**{group_name}**")
            rows = [
                {"Symbol": sym, "Enhet": unit, "Forklaring": desc}
                for sym, unit, desc in params
            ]
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Symbol":      st.column_config.TextColumn(width="small"),
                    "Enhet":       st.column_config.TextColumn(width="small"),
                    "Forklaring":  st.column_config.TextColumn(width="large"),
                },
            )
            st.markdown("")


# ---------------------------------------------------------------------------
# Fane 1 – Stangstål
# ---------------------------------------------------------------------------

def _render_bar_steel():
    with st.sidebar:
        st.markdown("---")
        p = _sidebar_common("bar")
        st.subheader("Stål og dimensjon")
        s = _sidebar_steel_section("bar")
        st.subheader("Innfesting og berg")
        g = _sidebar_grouting("bar")

    inp = {**p, **s, **g}

    N_k, V_k = decompose(inp["F_k"], inp["beta"])
    N_Ed, V_Ed = apply_load_factors(N_k, V_k, inp["gamma_FN"], inp["gamma_FV"])
    R_itd_v = r_itd(inp["A_s"], inp["f_yk"], inp["gamma_s"])
    R_Vd_v  = r_vd(inp["A_s"], inp["f_yk"], inp["gamma_s"])
    inp["mode"] = "steel"
    inp["type_label"] = "Stangstål"

    tc, ts, tsen, ttab = st.tabs(["Kapasitet", "Skisse", "Sensitivitet", "Tabeller"])
    with tc:
        _tab_capacity(inp, R_itd_v, R_Vd_v, "steel")
    with ts:
        _tab_sketch(inp)
    with tsen:
        _tab_sensitivity(inp, R_itd_v, R_Vd_v, "steel")
    with ttab:
        _tab_tables()
    _pdf_download_button(inp, R_itd_v, R_Vd_v, key="bar")
    _param_glossary()


# ---------------------------------------------------------------------------
# Fane 2 – Selvborende stag (leverandør)
# ---------------------------------------------------------------------------

def _render_supplier():
    with st.sidebar:
        st.markdown("---")
        p = _sidebar_common("sup")

        st.subheader("Leverandørdata")
        prod_names = SELF_DRILLING_ANCHORS["Produkt"].tolist()
        prod_idx   = st.selectbox("Produkt", range(len(prod_names)),
                                  format_func=lambda i: prod_names[i],
                                  key="sup_prod")
        row_prod   = SELF_DRILLING_ANCHORS.iloc[prod_idx]
        F_tk_def   = float(row_prod["F_tk"])
        F_vk_def   = float(row_prod["F_vk"])
        d_def      = float(row_prod["Ytre_diam"])

        F_tk = st.number_input("F_t,k [kN]", 10.0, 5000.0, F_tk_def, 10.0, key="sup_Ftk")
        F_vk = st.number_input("F_v,k [kN]", 10.0, 5000.0, F_vk_def, 10.0, key="sup_Fvk")

        st.subheader("Innfesting og berg")
        g = _sidebar_grouting("sup")
        g["d"] = d_def

    inp = {**p, **g, "F_tk": F_tk, "F_vk": F_vk, "d": d_def,
           "mode": "supplier", "type_label": "Selvborende stag (leverandør)"}

    R_itd_v = r_itd_supplier(F_tk, inp["gamma_s"])
    R_Vd_v  = r_vd_supplier(F_vk, inp["gamma_s"])

    tc, ts, tsen, ttab = st.tabs(["Kapasitet", "Skisse", "Sensitivitet", "Tabeller"])
    with tc:
        _tab_capacity(inp, R_itd_v, R_Vd_v, "supplier")
    with ts:
        _tab_sketch(inp)
    with tsen:
        _tab_sensitivity(inp, R_itd_v, R_Vd_v, "supplier")
    with ttab:
        _tab_tables()
    _pdf_download_button(inp, R_itd_v, R_Vd_v, key="sup")
    _param_glossary()


# ---------------------------------------------------------------------------
# Fane 3 – Innstøpt bergbolt
# ---------------------------------------------------------------------------

def _render_rock_bolt():
    with st.sidebar:
        st.markdown("---")
        p = _sidebar_common("rb")
        # Override default beta to 0
        if "rb_beta" not in st.session_state:
            st.session_state["rb_beta"] = 0.0
        st.subheader("Stål og dimensjon")
        s = _sidebar_steel_section("rb")
        st.subheader("Innfesting og berg")
        g = _sidebar_grouting("rb")

    inp = {**p, **s, **g, "mode": "steel", "type_label": "Innstøpt bergbolt"}

    N_k, V_k = decompose(inp["F_k"], inp["beta"])
    N_Ed, V_Ed = apply_load_factors(N_k, V_k, inp["gamma_FN"], inp["gamma_FV"])
    R_itd_v = r_itd(inp["A_s"], inp["f_yk"], inp["gamma_s"])
    R_Vd_v  = r_vd(inp["A_s"], inp["f_yk"], inp["gamma_s"])

    tc, ts, tsen, ttab = st.tabs(["Kapasitet", "Skisse", "Sensitivitet", "Tabeller"])
    with tc:
        _tab_capacity(inp, R_itd_v, R_Vd_v, "steel")
    with ts:
        _tab_sketch(inp)
    with tsen:
        _tab_sensitivity(inp, R_itd_v, R_Vd_v, "steel")
    with ttab:
        _tab_tables()
    _pdf_download_button(inp, R_itd_v, R_Vd_v, key="rb")
    _param_glossary()


# ---------------------------------------------------------------------------
# Hoved render-funksjon
# ---------------------------------------------------------------------------

def render():
    st.header("Kapasitetsberegning – Passive forankringer i berg")
    st.caption("NGI-rapport 20210114-01-R (Sikringshåndboken, NVE)")

    with st.sidebar:
        fane = st.radio(
            "Forankringstype",
            ["Stangstål", "Selvborende stag (leverandør)", "Innstøpt bergbolt"],
            key="anchor_tab"
        )

    if fane == "Stangstål":
        _render_bar_steel()
    elif fane == "Selvborende stag (leverandør)":
        _render_supplier()
    else:
        _render_rock_bolt()

