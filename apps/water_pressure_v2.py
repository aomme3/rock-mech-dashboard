import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.mech import resolve_to_plane_components, moment_about_point
from utils.barton import scale_jrc_jcs, tau_barton_bandis_mpa, phi_active_deg_from_tau_sigma

GAMMA_W = 9.81  # kN/m^3 (kPa per m vannsøyle)

def plot_pressure(x, p, title, xlabel):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p, mode="lines", name="p"))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Trykk (kPa)",
                      height=340, margin=dict(l=10,r=10,t=45,b=10))
    return fig

def render():
    st.header("Vanntrykk: utglidningsplan + baksprekk (koblet) • moment • FS")

    with st.sidebar:
        st.subheader("Geometri")
        alpha = st.slider("Fall α (deg)", 5.0, 85.0, 35.0, 0.5)
        L = st.slider("Planlengde L (m)", 0.5, 120.0, 8.0, 0.5)
        width = st.number_input("Bredde (m)", 0.1, 50.0, 1.0, 0.1)
        a = np.radians(alpha)

        st.subheader("Baksprekk")
        s_back = st.slider("Posisjon s_back (m) fra tå", 0.0, float(L), float(L), 0.5)
        H_back = st.slider("H (m)", 0.5, 80.0, 6.0, 0.5)
        fill_back = st.slider("Fyllingsgrad (0–1)", 0.0, 1.0, 0.7, 0.05)
        h_water = H_back * fill_back

        x_back = s_back * np.cos(a)
        y_back = s_back * np.sin(a)

        st.subheader("Hydraulikk")
        coupled = st.checkbox("Koble plantrykk til baksprekk (hydrostatisk)", value=True)

        st.subheader("Plantrykk hvis IKKE koblet")
        plane_mode = st.selectbox("Plan-konfig", ["Triangulært (p_toe→0)", "Uniformt", "Lineært"], index=0, disabled=coupled)
        fill_plane = st.slider("Fyllingsgrad langs plan (0–1)", 0.0, 1.0, 0.6, 0.05, disabled=coupled)

        if not coupled:
            L_wet = L * fill_plane
            if plane_mode.startswith("Triang"):
                head_toe = st.slider("Hode ved tå (m)", 0.0, 200.0, 20.0, 1.0)
                p_toe, p_top = GAMMA_W*head_toe, 0.0
            elif plane_mode == "Uniformt":
                p = st.slider("p (kPa)", 0.0, 4000.0, 200.0, 10.0)
                p_toe, p_top = p, p
            else:
                head_toe = st.slider("Hode ved tå (m)", 0.0, 200.0, 20.0, 1.0)
                head_top = st.slider("Hode ved topp (m)", 0.0, 200.0, 0.0, 1.0)
                p_toe, p_top = GAMMA_W*head_toe, GAMMA_W*head_top
        else:
            L_wet, p_toe, p_top = 0.0, 0.0, 0.0

        st.subheader("FS")
        do_fs = st.checkbox("Beregn FS", value=True)

        W_mode = st.selectbox("Egenvekt", ["Gi W (kN/m)", "W=γ·A"], index=0)
        if W_mode == "Gi W (kN/m)":
            W = st.number_input("W (kN/m)", 0.0, 1e7, 500.0, 10.0)
        else:
            gamma_rock = st.number_input("γ (kN/m³)", 10.0, 35.0, 26.0, 0.5)
            A_block = st.number_input("A blokk (m²) (dybde=1m)", 0.01, 1e5, 20.0, 0.5)
            W = gamma_rock * A_block

        N_extra = st.number_input("Ekstra N inn i planet (kN/m)", -1e6, 1e6, 0.0, 10.0)

        model = st.selectbox("Motstand", ["Konstant friksjon (φ')", "Barton–Bandis (peak)"], index=1)
        if model.startswith("Konstant"):
            phi_prime = st.slider("φ' (deg)", 10.0, 60.0, 35.0, 0.5)
            c_kpa = st.number_input("c' (kPa)", 0.0, 5000.0, 0.0, 10.0)
            u_method = st.selectbox("u-metode", ["Middel ū", "Maks u_max"], index=0)
        else:
            JRC0 = st.slider("JRC", 0.0, 20.0, 12.0, 0.1)
            JCS0 = st.slider("JCS (MPa)", 20.0, 200.0, 80.0, 1.0)
            phi_b = st.slider("φb (deg)", 25.0, 40.0, 30.0, 0.1)
            use_scaling = st.checkbox("Skaler JRC/JCS med L", value=True)
            L0_ref = st.number_input("L0 (m)", 0.01, 1.0, 0.10, 0.01)
            u_method = st.selectbox("u-metode", ["Middel ū", "Maks u_max", "Integrert punktvis"], index=2)

    # --- Hydraulikk: baksprekk hydrostatisk
    R_back = 0.0; Fx_back=0.0; Fy_back=0.0; M_back=0.0; zbar_back=0.0
    if h_water > 0:
        R_back = 0.5 * GAMMA_W * (h_water**2) * width
        zbar_back = h_water/3.0
        Fx_back, Fy_back = R_back, 0.0
        M_back = moment_about_point(Fx_back, Fy_back, x_back, y_back+zbar_back, 0.0, 0.0)

    # --- Hydraulikk: plan poretrykk u(s)
    s = np.linspace(0.0, L, 600)
    y_s = s*np.sin(a)

    if coupled:
        y_pz = y_back + h_water
        u_plane = GAMMA_W * np.maximum(y_pz - y_s, 0.0)
    else:
        u_plane = np.zeros_like(s)
        if L_wet > 0:
            mask = s <= (L_wet + 1e-9)
            u_plane[mask] = p_toe + (p_top - p_toe)*(s[mask]/max(L_wet,1e-9))

    R_plane = float(np.trapz(u_plane, s) * width)
    sbar_plane = float(np.trapz(s*u_plane, s)/np.trapz(u_plane, s)) if R_plane>1e-12 else L/2.0
    n_out = np.array([-np.sin(a), -np.cos(a)])
    Fx_plane, Fy_plane = R_plane*n_out[0], R_plane*n_out[1]
    M_plane = moment_about_point(Fx_plane, Fy_plane, sbar_plane*np.cos(a), sbar_plane*np.sin(a), 0.0, 0.0)

    # Summér vannkrefter
    Fx_water = Fx_plane + Fx_back
    Fy_water = Fy_plane + Fy_back
    M_water = M_plane + M_back

    T_plane, N_plane = resolve_to_plane_components(Fx_plane, Fy_plane, alpha)
    T_back,  N_back  = resolve_to_plane_components(Fx_back,  Fy_back,  alpha)
    T_water, N_water = resolve_to_plane_components(Fx_water, Fy_water, alpha)

    # --- FS
    FS = None
    details = {}

    if do_fs:
        Fx_W, Fy_W = 0.0, -W
        T_W, N_W = resolve_to_plane_components(Fx_W, Fy_W, alpha)

        T_drive = T_W + T_water
        N_total = N_W + N_extra + N_water

        A_plane = L*width
        sigma_n_total_kpa = N_total / max(A_plane, 1e-9)
        u_avg = float(np.trapz(u_plane, s)/max(L,1e-9)) if L>0 else 0.0
        u_max = float(np.max(u_plane)) if len(u_plane) else 0.0

        if u_method == "Middel ū":
            u_use = u_avg
        elif u_method == "Maks u_max":
            u_use = u_max
        else:
            u_use = u_avg  # integrert håndteres separat

        if model.startswith("Konstant"):
            sigma_eff = sigma_n_total_kpa - u_use
            tau_res_kpa = max(sigma_eff,0.0)*np.tan(np.radians(phi_prime)) + c_kpa
            R_resist = tau_res_kpa*A_plane
        else:
            if use_scaling:
                JRC, JCS = scale_jrc_jcs(JRC0, JCS0, L, L0_ref)
            else:
                JRC, JCS = float(JRC0), float(JCS0)

            if u_method != "Integrert punktvis":
                sigma_eff_mpa = max((sigma_n_total_kpa - u_use)/1000.0, 1e-6)
                tau_mpa = tau_barton_bandis_mpa(sigma_eff_mpa, JRC, JCS, phi_b)
                tau_res_kpa = tau_mpa*1000.0
                R_resist = tau_res_kpa*A_plane
                phi_act = phi_active_deg_from_tau_sigma(tau_mpa, sigma_eff_mpa)
            else:
                sigma_eff_kpa_s = sigma_n_total_kpa - u_plane
                sigma_eff_mpa_s = np.maximum(sigma_eff_kpa_s/1000.0, 1e-6)
                tau_mpa_s = np.array([tau_barton_bandis_mpa(x, JRC, JCS, phi_b) for x in sigma_eff_mpa_s])
                tau_kpa_s = tau_mpa_s*1000.0
                R_resist = float(np.trapz(tau_kpa_s, s) * width)
                tau_res_kpa = R_resist / max(A_plane, 1e-9)
                sigma_eff_eq_kpa = max(sigma_n_total_kpa - u_avg, 0.0)
                phi_act = float(np.degrees(np.arctan(tau_res_kpa / max(sigma_eff_eq_kpa,1e-6))))

            details["JRC_used"] = JRC
            details["JCS_used"] = JCS
            details["phi_active_deg"] = phi_act
            details["integrated"] = (u_method == "Integrert punktvis")

        R_drive = max(T_drive, 0.0)
        FS = (R_resist / R_drive) if R_drive>1e-9 else np.inf

        details.update({
            "FS": FS,
            "T_drive_kN": T_drive,
            "N_total_kN": N_total,
            "sigma_n_total_kpa": sigma_n_total_kpa,
            "u_avg_kpa": u_avg,
            "u_max_kpa": u_max,
            "u_used_kpa": u_use,
            "tau_res_kpa": tau_res_kpa,
            "A_plane_m2": A_plane
        })

    # --- UI
    t1, t2, t3 = st.tabs(["Trykk", "Resultanter", "FS"])

    with t1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_pressure(s, u_plane, "u(s) langs plan", "s fra tå (m)"), use_container_width=True)

        if h_water > 0:
            z = np.linspace(0.0, h_water, 200)
            pz = GAMMA_W*(h_water - z)
            c2.plotly_chart(plot_pressure(z, pz, "Trykk i baksprekk (hydrostatisk)", "z opp fra base (m)"), use_container_width=True)
        else:
            c2.info("Ingen vann i baksprekk.")

        st.caption(f"ū={float(np.trapz(u_plane,s)/max(L,1e-9)):.1f} kPa • u_max={float(np.max(u_plane)):.1f} kPa")

    with t2:
        df = pd.DataFrame([
            {"Bidrag":"Vann på plan", "R (kN/m)":R_plane, "Fx":Fx_plane, "Fy":Fy_plane, "T":T_plane, "N":N_plane, "M_toe (kN·m/m)":M_plane},
            {"Bidrag":"Vann i baksprekk", "R (kN/m)":R_back, "Fx":Fx_back, "Fy":Fy_back, "T":T_back, "N":N_back, "M_toe (kN·m/m)":M_back},
            {"Bidrag":"SUM vann", "R (kN/m)":R_plane+R_back, "Fx":Fx_water, "Fy":Fy_water, "T":T_water, "N":N_water, "M_toe (kN·m/m)":M_water},
        ])
        st.dataframe(df, use_container_width=True)

        st.caption("T: + drivende ned langs plan. N: + øker sammenpressing, − reduserer.")

    with t3:
        if not do_fs:
            st.info("FS er av.")
        else:
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("FS", "∞" if np.isinf(details["FS"]) else f"{details['FS']:.2f}")
            m2.metric("σn total (kPa)", f"{details['sigma_n_total_kpa']:.1f}")
            m3.metric("ū/u_max (kPa)", f"{details['u_avg_kpa']:.1f}/{details['u_max_kpa']:.1f}")
            m4.metric("T drivende (kN/m)", f"{details['T_drive_kN']:.1f}")

            rows = [
                ["N total (kN/m)", details["N_total_kN"]],
                ["A plan (m²/m)", details["A_plane_m2"]],
                ["u brukt (kPa)", details["u_used_kpa"]],
                ["τ_res (kPa)", details["tau_res_kpa"]],
            ]
            if model == "Barton–Bandis (peak)":
                rows += [
                    ["JRC brukt", details["JRC_used"]],
                    ["JCS brukt (MPa)", details["JCS_used"]],
                    ["φ_active (deg)", details["phi_active_deg"]],
                    ["Integrert punktvis?", "Ja" if details.get("integrated", False) else "Nei"],
                ]
            st.dataframe(pd.DataFrame(rows, columns=["Størrelse","Verdi"]), use_container_width=True)