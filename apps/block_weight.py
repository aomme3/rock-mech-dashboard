import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render():
    st.title("Blokkvekt → normalspenning på glideplan (2D snitt)")

    g = 9.81  # m/s²

    ROCK_DENSITIES = {
        "Granitt": 2700,
        "Gneis": 2750,
        "Basalt": 2950,
        "Dioritt / gabbro": 2900,
        "Granodioritt": 2700,
        "Kvartsitt": 2650,
        "Marmor": 2700,
        "Kalkstein": 2600,
        "Sandstein": 2400,
        "Skifer (glimmerskifer/shale)": 2500,
        "Skifer (slate)": 2700,
        "Fyllitt": 2700,
        "Amfibolitt": 3000,
        "Dacitt/andesitt": 2700,
    }

    def line_from_point_angle(p, deg):
        th = np.radians(deg)
        m = np.tan(th)
        b = p[1] - m * p[0]
        return m, b

    def intersect_lines(m1, b1, m2, b2):
        if abs(m1 - m2) < 1e-12:
            return None
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return float(x), float(y)

    def polygon_area(points):
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        x2 = np.r_[x, x[0]]
        y2 = np.r_[y, y[0]]
        return 0.5 * abs(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1]))

    with st.sidebar:
        st.header("Geometri")
        alpha = st.slider("Glideplan helning α (°)", 5.0, 85.0, 35.0, 0.5)
        beta = st.slider("Terreng helning β (°)", -20.0, 85.0, 10.0, 0.5)
        gamma = st.slider("Frontflate helning γ (°)", 5.0, 90.0, 75.0, 0.5)

        st.divider()
        st.header("Størrelse")
        Hf = st.slider("Front-høyde Hf (m)", 0.5, 60.0, 8.0, 0.5)
        width = st.number_input("Bredde (m) (snitt=1 m)", 0.1, 50.0, 1.0, 0.1)

        st.divider()
        st.header("Tetthet")
        rock_choice = st.selectbox("Bergart (ρ)", ["(egen verdi)"] + list(ROCK_DENSITIES.keys()), index=1)
        if rock_choice == "(egen verdi)":
            rho = st.number_input("ρ (kg/m³)", 1500, 3500, 2700, 50)
        else:
            rho = ROCK_DENSITIES[rock_choice]
            st.caption(f"ρ = {rho} kg/m³")

    a = np.radians(alpha)
    m_plane, b_plane = np.tan(a), 0.0  # y = m_plane x

    # Front-top punkt: frontflate gjennom (0,0) og y=Hf
    if abs(np.cos(np.radians(gamma))) < 1e-6:  # ~vertikal
        x_ft = 0.0
    else:
        m_front = np.tan(np.radians(gamma))
        x_ft = Hf / max(m_front, 1e-12)
    y_ft = Hf

    # Terrenglinje gjennom front-top med helning beta
    m_terr, b_terr = line_from_point_angle((x_ft, y_ft), beta)

    # Terreng ∩ plan: gir maksimal utstrekning før blokka "lukker seg"
    ip = intersect_lines(m_terr, b_terr, m_plane, b_plane)
    if ip is None:
        st.error("Terreng og glideplan er parallelle (β≈α). Juster vinkler.")
        return

    x_int, y_int = ip
    if x_int <= x_ft + 1e-9:
        st.error("Terreng møter planet før/ved front-top (x_int ≤ x_ft) → ingen plass til baksprekk bak front-top. Juster β/γ/Hf.")
        return

    # Maks avstand langs terreng fra front-top til (x_int, y_int)
    terr_scale = np.sqrt(1.0 + m_terr**2)  # = 1/cos(beta) i praksis
    d_max = (x_int - x_ft) * terr_scale
    d_max_eff = max(0.1, 0.98 * d_max)

    with st.sidebar:
        st.divider()
        st.header("Baksprekk")
        d_back = st.slider(
            "Avstand til baksprekk fra toppen av frontflaten (m) langs terreng",
            0.0,
            float(d_max_eff),
            float(min(4.0, d_max_eff)),
            0.1,
        )
        st.caption(f"Maks (beregnet) ≈ {d_max:.2f} m (slider-tak = 98%)")

    # Konverter d_back -> x_back (baksprekk er vertikal ved x=x_back)
    x_back = x_ft + d_back / max(terr_scale, 1e-12)

    # Fotpunkt på planet
    y_back = m_plane * x_back

    # Toppunkt på terreng
    y_top_back = m_terr * x_back + b_terr
    if y_top_back <= y_back + 1e-6:
        st.error("Baksprekk treffer ikke terreng over planet (degenerert blokk). Reduser d_back eller juster β/γ/Hf.")
        return

    # Lengde av kontakt langs planet fra tå (0,0) til x_back
    s_back = x_back / max(np.cos(a), 1e-9)

    # Blokkpolygon: toe -> front-top -> back-top -> back-foot
    poly = [(0.0, 0.0), (x_ft, y_ft), (x_back, y_top_back), (x_back, y_back)]

    A_block = polygon_area(poly)
    gamma_kN_m3 = rho * g / 1000.0
    W = gamma_kN_m3 * (A_block * width)

    A_plane = s_back * width
    N = W * np.cos(a)
    T = W * np.sin(a)

    sigma_n_kPa = N / max(A_plane, 1e-12)
    sigma_n_MPa = sigma_n_kPa / 1000.0

    # Plot geometri
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, x_int], y=[0, m_plane * x_int], mode="lines", name="Glideplan"))
    fig.add_trace(go.Scatter(x=[0, x_int], y=[m_terr * 0 + b_terr, m_terr * x_int + b_terr], mode="lines", name="Terreng"))
    fig.add_trace(go.Scatter(x=[0, x_ft], y=[0, y_ft], mode="lines", name="Frontflate"))
    fig.add_trace(go.Scatter(x=[x_back, x_back], y=[y_back, y_top_back], mode="lines", name="Baksprekk"))

    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Blokk", fill="toself"))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        title="Geometri (2D snitt)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Resultater")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("A_blokk (m²)", f"{A_block:.2f}")
        m2.metric("W (kN)", f"{W:.1f}")
        m3.metric("N = W·cosα (kN)", f"{N:.1f}")
        m4.metric("T = W·sinα (kN)", f"{T:.1f}")

        n1, n2, n3 = st.columns(3)
        n1.metric("A_plan (m²)", f"{A_plane:.2f}")
        n2.metric("σn (kPa)", f"{sigma_n_kPa:.1f}")
        n3.metric("σn (MPa)", f"{sigma_n_MPa:.4f}")

        st.caption("σn er fra egenvekt alene. Vanntrykk/anker/sidestøtte må legges til separat.")

    with c2:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tallgrunnlag / eksport"):
        df = pd.DataFrame([
            ["alpha_deg", alpha],
            ["beta_deg", beta],
            ["gamma_deg", gamma],
            ["Hf_m", Hf],
            ["d_back_m (langs terreng)", d_back],
            ["d_max_m (beregnet)", d_max],
            ["x_back_m", x_back],
            ["s_back_m (langs plan fra tå)", s_back],
            ["width_m", width],
            ["rho_kg_m3", rho],
            ["A_block_m2", A_block],
            ["A_plane_m2", A_plane],
            ["W_kN", W],
            ["N_kN", N],
            ["T_kN", T],
            ["sigma_n_kPa", sigma_n_kPa],
            ["sigma_n_MPa", sigma_n_MPa],
            ["x_int_m (terreng∩plan)", x_int],
        ], columns=["Parameter", "Value"])
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Last ned CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="block_normalstress_results.csv",
            mime="text/csv",
        )