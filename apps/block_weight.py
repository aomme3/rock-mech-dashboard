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

    def dist(p, q):
        return float(np.hypot(q[0] - p[0], q[1] - p[1]))

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
    P_front_top = (x_ft, y_ft)

    # Terrenglinje gjennom front-top med helning beta
    m_terr, b_terr = line_from_point_angle(P_front_top, beta)

    # Terreng ∩ plan: gir maksimal utstrekning for en lukket blokk
    ip = intersect_lines(m_terr, b_terr, m_plane, b_plane)
    if ip is None:
        st.error("Terreng og glideplan er parallelle (β≈α). Juster vinkler.")
        return

    x_int, y_int = ip
    if x_int <= x_ft + 1e-9:
        st.error("Terreng møter planet før/ved front-top (x_int ≤ x_ft) → ingen plass til baksprekk. Juster β/γ/Hf.")
        return

    # Maks avstand langs terreng fra front-top til (x_int, y_int)
    terr_scale = np.sqrt(1.0 + m_terr**2)
    d_max = (x_int - x_ft) * terr_scale
    d_max_eff = max(0.1, 0.98 * d_max)

    # θ: vinkel fra horisontal, 0–180. 90=vertikal. >90 = faller "bakover".
    # Unngå nær-parallell med planet ved å kreve |θ-α| > margin når θ<90 (tan positiv),
    # og også unngå θ nær 180 (tan ~0) / nær 90 (tan -> inf) med litt margin.
    theta_lo, theta_hi = 5.0, 175.0
    margin = 0.5

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

        theta = st.slider(
            "Baksprekk-helning θ (°) fra horisontal (90=vertikal, >90 faller bakover)",
            float(theta_lo),
            float(theta_hi),
            90.0,
            0.5,
        )

    # Back-top point on terrain at distance d_back from front-top along terrain
    x_top = x_ft + d_back / max(terr_scale, 1e-12)
    y_top = m_terr * x_top + b_terr
    P_back_top = (x_top, y_top)

    # Crack line through back-top with slope m_crack = tan(theta)
    m_crack = np.tan(np.radians(theta))
    b_crack = y_top - m_crack * x_top

    # Foot = intersection(crack, plane)
    denom = (m_plane - m_crack)
    if abs(denom) < 1e-10:
        st.error("Baksprekk er (nesten) parallell med glideplanet. Endre θ (eller α).")
        return

    x_foot = b_crack / denom
    y_foot = m_plane * x_foot
    P_back_foot = (x_foot, y_foot)

    # --- Gyldighetskontroller ---
    # 1) Fotpunkt må ligge på plan-segmentet som kan være bunn av blokka: (0, x_int)
    eps = 1e-6
    if not (0.0 + eps < x_foot < x_int - eps):
        st.error(
            "Ugyldig: baksprekk treffer glideplanet utenfor blokkområdet (utenfor [0, x_int]). "
            "Juster θ eller d_back (evt. β/α)."
        )
        return

    # 2) Retningslogikk:
    #    - θ < 90: sprekken faller mot front → forvent x_foot < x_top
    #    - θ > 90: sprekken faller bakover → forvent x_foot > x_top
    if theta < 90.0 - margin and not (x_foot < x_top - eps):
        st.error("Ugyldig: θ<90 betyr fall mot front, men fotpunkt havnet ikke foran toppunktet. Prøv annen θ/d_back.")
        return
    if theta > 90.0 + margin and not (x_foot > x_top + eps):
        st.error("Ugyldig: θ>90 betyr fall bakover, men fotpunkt havnet ikke bak toppunktet. Prøv annen θ/d_back.")
        return

    # 3) Terreng ved toppunkt må være over planet ved fotpunkt (ellers blir polygon rar)
    if y_top <= y_foot + 1e-6:
        st.error("Ugyldig: baksprekk-toppen ligger ikke over glideplanet. Juster β/d_back.")
        return

    # Kontaktlengde langs planet fra tå til fotpunkt
    s_contact = x_foot / max(np.cos(a), 1e-9)

    # Blokkpolygon: toe -> front-top -> back-top -> back-foot
    poly = [(0.0, 0.0), P_front_top, P_back_top, P_back_foot]

    A_block = polygon_area(poly)
    gamma_kN_m3 = rho * g / 1000.0
    W = gamma_kN_m3 * (A_block * width)

    A_plane = s_contact * width
    N = W * np.cos(a)
    T = W * np.sin(a)

    sigma_n_kPa = N / max(A_plane, 1e-12)
    sigma_n_MPa = sigma_n_kPa / 1000.0

    # Requested datapoints
    L_release = dist(P_back_top, P_back_foot)      # lengde langs baksprekk
    dx_crest_to_back = x_top - x_ft                # horisontal avstand crest -> baksprekk-top

    # Plot geometry
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, x_int], y=[0, m_plane * x_int], mode="lines", name="Glideplan"))
    fig.add_trace(go.Scatter(x=[x_ft, x_int], y=[y_ft, m_terr * x_int + b_terr], mode="lines", name="Terreng"))
    fig.add_trace(go.Scatter(x=[0, x_ft], y=[0, y_ft], mode="lines", name="Frontflate"))
    fig.add_trace(go.Scatter(x=[x_top, x_foot], y=[y_top, y_foot], mode="lines", name="Baksprekk"))

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

        r1, r2, r3 = st.columns(3)
        r1.metric("Lengde avløsende sprekk (m)", f"{L_release:.2f}")
        r2.metric("Δx topp→baksprekk (m)", f"{dx_crest_to_back:.2f}")
        r3.metric("θ baksprekk (°)", f"{theta:.1f}")

        st.caption("σn er fra egenvekt alene. Vanntrykk/anker/sidestøtte må legges til separat.")

    with c2:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tallgrunnlag / eksport"):
        df = pd.DataFrame(
            [
                ["alpha_deg", alpha],
                ["beta_deg", beta],
                ["gamma_deg", gamma],
                ["theta_backcrack_deg", theta],
                ["Hf_m", Hf],
                ["d_back_m (langs terreng)", d_back],
                ["d_max_m (beregnet)", d_max],
                ["width_m", width],
                ["rho_kg_m3", rho],
                ["A_block_m2", A_block],
                ["A_plane_m2", A_plane],
                ["W_kN", W],
                ["N_kN", N],
                ["T_kN", T],
                ["sigma_n_kPa", sigma_n_kPa],
                ["sigma_n_MPa", sigma_n_MPa],
                ["L_release_m", L_release],
                ["dx_crest_to_back_m", dx_crest_to_back],
                ["x_front_top_m", x_ft],
                ["x_back_top_m", x_top],
                ["x_back_foot_m", x_foot],
                ["x_int_m (terreng∩plan)", x_int],
            ],
            columns=["Parameter", "Value"],
        )
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Last ned CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="block_normalstress_results.csv",
            mime="text/csv",
        )