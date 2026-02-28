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

    # Gyldig theta-range: må være brattere enn glideplanet for å krysse det "ned mot front"
    theta_min = min(max(alpha + 0.5, 5.0), 89.0)
    theta_max = 89.5
    if theta_min >= theta_max:
        st.error("α er så bratt at gyldig baksprekk-helning ikke finnes (θ må være > α). Reduser α.")
        return

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
            "Baksprekk-helning θ (°) fra horisontal (dipper mot front)",
            float(theta_min),
            float(theta_max),
            float(max(75.0, theta_min)),
            0.5,
        )
        st.caption("θ=90° gir vertikal baksprekk. Gyldig krav: θ > α.")

    # Back-top point on terrain at distance d_back from front-top ALONG the terrain line
    # d_back = (x_top - x_ft) * sqrt(1+m_terr^2)  =>  x_top = x_ft + d_back/terr_scale
    x_top = x_ft + d_back / max(terr_scale, 1e-12)
    y_top = m_terr * x_top + b_terr
    P_back_top = (x_top, y_top)

    # Back crack line: through P_back_top with slope m_crack = tan(theta) (positive),
    # so that moving left (toward smaller x) goes downward (toward smaller y).
    m_crack = np.tan(np.radians(theta))
    b_crack = y_top - m_crack * x_top

    # Foot point = intersection(back crack, plane)
    # plane: y = m_plane x; crack: y = m_crack x + b_crack
    # => (m_plane - m_crack)x = b_crack
    denom = (m_plane - m_crack)
    if abs(denom) < 1e-10:
        st.error("Baksprekk er (nesten) parallell med glideplanet. Øk θ (eller endre α).")
        return

    x_foot = b_crack / denom
    y_foot = m_plane * x_foot
    P_back_foot = (x_foot, y_foot)

    # Validity checks for geometry
    # Expect foot to be between toe (0) and back-top (x_top) if it dips toward front
    if not (0.0 < x_foot < x_top - 1e-6):
        st.error(
            "Ugyldig kombinasjon: baksprekklinjen treffer ikke glideplanet mellom tå og baksprekk-toppen.\n"
            "Prøv større θ (brattere), mindre d_back, eller endre α/β."
        )
        return

    # Also require that terrain is above the plane at relevant x (so polygon closes "over" the plane)
    # (We mainly need y_top > y_foot, already implied by x_foot < x_top and slopes)
    if y_top <= y_foot + 1e-6:
        st.error("Ugyldig: terreng ved baksprekk er ikke over glideplanet. Endre β eller d_back.")
        return

    # Contact length along plane from toe to foot:
    s_contact = x_foot / max(np.cos(a), 1e-9)

    # Block polygon: toe -> front-top -> back-top -> back-foot (on plane)
    poly = [(0.0, 0.0), P_front_top, P_back_top, P_back_foot]

    A_block = polygon_area(poly)                  # m²
    gamma_kN_m3 = rho * g / 1000.0               # kN/m³
    W = gamma_kN_m3 * (A_block * width)          # kN (per m "inn i arket" hvis width=1)

    A_plane = s_contact * width                  # m²
    N = W * np.cos(a)                            # kN
    T = W * np.sin(a)                            # kN

    sigma_n_kPa = N / max(A_plane, 1e-12)        # kPa
    sigma_n_MPa = sigma_n_kPa / 1000.0

    # New requested datapoints:
    # 1) Length of releasing crack (baksprekk) along its line
    L_release = dist(P_back_top, P_back_foot)

    # 2) Horizontal distance between crest (front-top) and back crack top
    dx_crest_to_back = x_top - x_ft

    # Plot geometry
    fig = go.Figure()

    # plane segment (0 -> x_int)
    fig.add_trace(go.Scatter(x=[0, x_int], y=[0, m_plane * x_int], mode="lines", name="Glideplan"))
    # terrain segment (front-top -> intersection)
    fig.add_trace(go.Scatter(x=[x_ft, x_int], y=[y_ft, m_terr * x_int + b_terr], mode="lines", name="Terreng"))
    # also show terrain back to x=0 for context
    fig.add_trace(go.Scatter(x=[0, x_ft], y=[m_terr * 0 + b_terr, y_ft], mode="lines", name="Terreng (forlengelse)", line=dict(dash="dot")))

    # front face
    fig.add_trace(go.Scatter(x=[0, x_ft], y=[0, y_ft], mode="lines", name="Frontflate"))

    # back crack (slanted)
    fig.add_trace(go.Scatter(x=[x_top, x_foot], y=[y_top, y_foot], mode="lines", name="Baksprekk"))

    # polygon
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
        df = pd.DataFrame([
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
        ], columns=["Parameter", "Value"])
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Last ned CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="block_normalstress_results.csv",
            mime="text/csv",
        )