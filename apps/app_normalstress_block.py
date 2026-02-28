import streamlit as st
import numpy as np
import plotly.graph_objects as go


def render():

    st.title("Normalspenning fra blokkvekt")

    g = 9.81

    ROCK_DENSITIES = {
        "Granitt": 2700,
        "Gneis": 2750,
        "Basalt": 2950,
        "Kvartsitt": 2650,
        "Kalkstein": 2600,
        "Sandstein": 2400,
        "Skifer": 2500,
    }

    st.sidebar.header("Geometri")

    alpha = st.sidebar.slider("Glideplan α (°)", 5.0, 85.0, 35.0)
    beta = st.sidebar.slider("Terreng β (°)", -20.0, 60.0, 10.0)
    gamma = st.sidebar.slider("Front γ (°)", 10.0, 90.0, 75.0)

    Hf = st.sidebar.slider("Fronthøyde (m)", 1.0, 40.0, 8.0)

    rock = st.sidebar.selectbox(
        "Bergart",
        list(ROCK_DENSITIES.keys())
    )

    rho = ROCK_DENSITIES[rock]

    # --- Enkel blokkmodell ---
    alpha_r = np.radians(alpha)

    block_length = Hf / np.tan(alpha_r)

    A_block = 0.5 * block_length * Hf

    gamma_kN = rho * g / 1000
    W = gamma_kN * A_block

    N = W * np.cos(alpha_r)
    sigma_n = N / block_length

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Blokkareal", f"{A_block:.2f} m²")
        st.metric("Blokkvekt", f"{W:.1f} kN")
        st.metric("Normalspenning", f"{sigma_n:.1f} kPa")

    with col2:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[0, block_length, 0],
            y=[0, 0, Hf],
            fill="toself"
        ))

        fig.update_layout(
            title="Blokkgeometri",
            yaxis_scaleanchor="x"
        )

        st.plotly_chart(fig, use_container_width=True)