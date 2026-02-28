import streamlit as st

from apps import barton_bandis, water_pressure_v2, block_weight

st.set_page_config(
    page_title="Rock Mechanics Dashboard",
    layout="wide"
)

st.sidebar.title("Rock Mechanics")

page = st.sidebar.radio(
    "Velg app",
    [
        "Barton–Bandis",
        "Vanntrykk + FS",
        "Blokkvekt / Normalspenning"
    ],
    index=0
)

if page == "Barton–Bandis":
    barton_bandis.render()

elif page == "Vanntrykk + FS":
    water_pressure_v2.render()

else:
    block_weight.render()