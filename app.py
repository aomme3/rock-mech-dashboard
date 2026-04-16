import streamlit as st

from apps import barton_bandis, water_pressure_v2, block_weight, anchor_capacity

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
        "Blokkvekt / Normalspenning",
        "Passive forankringer i berg",
    ],
    index=0
)

if page == "Barton–Bandis":
    barton_bandis.render()

elif page == "Vanntrykk + FS":
    water_pressure_v2.render()

elif page == "Blokkvekt / Normalspenning":
    block_weight.render()

else:
    anchor_capacity.render()