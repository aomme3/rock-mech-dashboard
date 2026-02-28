import streamlit as st

from apps import barton_bandis, water_pressure_v2

st.set_page_config(page_title="Rock Mechanics Dashboard", layout="wide")

st.sidebar.title("Rock Mechanics")
page = st.sidebar.radio(
    "Velg app",
    ["Barton–Bandis", "Vanntrykk + FS"],
    index=0
)

if page == "Barton–Bandis":
    barton_bandis.render()
else:
    water_pressure_v2.render()