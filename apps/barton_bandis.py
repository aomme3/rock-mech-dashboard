import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.barton import scale_jrc_jcs

PRESETS = {
    "Custom": {"JRC0": 12.0, "JCS0": 80.0, "phi_b": 30.0, "L": 5.0, "L0": 0.10},
    "Granitt (typisk)": {"JRC0": 12.0, "JCS0": 120.0, "phi_b": 32.0, "L": 5.0, "L0": 0.10},
    "Gneis (typisk)": {"JRC0": 14.0, "JCS0": 100.0, "phi_b": 31.0, "L": 5.0, "L0": 0.10},
    "Glimmerskifer (typisk)": {"JRC0": 10.0, "JCS0": 60.0, "phi_b": 29.0, "L": 5.0, "L0": 0.10},
}

def _interp(x, y, x0):
    return float(np.interp(float(x0), x, y))

def _curves(JRC0, JCS0, phi_b, L, L0, sigma_max, npts, use_scaling=True):
    sigma = np.linspace(1e-3, float(sigma_max), int(npts))
    sn = np.maximum(sigma, 1e-6)

    # Unscaled
    angle_u = phi_b + JRC0 * np.log10(JCS0 / sn)
    tau_u = sn * np.tan(np.radians(angle_u))
    phi_u = np.degrees(np.arctan(tau_u / sn))

    # Scaled
    if use_scaling:
        JRCs, JCSs = scale_jrc_jcs(JRC0, JCS0, L, L0)
    else:
        JRCs, JCSs = float(JRC0), float(JCS0)

    angle_s = phi_b + JRCs * np.log10(JCSs / sn)
    tau_s = sn * np.tan(np.radians(angle_s))
    phi_s = np.degrees(np.arctan(tau_s / sn))

    return sigma, (tau_u, phi_u), (tau_s, phi_s), (JRCs, JCSs)

def render():
    st.header("Barton–Bandis: τ(σn) og φ_active(σn)")

    # -------- Sidebar --------
    with st.sidebar:
        st.subheader("Preset")
        preset = st.selectbox("Velg", list(PRESETS.keys()), index=0, key="bb_preset")

        if st.button("Bruk preset-verdier", key="bb_apply_preset"):
            p = PRESETS[preset]
            st.session_state["bb_JRC0"] = p["JRC0"]
            st.session_state["bb_JCS0"] = p["JCS0"]
            st.session_state["bb_phi_b"] = p["phi_b"]
            st.session_state["bb_L"] = p["L"]
            st.session_state["bb_L0"] = p["L0"]

        st.subheader("Input")
        JRC0 = st.slider("JRC0", 0.0, 20.0, float(st.session_state.get("bb_JRC0", 12.0)), 0.1, key="bb_JRC0")
        JCS0 = st.slider("JCS0 (MPa)", 20.0, 200.0, float(st.session_state.get("bb_JCS0", 80.0)), 1.0, key="bb_JCS0")
        phi_b = st.slider("φb (deg)", 25.0, 40.0, float(st.session_state.get("bb_phi_b", 30.0)), 0.1, key="bb_phi_b")

        st.subheader("Skalering")
        L = st.slider("L (m)", 1.0, 20.0, float(st.session_state.get("bb_L", 5.0)), 0.5, key="bb_L")
        L0 = st.number_input("L0 (m)", 0.01, 1.0, float(st.session_state.get("bb_L0", 0.10)), 0.01, key="bb_L0")
        use_scaling = st.checkbox("Bruk skalering", value=True, key="bb_use_scaling")
        show_both = st.checkbox("Vis både uskalert og skalert", value=True, key="bb_show_both")

        st.subheader("σn")
        sigma_max = st.slider("σn maks (MPa)", 0.1, 5.0, 5.0, 0.1, key="bb_sigma_max")
        npts = st.slider("Punkter", 50, 500, 200, 10, key="bb_npts")

    sigma, (tau_u, phi_u), (tau_s, phi_s), (JRCs, JCSs) = _curves(
        JRC0, JCS0, phi_b, L, L0, sigma_max, npts, use_scaling=use_scaling
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("JRC brukt (skalert)", f"{JRCs:.2f}")
    m2.metric("JCS brukt (skalert) MPa", f"{JCSs:.2f}")
    m3.metric("φb", f"{phi_b:.1f}°")
    m4.metric("L/L0", f"{L:.2f}/{L0:.2f}")

    t1, t2, t3, t4 = st.tabs(["Model", "Sensitivity", "Monte Carlo", "Export"])

    # -------- Tab: Model --------
    with t1:
        c1, c2 = st.columns(2)

        fig1 = go.Figure()
        if show_both:
            fig1.add_trace(go.Scatter(x=sigma, y=tau_u, mode="lines", name="Uskalert"))
        fig1.add_trace(go.Scatter(x=sigma, y=tau_s, mode="lines", name="Skalert"))
        fig1.update_layout(
            title="τ vs σn",
            xaxis_title="σn (MPa)",
            yaxis_title="τ (MPa)",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        c1.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        if show_both:
            fig2.add_trace(go.Scatter(x=sigma, y=phi_u, mode="lines", name="Uskalert"))
        fig2.add_trace(go.Scatter(x=sigma, y=phi_s, mode="lines", name="Skalert"))
        fig2.update_layout(
            title="φ_active vs σn",
            xaxis_title="σn (MPa)",
            yaxis_title="φ_active (deg)",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        c2.plotly_chart(fig2, use_container_width=True)

        st.caption("σn starter ved 0.001 MPa for å unngå log10(0).")

    # -------- Tab: Sensitivity --------
    with t2:
        st.subheader("Sensitivity (tornado + spider)")

        labels = {"JRC0": "JRC0", "JCS0": "JCS0", "phi_b": "φb", "L": "L", "L0": "L0"}

        left, right = st.columns(2)
        with left:
            out_choice = st.selectbox("Output", ["τ ved valgt σn", "φ_active ved valgt σn"], key="sens_out")
            sigma_target = st.slider(
                "σn (MPa)",
                0.05,
                float(sigma_max),
                min(1.0, float(sigma_max)),
                0.05,
                key="sens_sigma",
            )
            curve = st.radio("Kurve", ["Skalert", "Uskalert"], horizontal=True, key="sens_curve")
            step_pct = st.slider("± endring (%)", 1, 50, 10, 1, key="sens_step")

            vary = st.multiselect(
                "Varier parametre",
                options=list(labels.keys()),
                default=["JRC0", "JCS0", "phi_b", "L"],
                format_func=lambda k: labels[k],
                key="sens_vary",
            )

        series = (tau_s if out_choice.startswith("τ") else phi_s) if curve == "Skalert" else (
            tau_u if out_choice.startswith("τ") else phi_u
        )
        y_base = _interp(sigma, series, sigma_target)

        def eval_output(JRC0_, JCS0_, phi_b_, L_, L0_):
            sig, (tu, pu), (ts, ps), _ = _curves(JRC0_, JCS0_, phi_b_, L_, L0_, sigma_max, npts, use_scaling=use_scaling)
            ser = (ts if out_choice.startswith("τ") else ps) if curve == "Skalert" else (tu if out_choice.startswith("τ") else pu)
            return _interp(sig, ser, sigma_target)

        effects = []
        spider = {"Parameter": [], "-%": [], "+%": []}
        base = dict(JRC0=JRC0, JCS0=JCS0, phi_b=phi_b, L=L, L0=L0)

        for p in vary:
            lo = base.copy()
            hi = base.copy()
            lo[p] = base[p] * (1 - step_pct / 100)
            hi[p] = base[p] * (1 + step_pct / 100)

            y_lo = eval_output(lo["JRC0"], lo["JCS0"], lo["phi_b"], lo["L"], lo["L0"])
            y_hi = eval_output(hi["JRC0"], hi["JCS0"], hi["phi_b"], hi["L"], hi["L0"])

            effects.append((labels[p], y_lo, y_hi))
            spider["Parameter"].append(labels[p])
            spider["-%"].append(100 * (y_lo - y_base) / y_base if abs(y_base) > 1e-12 else np.nan)
            spider["+%"].append(100 * (y_hi - y_base) / y_base if abs(y_base) > 1e-12 else np.nan)

        with right:
            st.metric("Baseline", f"{y_base:.4g}")
            st.caption(f"σn={sigma_target:.2f} MPa • {curve}")

        df_t = pd.DataFrame(effects, columns=["Parameter", "Lav", "Høy"])
        df_t["Spenn"] = (df_t["Høy"] - df_t["Lav"]).abs()
        df_t = df_t.sort_values("Spenn", ascending=True)

        fig_t = go.Figure()
        for _, r in df_t.iterrows():
            fig_t.add_trace(go.Bar(y=[r["Parameter"]], x=[r["Høy"] - y_base], orientation="h", showlegend=False))
            fig_t.add_trace(go.Bar(y=[r["Parameter"]], x=[r["Lav"] - y_base], orientation="h", showlegend=False))
        fig_t.update_layout(
            barmode="overlay",
            xaxis_title="Δ output (fra baseline)",
            height=440,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_t, use_container_width=True)

        sdf = pd.DataFrame(spider)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sdf["Parameter"], y=sdf["-%"], mode="lines+markers", name=f"-{step_pct}%"))
        fig_s.add_trace(go.Scatter(x=sdf["Parameter"], y=sdf["+%"], mode="lines+markers", name=f"+{step_pct}%"))
        fig_s.update_layout(
            yaxis_title="% endring",
            xaxis_title="Parameter",
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # -------- Tab: Monte Carlo --------
    with t3:
        st.subheader("Monte Carlo")

        c1, c2, c3 = st.columns(3)
        with c1:
            mc_n = st.slider("N", 200, 20000, 2000, 200, key="mc_n")
        with c2:
            mc_sigma = st.slider(
                "σn (MPa)",
                0.05,
                float(sigma_max),
                min(1.0, float(sigma_max)),
                0.05,
                key="mc_sigma",
            )
        with c3:
            mc_out = st.selectbox("Output", ["τ", "φ_active"], key="mc_out")

        mc_curve = st.radio("Kurve", ["Skalert", "Uskalert"], horizontal=True, key="mc_curve")

        u1, u2, u3, u4, u5 = st.columns(5)
        with u1:
            sd_JRC = st.number_input("Std JRC0", 0.0, 10.0, 1.0, 0.1, key="mc_sd_jrc")
        with u2:
            sd_JCS = st.number_input("Std JCS0 (MPa)", 0.0, 100.0, 10.0, 1.0, key="mc_sd_jcs")
        with u3:
            sd_phi = st.number_input("Std φb", 0.0, 10.0, 1.0, 0.1, key="mc_sd_phi")
        with u4:
            sd_L = st.number_input("Std L", 0.0, 10.0, 1.0, 0.1, key="mc_sd_L")
        with u5:
            sd_L0 = st.number_input("Std L0", 0.0, 0.5, 0.02, 0.01, key="mc_sd_L0")

        rng = np.random.default_rng(12345)
        JRC_samp = np.clip(rng.normal(JRC0, sd_JRC, mc_n), 0.0, 20.0)
        JCS_samp = np.clip(rng.normal(JCS0, sd_JCS, mc_n), 20.0, 200.0)
        phi_samp = np.clip(rng.normal(phi_b, sd_phi, mc_n), 25.0, 40.0)
        L_samp = np.clip(rng.normal(L, sd_L, mc_n), 1.0, 20.0)
        L0_samp = np.clip(rng.normal(L0, sd_L0, mc_n), 0.01, 1.0)

        out = np.empty(mc_n, float)
        sn = float(mc_sigma)

        for i in range(mc_n):
            if mc_curve == "Skalert" and use_scaling:
                jrc, jcs = scale_jrc_jcs(float(JRC_samp[i]), float(JCS_samp[i]), float(L_samp[i]), float(L0_samp[i]))
            else:
                jrc, jcs = float(JRC_samp[i]), float(JCS_samp[i])

            angle = float(phi_samp[i]) + jrc * np.log10(jcs / max(sn, 1e-6))
            tau = sn * np.tan(np.radians(angle))

            if mc_out == "τ":
                out[i] = tau
            else:
                out[i] = np.degrees(np.arctan(tau / max(sn, 1e-6)))

        q05, q50, q95 = np.percentile(out, [5, 50, 95])
        mean, sd = float(out.mean()), float(out.std(ddof=1))

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean", f"{mean:.4g}")
        s2.metric("Std", f"{sd:.4g}")
        s3.metric("P50", f"{q50:.4g}")
        s4.metric("P5/P95", f"{q05:.4g}/{q95:.4g}")

        df_mc = pd.DataFrame(
            {"output": out, "JRC0": JRC_samp, "JCS0": JCS_samp, "phi_b": phi_samp, "L": L_samp, "L0": L0_samp}
        )

        fig_h = px.histogram(df_mc, x="output", nbins=40, title=f"{mc_out} ved σn={sn:.2f} MPa ({mc_curve})")
        fig_h.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        corr = (
            df_mc.corr(numeric_only=True)["output"]
            .drop("output")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        fig_c = go.Figure(go.Bar(x=corr.index, y=corr.values))
        fig_c.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Korrelasjon")
        st.plotly_chart(fig_c, use_container_width=True)

        st.download_button(
            "Last ned Monte Carlo CSV",
            df_mc.to_csv(index=False).encode("utf-8"),
            file_name="barton_bandis_monte_carlo.csv",
            mime="text/csv",
            key="mc_dl",
        )

    # -------- Tab: Export --------
    with t4:
        df = pd.DataFrame(
            {
                "sigma_n_MPa": sigma,
                "tau_unscaled_MPa": tau_u,
                "phi_unscaled_deg": phi_u,
                "tau_scaled_MPa": tau_s,
                "phi_scaled_deg": phi_s,
            }
        )
        st.dataframe(df, use_container_width=True, height=320)
        st.download_button(
            "Last ned kurver (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="barton_bandis_curves.csv",
            mime="text/csv",
            key="bb_dl_curves",
        )
