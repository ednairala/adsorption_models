"""
app.py  —  Unified Streamlit interface for the Adsorption Library
===================================================================
Homepage → choose Equilibrium Isotherms or Kinetics module.
Navigation is handled via st.session_state['page'].

HOW TO RUN:
    pip install streamlit plotly pandas scipy numpy
    streamlit run app.py

Folder structure required:
    app.py
    adsorption_isotherms_v6.py
    kinetics/
        __init__.py
        batch.py
        fixed_bed.py
        stats.py
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adsorption_isotherms_v6 import AdsorptionIsotherms, validate_and_clean
from kinetics import BatchKinetics, FixedBedKinetics, KineticsStats


# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adsorption Library",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="metric-container"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stDownloadButton > button { width: 100%; border-radius: 6px; }

    /* Homepage hero cards */
    .module-card {
        border: 1.5px solid rgba(128,128,128,0.18);
        border-radius: 16px;
        padding: 2rem 1.8rem;
        background: linear-gradient(135deg,
            rgba(255,255,255,0.06) 0%,
            rgba(255,255,255,0.02) 100%);
        transition: border-color 0.2s, box-shadow 0.2s;
        min-height: 260px;
    }
    .module-card:hover {
        border-color: rgba(128,128,128,0.45);
        box-shadow: 0 6px 24px rgba(0,0,0,0.09);
    }
    .module-card h2 { margin-top: 0.3rem; margin-bottom: 0.5rem; font-size: 1.4rem; }
    .module-card p  { opacity: 0.72; font-size: 0.93rem; line-height: 1.6; }
    .module-card ul { opacity: 0.72; font-size: 0.88rem; padding-left: 1.2rem; }
    .tag {
        display: inline-block;
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin-bottom: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .tag-eq  { background: rgba(29,158,117,0.15); color: #0F6E56; }
    .tag-kin { background: rgba(83,74,183,0.15);  color: #534AB7; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  SESSION STATE — page router
# ─────────────────────────────────────────────────────────────────────

if 'page' not in st.session_state:
    st.session_state['page'] = 'home'


# ─────────────────────────────────────────────────────────────────────
#  SHARED HELPERS  (used by both modules)
# ─────────────────────────────────────────────────────────────────────

@st.cache_data
def cached_fit_isotherm(_model, model_name, Ce_tuple, qe_tuple):
    Ce = np.array(Ce_tuple)
    qe = np.array(qe_tuple)
    return _model.fit(model_name, Ce, qe)


def build_plotly_figure(model_name, Ce_exp, qe_exp, result,
                         Ce_unit, qe_unit, show_ci):
    model = AdsorptionIsotherms()
    Ce_smooth = np.linspace(Ce_exp.min(), Ce_exp.max(), 300)
    qe_smooth = model.predict(model_name, Ce_smooth, result['parameters'])
    fig = go.Figure()

    if show_ci and result.get('covariance') is not None:
        pcov   = result['covariance']
        pnames = list(result['parameters'].keys())
        pvals  = np.array([result['parameters'][k] for k in pnames])
        model_fn = getattr(model, model_name)
        eps = 1e-6 * np.abs(pvals) + 1e-12
        J   = np.zeros((len(Ce_smooth), len(pvals)))
        for i in range(len(pvals)):
            p_plus  = pvals.copy(); p_plus[i]  += eps[i]
            p_minus = pvals.copy(); p_minus[i] -= eps[i]
            J[:, i] = (model_fn(Ce_smooth, *p_plus) -
                       model_fn(Ce_smooth, *p_minus)) / (2 * eps[i])
        var_pred = np.einsum('ij,jk,ik->i', J, pcov, J)
        std_pred = np.sqrt(np.maximum(var_pred, 0))
        ci_band  = 1.96 * std_pred
        fig.add_trace(go.Scatter(
            x=np.concatenate([Ce_smooth, Ce_smooth[::-1]]),
            y=np.concatenate([qe_smooth + ci_band, (qe_smooth - ci_band)[::-1]]),
            fill='toself', fillcolor='rgba(31,119,180,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip', name='95% CI'
        ))

    fig.add_trace(go.Scatter(
        x=Ce_smooth, y=qe_smooth, mode='lines',
        line=dict(color='#1f77b4', width=2.5),
        name=f'{model_name.capitalize()} fit'
    ))
    fig.add_trace(go.Scatter(
        x=Ce_exp, y=qe_exp, mode='markers',
        marker=dict(color='#d62728', size=10, symbol='circle',
                    line=dict(color='white', width=1)),
        name='Experimental'
    ))
    Ce_label = f"Ce ({Ce_unit})" if Ce_unit else "Ce"
    qe_label = f"qe ({qe_unit})" if qe_unit else "qe"
    fig.update_layout(
        xaxis_title=Ce_label, yaxis_title=qe_label,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified', plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False),
    )
    return fig


def results_to_dataframe(all_results):
    rows = []
    for name, res in all_results.items():
        row = {'Model': name.replace('_', ' ').title(),
               'R²': round(res['r_squared'], 5),
               'RMSE': round(res['rmse'], 5)}
        for param, val in res['parameters'].items():
            row[param] = round(val, 5)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_report_csv(Ce_exp, qe_exp, all_results, Ce_unit, qe_unit):
    lines = ["Adsorption Isotherm Fitting — Results Report", "",
             "MODEL COMPARISON", "Model,R²,RMSE,Parameters"]
    for name, res in all_results.items():
        param_str = " | ".join(
            f"{k}={v:.5f} ±{res['confidence_95'].get(k,0):.5f}"
            for k, v in res['parameters'].items()
        )
        lines.append(f"{name},{res['r_squared']:.5f},{res['rmse']:.5f},{param_str}")
    lines += ["", "EXPERIMENTAL DATA", f"Ce ({Ce_unit}),qe ({qe_unit})"]
    for ce, qe in zip(Ce_exp, qe_exp):
        lines.append(f"{ce},{qe}")
    return "\n".join(lines)


def _load_csv_data(uploaded_file, col_names=("Ce", "qe")):
    """Parse an uploaded CSV and return two clean numpy arrays."""
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    c1, c2 = df.columns[0], df.columns[1]
    a = pd.to_numeric(df[c1].astype(str).str.replace(',', '.'), errors='coerce').values
    b = pd.to_numeric(df[c2].astype(str).str.replace(',', '.'), errors='coerce').values
    return a, b


# ═════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ═════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("## 🔬 Adsorption Library")
    st.markdown(
        "Open-source numerical toolkit for fitting adsorption models "
        "to experimental data. Choose a module below to get started."
    )
    st.markdown("")

    col_eq, col_gap, col_kin = st.columns([1, 0.06, 1])

    with col_eq:
        st.markdown("""
        <div class="module-card">
          <span class="tag tag-eq">Equilibrium</span>
          <h2>⚖️ Isotherm Fitting</h2>
          <p>Fit experimental (Ce, qe) data to seven classical adsorption isotherm models
          with nonlinear regression, 95% confidence intervals, and interactive Plotly charts.</p>
          <ul>
            <li>Henry, Langmuir, Freundlich, Temkin</li>
            <li>BET, Dubinin–Radushkevich, Redlich–Peterson</li>
            <li>Model comparison table (R², RMSE)</li>
            <li>Linearised forms (Langmuir, Freundlich)</li>
            <li>CSV export of all results</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Open Isotherm Fitting →", use_container_width=True, key="btn_eq"):
            st.session_state['page'] = 'equilibrium'
            st.rerun()

    with col_kin:
        st.markdown("""
        <div class="module-card">
          <span class="tag tag-kin">Kinetics</span>
          <h2>⏱️ Kinetics Fitting</h2>
          <p>Fit time-resolved adsorption data from batch or fixed-bed experiments
          using reaction models, diffusional PDE models, and empirical breakthrough models.</p>
          <ul>
            <li>Batch: PFO, PSO, Elovich, PVSDM / PVDM / SDM</li>
            <li>Fixed-bed: Bohart–Adams, Thomas, Yoon–Nelson, Wolborska</li>
            <li>Design tools: LUB, BDST scale-up calculations</li>
            <li>Statistical validation: t-test, F-test, AIC</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Open Kinetics Fitting →", use_container_width=True, key="btn_kin"):
            st.session_state['page'] = 'kinetics'
            st.rerun()

    st.markdown("---")
    st.caption(
        "MIT License · Open source · "
        "[GitHub](https://github.com) · "
        "[Report an issue](https://github.com)"
    )


# ═════════════════════════════════════════════════════════════════════
#  EQUILIBRIUM MODULE  (original app.py content, preserved)
# ═════════════════════════════════════════════════════════════════════

def page_equilibrium():
    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        if st.button("← Back to Home"):
            st.session_state['page'] = 'home'
            st.rerun()
        st.title("⚖️ Isotherm Fitting")
        st.caption("Equilibrium module")
        st.divider()

        st.subheader("Axis units")
        Ce_unit = st.text_input("Ce unit", value="mg/L")
        qe_unit = st.text_input("qe unit", value="mg/g")
        st.divider()

        st.subheader("Models to fit")
        all_model_names = list(AdsorptionIsotherms().models.keys())
        selected_models = st.multiselect(
            "Select models", options=all_model_names, default=all_model_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        st.divider()

        st.subheader("Plot options")
        show_ci = st.toggle("Show 95% confidence band", value=True)

        st.divider()
        st.markdown("**About**\n\nNonlinear fitting via `scipy.optimize.curve_fit`.")

    # ── Main tabs ─────────────────────────────────────────────────────
    st.title("Adsorption Isotherm Fitting")
    st.markdown(
        "Upload your experimental data, select the isotherm models to test, "
        "and get publication-ready statistics and plots."
    )

    tab_about, tab_data, tab_results, tab_linearized = st.tabs([
        "ℹ️ About & Help", "📂 Data Input",
        "📊 Fitting & Results", "📈 Linearized Forms"
    ])

    # ── About tab ─────────────────────────────────────────────────────
    with tab_about:
        st.subheader("About this tool")
        st.markdown(
            "Nonlinear fitting of adsorption equilibrium data to seven classical "
            "isotherm models. Upload a CSV, choose your models, and download "
            "publication-ready results."
        )

        models_info = [
            {"name": "Henry",     "equation": "qe = KH·Ce",
             "application": "Dilute solutions, linear range"},
            {"name": "Langmuir",  "equation": "qe = (qmax·KL·Ce)/(1+KL·Ce)",
             "application": "Monolayer, homogeneous surface"},
            {"name": "Freundlich","equation": "qe = KF·Ce^(1/n)",
             "application": "Heterogeneous surface"},
            {"name": "Temkin",    "equation": "qe = B·ln(A·Ce)",
             "application": "Adsorbate–adsorbent interactions"},
            {"name": "BET",       "equation": "multilayer formula",
             "application": "Multilayer adsorption"},
            {"name": "Dubinin–Radushkevich", "equation": "qe = qs·exp(−K·ε²)",
             "application": "Pore-filling mechanism"},
            {"name": "Redlich–Peterson",     "equation": "qe = KR·Ce/(1+aR·Ce^g)",
             "application": "Hybrid Langmuir–Freundlich"},
        ]

        header_cols = st.columns([1.2, 1.8, 2])
        header_cols[0].markdown("**Model**")
        header_cols[1].markdown("**Equation**")
        header_cols[2].markdown("**Application**")
        st.divider()
        for m in models_info:
            cols = st.columns([1.2, 1.8, 2])
            cols[0].markdown(m["name"])
            cols[1].markdown(f"`{m['equation']}`")
            cols[2].markdown(m["application"])
            st.divider()

        with st.expander("How do I prepare my CSV file?"):
            st.markdown("Two columns: `Ce` then `qe`. Header row optional. Min 3 points.")

        with st.expander("What does R² mean?"):
            st.markdown("R² = 1 → perfect fit. R² > 0.99 → excellent. R² < 0.90 → poor.")

        with st.expander("Should I use linearized or nonlinear fitting?"):
            st.markdown(
                "Always prefer **nonlinear** fitting. Linearization distorts the "
                "error structure and produces biased parameter estimates."
            )

    # ── Data tab ──────────────────────────────────────────────────────
    with tab_data:
        st.subheader("Load experimental data")
        input_method = st.radio(
            "Choose input method",
            ["Upload CSV file", "Paste data manually", "Use example data"],
            horizontal=True
        )

        Ce_exp = None
        qe_exp = None

        if input_method == "Upload CSV file":
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "txt"])
            if uploaded_file is not None:
                try:
                    Ce_raw, qe_raw = _load_csv_data(uploaded_file)
                    Ce_exp, qe_exp = validate_and_clean(Ce_raw, qe_raw)
                    st.success(f"✓ Loaded {len(Ce_exp)} valid data points.")
                    st.dataframe(pd.DataFrame(
                        {f"Ce ({Ce_unit})": Ce_exp, f"qe ({qe_unit})": qe_exp}
                    ), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        elif input_method == "Paste data manually":
            raw_text = st.text_area(
                "Paste your data (Ce, qe per line)", height=220,
                placeholder="0.1, 0.9\n0.5, 3.2\n1.0, 4.8"
            )
            if raw_text.strip():
                try:
                    Ce_list, qe_list = [], []
                    for line in raw_text.strip().split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        sep = ';' if ';' in line else ','
                        parts = line.split(sep)
                        Ce_list.append(float(parts[0].strip().replace(',', '.')))
                        qe_list.append(float(parts[1].strip().replace(',', '.')))
                    Ce_exp, qe_exp = validate_and_clean(Ce_list, qe_list)
                    st.success(f"✓ Parsed {len(Ce_exp)} valid data points.")
                    st.dataframe(pd.DataFrame(
                        {f"Ce ({Ce_unit})": Ce_exp, f"qe ({qe_unit})": qe_exp}
                    ), use_container_width=True)
                except Exception as e:
                    st.error(f"Parsing error: {e}")
        else:
            st.info("Using built-in Langmuir-like example dataset.")
            Ce_exp = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
            qe_exp = np.array([0.9, 3.2, 4.8, 6.5, 8.2,  9.0,  9.5])
            st.dataframe(pd.DataFrame(
                {f"Ce ({Ce_unit})": Ce_exp, f"qe ({qe_unit})": qe_exp}
            ), use_container_width=True)

        if Ce_exp is not None and qe_exp is not None:
            st.session_state['eq_Ce'] = Ce_exp
            st.session_state['eq_qe'] = qe_exp
            st.session_state['eq_ready'] = True
        else:
            st.session_state['eq_ready'] = False

    # ── Results tab ───────────────────────────────────────────────────
    with tab_results:
        if not st.session_state.get('eq_ready', False):
            st.info("👈 Please load your data in the **Data Input** tab first.")
        else:
            Ce_exp = st.session_state['eq_Ce']
            qe_exp = st.session_state['eq_qe']

            if not selected_models:
                st.warning("Select at least one model in the sidebar.")
                st.stop()

            with st.spinner("Fitting models…"):
                model = AdsorptionIsotherms()
                all_results = {}
                errors = {}
                for mname in selected_models:
                    try:
                        res = cached_fit_isotherm(model, mname,
                                                  tuple(Ce_exp), tuple(qe_exp))
                        all_results[mname] = res
                    except Exception as e:
                        errors[mname] = str(e)

            for mname, err in errors.items():
                st.warning(f"**{mname}** could not be fitted: {err}")

            if not all_results:
                st.error("No models could be fitted. Check your data.")
                st.stop()

            all_results = dict(sorted(all_results.items(),
                                      key=lambda x: x[1]['r_squared'],
                                      reverse=True))
            best_model = list(all_results.keys())[0]
            best_res   = all_results[best_model]

            col1, col2, col3 = st.columns(3)
            col1.metric("Best model",  best_model.replace('_', ' ').title())
            col2.metric("R²",          f"{best_res['r_squared']:.5f}")
            col3.metric("RMSE",        f"{best_res['rmse']:.5f}")
            st.divider()

            st.subheader("Isotherm plot")
            model_to_plot = st.selectbox(
                "Select model to plot", options=list(all_results.keys()),
                index=0, format_func=lambda x: x.replace('_', ' ').title()
            )
            fig = build_plotly_figure(
                model_to_plot, Ce_exp, qe_exp,
                all_results[model_to_plot], Ce_unit, qe_unit, show_ci
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Fitted parameters")
            sel = all_results[model_to_plot]
            param_rows = [
                {'Parameter': p, 'Value': round(v, 6),
                 '± 95% CI': round(sel['confidence_95'].get(p, np.nan), 6)}
                for p, v in sel['parameters'].items()
            ]
            st.dataframe(pd.DataFrame(param_rows),
                         use_container_width=True, hide_index=True)
            st.divider()

            st.subheader("Model comparison")
            comparison_df = results_to_dataframe(all_results)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True,
                         column_config={'R²': st.column_config.ProgressColumn(
                             'R²', min_value=0, max_value=1, format="%.5f")})
            st.divider()

            st.subheader("Export results")
            dl1, dl2 = st.columns(2)
            with dl1:
                csv_report = generate_report_csv(
                    Ce_exp, qe_exp, all_results, Ce_unit, qe_unit)
                st.download_button("⬇ Full report (CSV)", data=csv_report,
                                   file_name="isotherm_results.csv", mime="text/csv")
            with dl2:
                st.download_button("⬇ Comparison table (CSV)",
                                   data=comparison_df.to_csv(index=False),
                                   file_name="model_comparison.csv", mime="text/csv")

    # ── Linearized tab ────────────────────────────────────────────────
    with tab_linearized:
        if not st.session_state.get('eq_ready', False):
            st.info("👈 Please load your data in the **Data Input** tab first.")
        else:
            Ce_exp = st.session_state['eq_Ce']
            qe_exp = st.session_state['eq_qe']
            st.subheader("Linearized isotherm forms")
            st.info(
                "Nonlinear fitting (Results tab) is statistically superior. "
                "These linearizations are provided for cross-checking with published literature."
            )
            lin_col1, lin_col2 = st.columns(2)
            model_obj = AdsorptionIsotherms()

            with lin_col1:
                st.markdown("**Langmuir: Ce/qe vs Ce**")
                try:
                    lang_lin = model_obj.linearize_langmuir(Ce_exp, qe_exp)
                    x_fit = np.linspace(Ce_exp.min(), Ce_exp.max(), 100)
                    y_fit = lang_lin['slope'] * x_fit + lang_lin['intercept']
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=lang_lin['x'], y=lang_lin['y'],
                                               mode='markers',
                                               marker=dict(color='#d62728', size=10)))
                    fig_l.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                                               line=dict(color='#1f77b4', width=2)))
                    fig_l.update_layout(plot_bgcolor='white',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        margin=dict(t=10),
                                        xaxis_title=f"Ce ({Ce_unit})",
                                        yaxis_title=f"Ce/qe")
                    st.plotly_chart(fig_l, use_container_width=True)
                    st.markdown(
                        f"qmax = **{lang_lin['qmax']:.4f}** {qe_unit}  |  "
                        f"KL = **{lang_lin['KL']:.4f}** L/{Ce_unit}"
                    )
                except Exception as e:
                    st.error(f"Langmuir linearization failed: {e}")

            with lin_col2:
                st.markdown("**Freundlich: ln(qe) vs ln(Ce)**")
                try:
                    frnd_lin = model_obj.linearize_freundlich(Ce_exp, qe_exp)
                    x_fit = np.linspace(frnd_lin['x'].min(), frnd_lin['x'].max(), 100)
                    y_fit = frnd_lin['slope'] * x_fit + frnd_lin['intercept']
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=frnd_lin['x'], y=frnd_lin['y'],
                                               mode='markers',
                                               marker=dict(color='#d62728', size=10)))
                    fig_f.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                                               line=dict(color='#2ca02c', width=2)))
                    fig_f.update_layout(plot_bgcolor='white',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        margin=dict(t=10),
                                        xaxis_title="ln(Ce)",
                                        yaxis_title="ln(qe)")
                    st.plotly_chart(fig_f, use_container_width=True)
                    st.markdown(
                        f"KF = **{frnd_lin['KF']:.4f}**  |  "
                        f"n = **{frnd_lin['n']:.4f}**  |  "
                        f"1/n = **{frnd_lin['slope']:.4f}**"
                    )
                except Exception as e:
                    st.error(f"Freundlich linearization failed: {e}")


# ═════════════════════════════════════════════════════════════════════
#  KINETICS MODULE
# ═════════════════════════════════════════════════════════════════════

def page_kinetics():
    with st.sidebar:
        if st.button("← Back to Home"):
            st.session_state['page'] = 'home'
            st.rerun()
        st.title("⏱️ Kinetics Fitting")
        st.caption("Kinetics module")
        st.divider()
        st.subheader("System type")
        system_type = st.radio(
            "Select system", ["Batch (discontinuous)", "Fixed-bed (continuous)"],
            label_visibility="collapsed"
        )
        st.divider()
        st.markdown("**Statistical options**")
        alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10],
                                 value=0.05)
        st.divider()
        st.markdown("**About**\n\nFits kinetic adsorption data using scipy BDF/TRF solvers.")

    st.title("Adsorption Kinetics Fitting")
    st.markdown(
        "Fit time-resolved adsorption data to reaction, diffusional, "
        "or breakthrough models. Upload your data and configure your system below."
    )

    if system_type == "Batch (discontinuous)":
        _kinetics_batch(alpha)
    else:
        _kinetics_fixedbed(alpha)


# ─────────────────────────────────────────────────────────────────────
#  BATCH KINETICS
# ─────────────────────────────────────────────────────────────────────

def _kinetics_batch(alpha):
    tab_about, tab_data, tab_reaction, tab_diffusion = st.tabs([
        "ℹ️ About", "📂 Data Input",
        "⚗️ Reaction Models", "🧬 Diffusion Models (PVSDM)"
    ])

    with tab_about:
        st.subheader("Batch kinetic models")
        st.markdown("""
**Reaction models** fit solid-phase loading *q(t)* directly using algebraic equations.
They are fast and widely used in literature.

| Model | Equation | Key parameter |
|---|---|---|
| Pseudo-first-order (PFO) | qₜ = qₑ (1 − e^{−k₁t}) | k₁ [1/min] |
| Pseudo-second-order (PSO) | qₜ = k₂qₑ²t / (1+k₂qₑt) | k₂ [g/(mg·min)] |
| Elovich | qₜ = (1/β) ln(1+αβt) | α, β |

**Diffusion models** (PVSDM/PVDM/SDM) solve the intraparticle PDE numerically
using the Method of Lines + BDF stiff ODE solver. They require the bulk concentration
profile C(t) and additional system parameters (particle radius, porosity, density).

**Model selection**: use AIC (lower = better) to compare models objectively.
        """)

        with st.expander("What is the Method of Lines (MOL)?"):
            st.markdown("""
The PVSDM PDE is discretised over N radial shells in the adsorbent particle.
Second-order finite differences convert spatial derivatives to algebraic expressions,
yielding a system of ODEs integrated with scipy's `solve_ivp(method='BDF')` —
the Python equivalent of MATLAB's `ode15s`, designed for stiff systems.
            """)

    with tab_data:
        st.subheader("Upload batch kinetic data")
        st.markdown("""
Upload a CSV with:
- **Two-column format**: `time, qt` (reaction models)
- **Three-column format**: `time, C_bulk, qt` (also enables diffusion models)

Time in minutes, qt in mg/g, C_bulk in mg/L.
        """)

        data_src = st.radio("Input method",
                            ["Upload CSV", "Paste data", "Use example data"],
                            horizontal=True)

        t_exp = qt_exp = C_exp = None

        if data_src == "Upload CSV":
            f = st.file_uploader("CSV file (time, qt) or (time, C_bulk, qt)",
                                 type=["csv", "txt"])
            if f is not None:
                try:
                    df = pd.read_csv(f)
                    df = df.apply(lambda col: pd.to_numeric(
                        col.astype(str).str.replace(',', '.'), errors='coerce'))
                    if df.shape[1] >= 3:
                        t_exp  = df.iloc[:, 0].values
                        C_exp  = df.iloc[:, 1].values
                        qt_exp = df.iloc[:, 2].values
                        st.success(f"✓ Loaded {len(t_exp)} points (t, C, q).")
                    else:
                        t_exp  = df.iloc[:, 0].values
                        qt_exp = df.iloc[:, 1].values
                        st.success(f"✓ Loaded {len(t_exp)} points (t, q).")
                    mask = np.isfinite(t_exp) & np.isfinite(qt_exp)
                    t_exp, qt_exp = t_exp[mask], qt_exp[mask]
                    if C_exp is not None:
                        C_exp = C_exp[mask]
                    preview = {'t (min)': t_exp, 'qt (mg/g)': qt_exp}
                    if C_exp is not None:
                        preview['C_bulk (mg/L)'] = C_exp
                    st.dataframe(pd.DataFrame(preview), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        elif data_src == "Paste data":
            raw = st.text_area("Paste data (t, qt) or (t, C_bulk, qt) per line",
                               height=200, placeholder="0, 0.0\n5, 2.1\n10, 3.8")
            if raw.strip():
                try:
                    rows = []
                    for line in raw.strip().split('\n'):
                        parts = [float(x.strip().replace(',', '.'))
                                 for x in line.replace(';', ',').split(',')
                                 if x.strip()]
                        if len(parts) >= 2:
                            rows.append(parts[:3] if len(parts) >= 3 else parts[:2])
                    arr = np.array([r + [np.nan] * (3 - len(r)) for r in rows])
                    t_exp  = arr[:, 0]
                    qt_exp = arr[:, 1]
                    if not np.all(np.isnan(arr[:, 2])):
                        C_exp = arr[:, 2]
                    st.success(f"✓ Parsed {len(t_exp)} points.")
                except Exception as e:
                    st.error(f"Parsing error: {e}")
        else:
            st.info("Using synthetic PSO example data.")
            t_exp  = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240], dtype=float)
            qt_exp = np.array([0, 1.8, 3.1, 4.9, 6.2, 7.5, 8.4, 9.2, 9.7, 10.0, 10.2])
            st.dataframe(pd.DataFrame({'t (min)': t_exp, 'qt (mg/g)': qt_exp}),
                         use_container_width=True)

        if t_exp is not None:
            st.session_state['kin_t']  = t_exp
            st.session_state['kin_qt'] = qt_exp
            st.session_state['kin_C']  = C_exp
            st.session_state['kin_batch_ready'] = True

    # ── Reaction models tab ───────────────────────────────────────────
    with tab_reaction:
        if not st.session_state.get('kin_batch_ready', False):
            st.info("👈 Load your data in the **Data Input** tab first.")
        else:
            t_exp  = st.session_state['kin_t']
            qt_exp = st.session_state['kin_qt']

            t_unit = "min"
            q_unit = "mg/g"

            with st.spinner("Fitting PFO, PSO, and Elovich models…"):
                bk = BatchKinetics()
                ks = KineticsStats()
                results = {}
                for name, fn in [("PFO", bk.fit_pfo),
                                  ("PSO", bk.fit_pso),
                                  ("Elovich", bk.fit_elovich)]:
                    try:
                        results[name] = fn(t_exp, qt_exp)
                    except Exception as e:
                        st.warning(f"{name} failed: {e}")

            if not results:
                st.error("No models converged. Check your data.")
                st.stop()

            # Metrics summary
            cols = st.columns(len(results))
            for i, (name, res) in enumerate(results.items()):
                with cols[i]:
                    st.metric(f"{name} — R²", f"{res['r_squared']:.4f}")

            st.divider()

            # Select model to plot
            model_sel = st.selectbox("Select model to inspect",
                                     list(results.keys()))
            sel_res = results[model_sel]

            # Plot
            t_smooth = np.linspace(0, t_exp.max(), 400)
            if model_sel == "PFO":
                qt_smooth = bk.pfo_equation(t_smooth,
                                             sel_res['parameters']['qe'],
                                             sel_res['parameters']['k1'])
            elif model_sel == "PSO":
                qt_smooth = bk.pso_equation(t_smooth,
                                             sel_res['parameters']['qe'],
                                             sel_res['parameters']['k2'])
            else:
                qt_smooth = bk.elovich_equation(t_smooth,
                                                 sel_res['parameters']['alpha'],
                                                 sel_res['parameters']['beta'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_smooth, y=qt_smooth, mode='lines',
                line=dict(color='#534AB7', width=2.5), name=f'{model_sel} fit'
            ))
            fig.add_trace(go.Scatter(
                x=t_exp, y=qt_exp, mode='markers',
                marker=dict(color='#d62728', size=10,
                            line=dict(color='white', width=1)),
                name='Experimental'
            ))
            fig.update_layout(
                xaxis_title=f"Time ({t_unit})",
                yaxis_title=f"qt ({q_unit})",
                plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom',
                            y=1.02, xanchor='right', x=1),
                hovermode='x unified',
                margin=dict(t=20, b=10, l=10, r=10),
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Parameters table
            st.subheader("Fitted parameters")
            param_rows = [
                {'Parameter': p, 'Value': round(v, 6),
                 '± 95% CI': round(sel_res['confidence_95'].get(p, np.nan), 6)}
                for p, v in sel_res['parameters'].items()
            ]
            st.dataframe(pd.DataFrame(param_rows),
                         use_container_width=True, hide_index=True)

            # Statistical test
            st.subheader("Statistical validation (t-test)")
            t_res = ks.ttest(qt_exp, sel_res['qt_predicted'],
                             n_params=len(sel_res['parameters']), alpha=alpha)
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("t statistic",   f"{t_res['t_statistic']:.4f}")
            tc2.metric("t critical",    f"{t_res['t_critical']:.4f}")
            tc3.metric("p-value",       f"{t_res['p_value']:.4f}")
            tc4.metric("Reject H₀?",   "Yes" if t_res['reject_H0'] else "No")
            st.info(t_res['interpretation'])

            st.divider()
            st.subheader("Model comparison (all three)")
            comp_rows = []
            for name, res in results.items():
                t_r = ks.ttest(qt_exp, res['qt_predicted'],
                               n_params=len(res['parameters']), alpha=alpha)
                comp_rows.append({
                    'Model': name,
                    'R²':    round(res['r_squared'], 5),
                    'RMSE':  round(res['rmse'], 5),
                    'AIC':   round(res['aic'], 2),
                    't p-value': round(t_r['p_value'], 4),
                    'Bias?': 'Yes' if t_r['reject_H0'] else 'No',
                    **{p: round(v, 5) for p, v in res['parameters'].items()}
                })
            comp_df = pd.DataFrame(comp_rows).sort_values('AIC')
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Export")
            st.download_button(
                "⬇ Download comparison CSV",
                data=comp_df.to_csv(index=False),
                file_name="batch_reaction_results.csv",
                mime="text/csv"
            )

    # ── Diffusion models tab ──────────────────────────────────────────
    with tab_diffusion:
        if not st.session_state.get('kin_batch_ready', False):
            st.info("👈 Load your data in the **Data Input** tab first.")
        else:
            C_exp = st.session_state.get('kin_C')
            t_exp = st.session_state['kin_t']

            if C_exp is None:
                st.warning(
                    "Diffusion models require the bulk concentration profile C(t). "
                    "Please upload a three-column CSV: `time, C_bulk, qt`."
                )
                st.stop()

            st.subheader("System parameters")
            st.markdown(
                "Configure the physical parameters of your system. "
                "These are required to solve the intraparticle PDE."
            )

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                Cb    = st.number_input("Initial bulk conc. C₀ (mg/L)", value=50.0, min_value=0.01)
                R_um  = st.number_input("Particle radius R (µm)", value=500.0, min_value=1.0)
                eps_p = st.number_input("Particle porosity ε_p", value=0.5,
                                        min_value=0.01, max_value=0.99)
            with pc2:
                rho_p = st.number_input("Particle density ρ_p (g/L)", value=800.0, min_value=1.0)
                V     = st.number_input("Solution volume V (L)", value=0.5, min_value=0.001)
                m     = st.number_input("Adsorbent mass m (g)", value=1.0, min_value=0.001)
            with pc3:
                kf_surf = st.number_input("Surface rate k_s (1/s)", value=0.01,
                                          min_value=1e-6, format="%.4f")
                N_nodes = st.slider("Radial nodes N", min_value=5, max_value=30, value=12)

            R = R_um * 1e-6  # convert µm → m

            st.markdown("**Initial parameter guesses for optimizer**")
            gi1, gi2, gi3 = st.columns(3)
            kf_init = gi1.number_input("kf guess (m/s)", value=1e-5,
                                       min_value=1e-9, format="%.2e")
            Dp_init = gi2.number_input("Dp guess (m²/s)", value=1e-11,
                                       min_value=1e-16, format="%.2e")
            Ds_init = gi3.number_input("Ds guess (m²/s)", value=1e-13,
                                       min_value=1e-16, format="%.2e")

            model_choice = st.selectbox(
                "Model variant",
                ["PVSDM (pore + surface diffusion)",
                 "PVDM  (pore diffusion only)",
                 "SDM   (surface diffusion only)"]
            )
            model_key = model_choice.split()[0].lower()

            # Simple linear isotherm as default (user can adapt via code)
            st.info(
                "A linear Henry isotherm (qₑ = KH · Cₚ) is used as the default "
                "equilibrium relationship inside the particle. "
                "Fit your isotherm in the Equilibrium module first, then plug in the "
                "function in the Python code for nonlinear isotherms."
            )
            KH_inner = st.number_input("Henry constant KH (mg/g per mg/L)",
                                       value=0.5, min_value=0.0)

            def isotherm_fn(Cp):
                return float(KH_inner) * Cp

            if st.button("▶ Run diffusion model fitting", use_container_width=True):
                t_s = t_exp * 60.0  # min → s
                with st.spinner("Running MOL solver + parameter estimation…"):
                    try:
                        bk = BatchKinetics()
                        res = bk.fit_pvsdm(
                            t_s, C_exp, Cb=Cb, R=R, rho_p=rho_p,
                            eps_p=eps_p, V=V, m=m,
                            isotherm_fn=isotherm_fn,
                            kf_init=kf_init, Dp_init=Dp_init, Ds_init=Ds_init,
                            kf_surf=kf_surf, N=N_nodes, model=model_key
                        )
                        st.success("Fitting complete!")

                        dm1, dm2, dm3 = st.columns(3)
                        dm1.metric("R²",   f"{res['r_squared']:.4f}")
                        dm2.metric("RMSE", f"{res['rmse']:.4f}")
                        dm3.metric("AIC",  f"{res['aic']:.1f}")

                        st.subheader("Estimated parameters")
                        p_df = pd.DataFrame(
                            [{'Parameter': k, 'Value': f"{v:.4e}"}
                             for k, v in res['parameters'].items()]
                        )
                        st.dataframe(p_df, use_container_width=True, hide_index=True)

                        # Plot C_bulk predicted vs observed
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Scatter(
                            x=t_exp, y=C_exp, mode='markers',
                            marker=dict(color='#d62728', size=10,
                                        line=dict(color='white', width=1)),
                            name='C_bulk experimental'
                        ))
                        fig_d.add_trace(go.Scatter(
                            x=t_exp, y=res['C_predicted'], mode='lines',
                            line=dict(color='#534AB7', width=2.5),
                            name=f'{model_key.upper()} fit'
                        ))
                        fig_d.update_layout(
                            xaxis_title="Time (min)", yaxis_title="C_bulk (mg/L)",
                            plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                            hovermode='x unified',
                            legend=dict(orientation='h', y=1.05, x=1,
                                        xanchor='right', yanchor='bottom'),
                            margin=dict(t=20, b=10, l=10, r=10),
                            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                        )
                        st.plotly_chart(fig_d, use_container_width=True)

                    except Exception as e:
                        st.error(f"Solver error: {e}")
                        st.markdown(
                            "**Troubleshooting**: try adjusting initial guesses, "
                            "increasing radial nodes, or checking system units."
                        )


# ─────────────────────────────────────────────────────────────────────
#  FIXED-BED KINETICS
# ─────────────────────────────────────────────────────────────────────

def _kinetics_fixedbed(alpha):
    tab_about, tab_data, tab_bt, tab_design = st.tabs([
        "ℹ️ About", "📂 Data Input",
        "📉 Breakthrough Models", "📐 Design Tools (LUB & BDST)"
    ])

    with tab_about:
        st.subheader("Fixed-bed (column) kinetic models")
        st.markdown("""
Breakthrough models predict the **effluent concentration ratio C/C₀** as a function
of time at a given flow rate and bed geometry.

| Model | Key parameters |
|---|---|
| Bohart–Adams | k_BA [L/(mg·min)], N₀ [mg/L] |
| Thomas | k_Th [L/(mg·min)], q_Th [mg/g] |
| Yoon–Nelson | k_YN [1/min], τ (50% breakthrough time) [min] |
| Wolborska | β [1/min], N₀ [mg/L] |

**Design tools**:
- **LUB** (Length of Unused Bed): fraction of bed not yet utilised at breakthrough
- **BDST** (Bed Depth Service Time): linear scale-up relationship between bed height and service time
        """)

    with tab_data:
        st.subheader("Upload breakthrough curve data")
        st.markdown("CSV with two columns: `time (min)`, `C/C0` (dimensionless, 0–1).")

        data_src = st.radio("Input method",
                            ["Upload CSV", "Paste data", "Use example data"],
                            horizontal=True)

        t_bt = CC0 = None

        if data_src == "Upload CSV":
            f = st.file_uploader("CSV: time (min), C/C0", type=["csv", "txt"])
            if f is not None:
                try:
                    df = pd.read_csv(f)
                    df = df.apply(lambda col: pd.to_numeric(
                        col.astype(str).str.replace(',', '.'), errors='coerce'))
                    t_bt = df.iloc[:, 0].values
                    CC0  = df.iloc[:, 1].values
                    mask = np.isfinite(t_bt) & np.isfinite(CC0)
                    t_bt, CC0 = t_bt[mask], CC0[mask]
                    st.success(f"✓ Loaded {len(t_bt)} points.")
                    st.dataframe(pd.DataFrame({'t (min)': t_bt, 'C/C0': CC0}),
                                 use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        elif data_src == "Paste data":
            raw = st.text_area("Paste (time, C/C0) per line", height=180,
                               placeholder="0, 0.00\n30, 0.05\n60, 0.21")
            if raw.strip():
                try:
                    rows = []
                    for line in raw.strip().split('\n'):
                        parts = [float(x.strip().replace(',', '.'))
                                 for x in line.replace(';', ',').split(',')
                                 if x.strip()]
                        if len(parts) >= 2:
                            rows.append(parts[:2])
                    arr = np.array(rows)
                    t_bt, CC0 = arr[:, 0], arr[:, 1]
                    st.success(f"✓ Parsed {len(t_bt)} points.")
                except Exception as e:
                    st.error(f"Parsing error: {e}")
        else:
            st.info("Using synthetic Thomas-like breakthrough example.")
            t_bt = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                             120, 150, 180, 210, 240], dtype=float)
            CC0  = np.array([0.00, 0.01, 0.02, 0.04, 0.08, 0.14, 0.22,
                             0.33, 0.46, 0.59, 0.71, 0.86, 0.94, 0.97,
                             0.99, 1.00])
            st.dataframe(pd.DataFrame({'t (min)': t_bt, 'C/C0': CC0}),
                         use_container_width=True)

        if t_bt is not None and CC0 is not None:
            st.session_state['kin_t_bt']  = t_bt
            st.session_state['kin_CC0']   = CC0
            st.session_state['kin_fb_ready'] = True

    # ── Breakthrough fitting tab ──────────────────────────────────────
    with tab_bt:
        if not st.session_state.get('kin_fb_ready', False):
            st.info("👈 Load your data in the **Data Input** tab first.")
        else:
            t_bt = st.session_state['kin_t_bt']
            CC0  = st.session_state['kin_CC0']

            st.subheader("Column parameters")
            bp1, bp2, bp3 = st.columns(3)
            C0_col = bp1.number_input("Influent C₀ (mg/L)", value=50.0, min_value=0.01)
            Q_col  = bp2.number_input("Flow rate Q (L/min)", value=0.01, min_value=1e-6,
                                      format="%.4f")
            Z_col  = bp3.number_input("Bed height Z (cm)", value=10.0, min_value=0.1)
            m_col  = st.number_input("Adsorbent mass (g)", value=5.0, min_value=0.001)

            with st.spinner("Fitting all breakthrough models…"):
                fb = FixedBedKinetics()
                ks = KineticsStats()
                bt_results = {}
                bt_errors  = {}
                fits = {
                    "Bohart-Adams": lambda: fb.fit_bohart_adams(t_bt, CC0, C0_col, Q_col, Z_col),
                    "Thomas":       lambda: fb.fit_thomas(t_bt, CC0, C0_col, Q_col, m_col),
                    "Yoon-Nelson":  lambda: fb.fit_yoon_nelson(t_bt, CC0),
                    "Wolborska":    lambda: fb.fit_wolborska(t_bt, CC0, C0_col, Q_col, Z_col),
                }
                for name, fn in fits.items():
                    try:
                        bt_results[name] = fn()
                    except Exception as e:
                        bt_errors[name] = str(e)

            for name, err in bt_errors.items():
                st.warning(f"**{name}** failed: {err}")

            if not bt_results:
                st.error("No models converged.")
                st.stop()

            # Metric row
            cols = st.columns(len(bt_results))
            for i, (name, res) in enumerate(bt_results.items()):
                with cols[i]:
                    st.metric(f"{name}", f"R² = {res['r_squared']:.4f}")

            st.divider()

            # Overlay plot of all models
            st.subheader("Breakthrough curves — all models")
            t_smooth = np.linspace(t_bt.min(), t_bt.max(), 500)
            colors   = ['#534AB7', '#1f77b4', '#2ca02c', '#e07b39']
            fig_bt   = go.Figure()

            for (name, res), color in zip(bt_results.items(), colors):
                try:
                    if name == "Bohart-Adams":
                        cc_smooth = fb.bohart_adams(
                            t_smooth,
                            res['parameters']['kBA'], res['parameters']['N0'],
                            C0_col, Q_col, Z_col
                        )
                    elif name == "Thomas":
                        cc_smooth = fb.thomas(
                            t_smooth,
                            res['parameters']['kTh'], res['parameters']['qTh'],
                            C0_col, Q_col, m_col
                        )
                    elif name == "Yoon-Nelson":
                        cc_smooth = fb.yoon_nelson(
                            t_smooth,
                            res['parameters']['kYN'], res['parameters']['tau']
                        )
                    else:
                        cc_smooth = fb.wolborska(
                            t_smooth,
                            res['parameters']['beta'], res['parameters']['N0'],
                            C0_col, Q_col, Z_col
                        )
                    fig_bt.add_trace(go.Scatter(
                        x=t_smooth, y=cc_smooth, mode='lines',
                        line=dict(color=color, width=2),
                        name=f'{name} fit'
                    ))
                except Exception:
                    pass

            fig_bt.add_trace(go.Scatter(
                x=t_bt, y=CC0, mode='markers',
                marker=dict(color='#d62728', size=10,
                            line=dict(color='white', width=1)),
                name='Experimental'
            ))
            fig_bt.update_layout(
                xaxis_title="Time (min)", yaxis_title="C/C₀",
                plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom',
                            y=1.02, xanchor='right', x=1),
                hovermode='x unified',
                margin=dict(t=20, b=10, l=10, r=10),
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0', range=[-0.05, 1.1]),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # Comparison table
            st.subheader("Model comparison")
            comp_rows = []
            for name, res in bt_results.items():
                t_r = ks.ttest(CC0, res['C_C0_predicted'],
                               n_params=len(res['parameters']), alpha=alpha)
                comp_rows.append({
                    'Model': name,
                    'R²':    round(res['r_squared'], 5),
                    'RMSE':  round(res['rmse'], 5),
                    'AIC':   round(res['aic'], 2),
                    't p-value': round(t_r['p_value'], 4),
                    'Bias?': 'Yes' if t_r['reject_H0'] else 'No',
                    **{p: round(v, 6) for p, v in res['parameters'].items()}
                })
            comp_df = pd.DataFrame(comp_rows).sort_values('AIC')
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download results CSV",
                               data=comp_df.to_csv(index=False),
                               file_name="breakthrough_results.csv",
                               mime="text/csv")

    # ── Design tools tab ──────────────────────────────────────────────
    with tab_design:
        if not st.session_state.get('kin_fb_ready', False):
            st.info("👈 Load your data in the **Data Input** tab first.")
        else:
            t_bt = st.session_state['kin_t_bt']
            CC0  = st.session_state['kin_CC0']

            st.subheader("Length of Unused Bed (LUB)")
            lub_cols = st.columns(3)
            C_break = lub_cols[0].slider("Breakthrough fraction C_b/C₀", 0.01, 0.30, 0.05)
            C_sat   = lub_cols[1].slider("Saturation fraction C_s/C₀",   0.70, 0.99, 0.95)
            Z_lub   = lub_cols[2].number_input("Bed height Z (cm)", value=10.0, min_value=0.1)
            Q_lub   = st.number_input("Flow rate Q (L/min)", value=0.01, format="%.4f",
                                      min_value=1e-6, key="Q_lub")

            try:
                fb = FixedBedKinetics()
                lub_res = fb.lub(t_bt, CC0, C0=50.0, Q=Q_lub,
                                 Z=Z_lub, C_break=C_break, C_sat=C_sat)
                lm1, lm2, lm3, lm4 = st.columns(4)
                lm1.metric("LUB (cm)",            f"{lub_res['LUB']:.3f}")
                lm2.metric("Bed utilisation",     f"{lub_res['utilization_fraction']*100:.1f}%")
                lm3.metric("t breakthrough (min)", f"{lub_res['t_breakthrough']:.1f}")
                lm4.metric("Throughput at break", f"{lub_res['throughput_at_break_L']:.2f} L")
            except Exception as e:
                st.warning(f"LUB calculation: {e}")

            st.divider()
            st.subheader("BDST — Bed Depth Service Time")
            st.markdown("""
For full BDST analysis you need breakthrough times from **multiple columns at
different bed heights**. Enter those below, or use the single-column estimation.
            """)

            bdst_mode = st.radio("BDST mode",
                                 ["Single column (current data)",
                                  "Multi-column (enter t_b per Z)"],
                                 horizontal=True)

            if bdst_mode == "Single column (current data)":
                C0_b = st.number_input("Influent C₀ (mg/L)", value=50.0,
                                       min_value=0.01, key="C0_bdst")
                Q_b  = st.number_input("Flow rate Q (L/min)", value=0.01,
                                       format="%.4f", min_value=1e-6, key="Q_bdst")
                A_b  = st.number_input("Cross-section A (cm²)", value=2.0,
                                       min_value=0.01, key="A_bdst")
                try:
                    bdst_res = fb.bdst(t_bt, CC0, C0=C0_b, Q=Q_b, A=A_b)
                    bm1, bm2, bm3 = st.columns(3)
                    bm1.metric("BDST slope",     f"{bdst_res['slope']:.4f}")
                    bm2.metric("BDST intercept", f"{bdst_res['intercept']:.2f}")
                    bm3.metric("R² (BDST line)", f"{bdst_res['r_squared']:.4f}")
                    if bdst_res['kBA_estimate']:
                        st.info(f"Estimated k_BA ≈ {bdst_res['kBA_estimate']:.4e} L/(mg·min)")
                    st.dataframe(bdst_res['service_times'], use_container_width=True,
                                 hide_index=True)
                except Exception as e:
                    st.warning(f"BDST: {e}")
            else:
                st.markdown("Enter Z (cm) and t_b (min) pairs, one per line:")
                raw_mc = st.text_area("Z, t_b per line", height=120,
                                      placeholder="5, 45\n10, 98\n15, 152\n20, 207")
                C0_mc = st.number_input("Influent C₀ (mg/L)", value=50.0, min_value=0.01)
                Q_mc  = st.number_input("Q (L/min)", value=0.01, format="%.4f")
                A_mc  = st.number_input("A (cm²)", value=2.0, min_value=0.01)
                if raw_mc.strip():
                    try:
                        rows_mc = []
                        for line in raw_mc.strip().split('\n'):
                            parts = [float(x.strip()) for x in line.split(',')]
                            if len(parts) >= 2:
                                rows_mc.append(parts[:2])
                        arr_mc  = np.array(rows_mc)
                        bdst_mc = fb.bdst_multicolumn(
                            arr_mc[:, 0], arr_mc[:, 1],
                            C0=C0_mc, Q=Q_mc, A=A_mc
                        )
                        mm1, mm2, mm3, mm4 = st.columns(4)
                        mm1.metric("N₀ (mg/L)",  f"{bdst_mc['N0']:.1f}")
                        mm2.metric("k_BA",        f"{bdst_mc['kBA']:.4e}" if bdst_mc['kBA'] else "—")
                        mm3.metric("R²",          f"{bdst_mc['r_squared']:.4f}")
                        mm4.metric("Linear vel.", f"{bdst_mc['linear_velocity_u']:.4f} cm/min")

                        # BDST plot
                        Z_arr = arr_mc[:, 0]
                        t_arr = arr_mc[:, 1]
                        Z_fit = np.linspace(0, Z_arr.max() * 1.1, 100)
                        t_fit = bdst_mc['slope'] * Z_fit + bdst_mc['intercept']
                        fig_bdst = go.Figure()
                        fig_bdst.add_trace(go.Scatter(
                            x=Z_arr, y=t_arr, mode='markers',
                            marker=dict(color='#d62728', size=10), name='Data'))
                        fig_bdst.add_trace(go.Scatter(
                            x=Z_fit, y=t_fit, mode='lines',
                            line=dict(color='#534AB7', width=2), name='BDST line'))
                        fig_bdst.update_layout(
                            xaxis_title="Bed depth Z (cm)",
                            yaxis_title="Service time t_b (min)",
                            plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=20, b=10, l=10, r=10),
                            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                        )
                        st.plotly_chart(fig_bdst, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Multi-column BDST: {e}")


# ═════════════════════════════════════════════════════════════════════
#  ROUTER
# ═════════════════════════════════════════════════════════════════════

page = st.session_state.get('page', 'home')

if page == 'home':
    page_home()
elif page == 'equilibrium':
    page_equilibrium()
elif page == 'kinetics':
    page_kinetics()
