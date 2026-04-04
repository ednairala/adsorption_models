"""
app.py  —  Streamlit web interface for Adsorption Isotherms v6

HOW TO RUN:
    pip install streamlit plotly pandas scipy numpy
    streamlit run app.py

HOW STREAMLIT WORKS (quick orientation):
    - Every time a user interacts with a widget (slider, button, upload),
      Streamlit re-runs this entire file from top to bottom.
    - st.session_state is a dictionary that PERSISTS between re-runs.
      We use it to store fitted results so they don't disappear on rerun.
    - st.sidebar.*  puts widgets in the left sidebar.
    - st.tabs()     creates tab panels.
    - @st.cache_data tells Streamlit to remember a function's output
      so it doesn't recompute when inputs haven't changed.
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import our upgraded library (must be in the same folder)
from adsorption_isotherms_v6 import AdsorptionIsotherms, validate_and_clean


# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adsorption Isotherm Fitting",
    page_icon="🔬",
    layout="wide",            # use full browser width
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────────────────
#  CUSTOM CSS  (inject a small style block to polish the look)
# ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Slightly tighten the main content area */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Give metric cards a soft border */
    [data-testid="metric-container"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    /* Style the download button */
    .stDownloadButton > button {
        width: 100%;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

# @st.cache_data means: if the same Ce_tuple and qe_tuple and model_name
# are passed again, return the cached result instead of re-fitting.
# The underscore in _model tells Streamlit not to hash that argument
# (the class instance is not hashable, but it doesn't change).
@st.cache_data
def cached_fit(_model, model_name, Ce_tuple, qe_tuple):
    """Fit one model; cached so it doesn't re-run on every widget interaction."""
    Ce = np.array(Ce_tuple)
    qe = np.array(qe_tuple)
    return _model.fit(model_name, Ce, qe)


def build_plotly_figure(model_name, Ce_exp, qe_exp, result,
                         Ce_unit, qe_unit, show_ci):
    """Build an interactive Plotly figure for a fitted isotherm.

    Plotly creates interactive charts — users can zoom, pan, hover
    over points to see exact values, and export as PNG.

    Args:
        model_name (str): Name of the fitted model.
        Ce_exp (ndarray): Experimental Ce values.
        qe_exp (ndarray): Experimental qe values.
        result (dict): Output from AdsorptionIsotherms.fit().
        Ce_unit (str): Unit string for x-axis label.
        qe_unit (str): Unit string for y-axis label.
        show_ci (bool): Whether to draw the 95% confidence band.

    Returns:
        plotly.graph_objects.Figure
    """
    model = AdsorptionIsotherms()
    Ce_smooth = np.linspace(Ce_exp.min(), Ce_exp.max(), 300)
    qe_smooth = model.predict(model_name, Ce_smooth, result['parameters'])

    Ce_label = f"Ce ({Ce_unit})" if Ce_unit else "Ce (Equilibrium Concentration)"
    qe_label = f"qe ({qe_unit})" if qe_unit else "qe (Amount Adsorbed)"

    fig = go.Figure()

    # ── Confidence band ──────────────────────────────────────────
    if show_ci and result.get('covariance') is not None:
        pcov   = result['covariance']
        pnames = list(result['parameters'].keys())
        pvals  = np.array([result['parameters'][k] for k in pnames])
        model_fn = getattr(model, model_name)
        n_params = len(pvals)

        # Numerical Jacobian (same technique as in the Python library)
        eps = 1e-6 * np.abs(pvals) + 1e-12
        J   = np.zeros((len(Ce_smooth), n_params))
        for i in range(n_params):
            p_plus  = pvals.copy(); p_plus[i]  += eps[i]
            p_minus = pvals.copy(); p_minus[i] -= eps[i]
            J[:, i] = (model_fn(Ce_smooth, *p_plus) -
                        model_fn(Ce_smooth, *p_minus)) / (2 * eps[i])

        var_pred = np.einsum('ij,jk,ik->i', J, pcov, J)
        std_pred = np.sqrt(np.maximum(var_pred, 0))
        ci_band  = 1.96 * std_pred

        # Plotly shaded area: upper + lower boundaries traced as one shape
        fig.add_trace(go.Scatter(
            x=np.concatenate([Ce_smooth, Ce_smooth[::-1]]),
            y=np.concatenate([qe_smooth + ci_band, (qe_smooth - ci_band)[::-1]]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name='95% CI'
        ))

    # ── Fitted curve ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=Ce_smooth, y=qe_smooth,
        mode='lines',
        line=dict(color='#1f77b4', width=2.5),
        name=f'{model_name.capitalize()} fit'
    ))

    # ── Experimental data points ─────────────────────────────────
    fig.add_trace(go.Scatter(
        x=Ce_exp, y=qe_exp,
        mode='markers',
        marker=dict(color='#d62728', size=10, symbol='circle',
                    line=dict(color='white', width=1)),
        name='Experimental data'
    ))

    fig.update_layout(
        xaxis_title=Ce_label,
        yaxis_title=qe_label,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False),
    )
    return fig


def results_to_dataframe(all_results):
    """Convert compare_models output to a tidy pandas DataFrame.

    Args:
        all_results (dict): Output from compare_models().

    Returns:
        pd.DataFrame: One row per model, columns for each metric.
    """
    rows = []
    for name, res in all_results.items():
        row = {
            'Model': name.replace('_', ' ').title(),
            'R²': round(res['r_squared'], 5),
            'RMSE': round(res['rmse'], 5),
        }
        # Add fitted parameters as columns
        for param, val in res['parameters'].items():
            row[param] = round(val, 5)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_report_csv(Ce_exp, qe_exp, all_results, Ce_unit, qe_unit):
    """Generate a CSV report string for download.

    Writes two sections:
      1. Model comparison table (sorted by R²)
      2. The original experimental data

    Args:
        Ce_exp, qe_exp: Experimental arrays.
        all_results (dict): From compare_models().
        Ce_unit, qe_unit (str): Unit labels.

    Returns:
        str: CSV content as a string.
    """
    lines = []
    lines.append("Adsorption Isotherm Fitting — Results Report")
    lines.append("")

    lines.append("MODEL COMPARISON")
    lines.append(f"Model,R²,RMSE,Parameters")
    for name, res in all_results.items():
        param_str = " | ".join(
            f"{k}={v:.5f} ±{res['confidence_95'].get(k,0):.5f}"
            for k, v in res['parameters'].items()
        )
        lines.append(f"{name},{res['r_squared']:.5f},{res['rmse']:.5f},{param_str}")

    lines.append("")
    lines.append("EXPERIMENTAL DATA")
    lines.append(f"Ce ({Ce_unit}),qe ({qe_unit})")
    for ce, qe in zip(Ce_exp, qe_exp):
        lines.append(f"{ce},{qe}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Adsorption Isotherms")
    st.caption("open-source fitting tool")

    st.divider()

    # ── Unit labels ──────────────────────────────────────────────
    st.subheader("Axis units")
    Ce_unit = st.text_input("Ce unit", value="mg/L",
                             help="Equilibrium concentration unit. "
                                  "Examples: mg/L, mmol/L, µg/L")
    qe_unit = st.text_input("qe unit", value="mg/g",
                             help="Amount adsorbed unit. "
                                  "Examples: mg/g, mmol/g, µmol/g")

    st.divider()

    # ── Model selection ──────────────────────────────────────────
    st.subheader("Models to fit")
    all_model_names = list(AdsorptionIsotherms().models.keys())

    # st.multiselect lets users pick which models to run
    selected_models = st.multiselect(
        "Select models",
        options=all_model_names,
        default=all_model_names,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    st.divider()

    # ── Plot options ─────────────────────────────────────────────
    st.subheader("Plot options")
    show_ci = st.toggle("Show 95% confidence band", value=True,
                         help="Shaded area around the fit curve showing "
                              "parameter uncertainty")

    st.divider()
    st.markdown("""
    **About this tool**

    Fits experimental adsorption data to common isotherm models
    using nonlinear least squares (scipy curve_fit).

    [View source on GitHub](#) · [Report an issue](#)
    """)


# ─────────────────────────────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────

st.title("Adsorption Isotherm Fitting Tool")
st.markdown(
    "Upload your experimental data, select the isotherm models to test, "
    "and get publication-ready statistics and plots."
)

# ─── Tabs ────────────────────────────────────────────────────────────
# st.tabs() creates clickable tab panels — users switch between them
# without navigating away from the page.
tab_about, tab_data, tab_results, tab_linearized = st.tabs([
    "ℹ️ About & Help",
    "📂 Data Input",
    "📊 Fitting & Results",
    "📈 Linearized Forms"
])

# ═════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT & HELP
# ═════════════════════════════════════════════════════════════════════

with tab_about:

    st.subheader("About this tool")
    st.markdown("""
                This web application provides nonlinear fitting of experimental adsorption data to common isotherm models 
                used in environmental, chemical, and materials science research.
    """)
    st.markdown("**Implemented models**")

    models_info = [
        {
            "name": "Henry",
            "equation": "qe = KH·Ce",
            "application": "Dilute solutions, linear range",
            "description": """
    **Henry's Law** assumes adsorption is proportional to concentration — the simplest possible model.
    Valid only at very low concentrations where the surface is far from saturated.

    - **KH**: Henry constant (slope of the linear relationship)
    - Best used as a baseline or for initial screening
    - Fails at higher concentrations where surface sites become limited
            """
        },
        {
            "name": "Langmuir",
            "equation": "qe = (qmax·KL·Ce)/(1+KL·Ce)",
            "application": "Monolayer, homogeneous surface",
            "description": """
    **Langmuir** assumes a finite number of identical, equivalent adsorption sites.
    Once all sites are filled (qmax), no more adsorption occurs — classic monolayer model.

    - **qmax**: Maximum adsorption capacity (mg/g)
    - **KL**: Langmuir affinity constant (L/mg) — higher = stronger binding
    - Assumes no interaction between adsorbed molecules
    - Most widely used model in literature
            """
        },
        {
            "name": "Freundlich",
            "equation": "qe = KF·Ce^(1/n)",
            "application": "Heterogeneous surface",
            "description": """
    **Freundlich** is an empirical model for heterogeneous surfaces with non-uniform energy sites.
    Unlike Langmuir, it has no saturation plateau — adsorption keeps increasing with concentration.

    - **KF**: Adsorption capacity constant
    - **n**: Heterogeneity factor — if n > 1, adsorption is favorable; n = 1 → linear (Henry)
    - 1/n (slope of ln-ln plot) indicates adsorption intensity
            """
        },
        {
            "name": "Temkin",
            "equation": "qe = B·ln(A·Ce)",
            "application": "Adsorbate–adsorbent interactions",
            "description": """
    **Temkin** accounts for adsorbate–adsorbent interactions by assuming adsorption heat
    decreases linearly with surface coverage (rather than being constant as in Langmuir).

    - **A**: Temkin equilibrium binding constant (L/g)
    - **B**: Constant related to the heat of adsorption (J/mol)
    - Useful for chemisorption systems where surface interactions matter
            """
        },
        {
            "name": "BET",
            "equation": "multilayer formula",
            "application": "Multilayer adsorption",
            "description": """
    **Brunauer–Emmett–Teller (BET)** extends Langmuir to allow multiple adsorption layers.
    Each layer acts as a new surface for the next, leading to much higher capacities.

    - **qm**: Monolayer capacity
    - **C**: BET energy constant (related to first-layer adsorption energy)
    - **Cs**: Saturation concentration — x = Ce/Cs must be < 1
    - Standard model for gas-phase surface area measurements (N₂ adsorption)
            """
        },
        {
            "name": "Dubinin–Radushkevich",
            "equation": "qe = qs·exp(−K·ε²)",
            "application": "Pore-filling mechanism",
            "description": """
    **Dubinin–Radushkevich (D-R)** describes adsorption into micropores via a pore-filling mechanism.
    Uses the Polanyi adsorption potential ε = ln(1 + 1/Ce).

    - **qs**: Theoretical saturation capacity
    - **K**: Constant related to mean free energy of adsorption
    - Mean adsorption energy E = 1/√(2K) — if E < 8 kJ/mol → physisorption; E > 8 → chemisorption
            """
        },
        {
            "name": "Redlich–Peterson",
            "equation": "qe = KR·Ce/(1+aR·Ce^g)",
            "application": "Hybrid L–F model",
            "description": """
    **Redlich–Peterson** is a three-parameter hybrid that combines features of Langmuir and Freundlich.
    The exponent g (0–1) controls which model it approaches.

    - **KR, aR**: Redlich–Peterson constants
    - **g**: Exponent — g = 1 → reduces to Langmuir; aR → 0 → reduces to Henry
    - More flexible than 2-parameter models, useful when neither L nor F fits well
    - Requires more data points to avoid overfitting
            """
        },
    ]

    # Table header
    header_cols = st.columns([1.2, 1.8, 2, 0.6])
    header_cols[0].markdown("**Model**")
    header_cols[1].markdown("**Equation**")
    header_cols[2].markdown("**Application**")
    header_cols[3].markdown("**Info**")
    st.divider()

    for m in models_info:
        cols = st.columns([1.2, 1.8, 2, 0.6])
        cols[0].markdown(m["name"])
        cols[1].markdown(f"`{m['equation']}`")
        cols[2].markdown(m["application"])
        with cols[3].popover("💬"):
            st.markdown(f"### {m['name']}")
            st.markdown(m["description"])
        st.divider()

    st.markdown("""

        **Fitting method**

        Nonlinear least squares using `scipy.optimize.curve_fit`.
        Parameters are reported with 95% confidence intervals derived
        from the covariance matrix (Jacobian propagation).

        **Data format**

        CSV file with two columns: `Ce` (equilibrium concentration)
        and `qe` (amount adsorbed). European decimal format (comma
        as decimal separator) is automatically detected and converted.

        **Version history**

        - v6: BET registered, Temkin/BET log-zero guards, confidence
        intervals, linearized forms, unit labels, data validation.
        - v5: Original interactive CLI version.

        **License:** MIT — free to use, modify, and distribute.
        """)

    st.divider()
    st.subheader("Quick start guide")

    with st.expander("How do I prepare my CSV file?"):
            st.markdown("""
            Your CSV needs two columns in this order:

            ```
            Ce,qe
            0.10,0.90
            0.50,3.20
            1.00,4.80
            2.00,6.50
            5.00,8.20
            10.00,9.00
            20.00,9.50
            ```

            - Column names can be anything — only column order matters.
            - Minimum **3 data points** are required (more = better fits).
            - Empty cells and rows with zero values are automatically removed.
            - European format (e.g. `3,14` for 3.14) is supported.
            """)

    with st.expander("What does R² mean?"):
            st.markdown("""
            **R² (coefficient of determination)** measures how well the
            model fits your data. It ranges from 0 to 1:

            - R² = 1.0 → perfect fit
            - R² > 0.99 → excellent fit (publication-ready)
            - R² > 0.95 → good fit
            - R² < 0.90 → model may not be appropriate

            Use the **RMSE** (root mean square error) alongside R² — a low
            RMSE means the average prediction error is small in absolute terms.
            """)

    with st.expander("What does the 95% confidence band mean?"):
            st.markdown("""
            The shaded area around the fitted curve represents the **95%
            confidence interval** for the model prediction at each Ce value.

            It reflects the uncertainty in the fitted parameters. A narrow
            band means the parameters are well-constrained by the data.
            A wide band suggests you may need more data points, especially
            at extreme concentration values.
            """)

    with st.expander("Should I use linearized or nonlinear fitting?"):
            st.markdown("""
            **Always prefer the nonlinear fitting** (Results tab) for
            parameter estimation. It minimizes the true least-squares
            criterion on the original data.

            Linearization (e.g., Ce/qe vs Ce for Langmuir) was historically
            used because it turned a nonlinear problem into a simple ruler-
            and-graph problem. However, it distorts the error structure:
            errors that are uniform in qe become non-uniform after the
            transformation. This leads to biased parameter estimates.

            The Linearized Forms tab is included for:
            - Cross-checking with older published literature
            - Visual verification that your data follows a particular model
            """)

# ═════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA INPUT
# ═════════════════════════════════════════════════════════════════════

with tab_data:

    st.subheader("Load experimental data")

    # st.radio creates mutually exclusive options (like radio buttons)
    input_method = st.radio(
        "Choose input method",
        ["Upload CSV file", "Paste data manually", "Use example data"],
        horizontal=True
    )

    Ce_exp = None
    qe_exp = None

    # ── Option A: CSV upload ──────────────────────────────────────
    if input_method == "Upload CSV file":
        st.markdown("""
        Upload a CSV with **two columns**: `Ce` and `qe`.
        The first row should be the header. Example:
        ```
        Ce,qe
        0.5,2.1
        1.0,3.5
        ```
        European decimal format (e.g. `3,14`) is supported.
        """)

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv", "txt"],
            help="File must have Ce in column 1 and qe in column 2."
        )

        if uploaded_file is not None:
            try:
                # pd.read_csv reads the file into a DataFrame
                # (think of it as a spreadsheet in Python)
                df = pd.read_csv(uploaded_file)

                # Accept any column names for the first two columns
                df.columns = [c.strip() for c in df.columns]
                col1, col2 = df.columns[0], df.columns[1]

                Ce_raw = pd.to_numeric(
                    df[col1].astype(str).str.replace(',', '.'), errors='coerce'
                ).values
                qe_raw = pd.to_numeric(
                    df[col2].astype(str).str.replace(',', '.'), errors='coerce'
                ).values

                Ce_exp, qe_exp = validate_and_clean(Ce_raw, qe_raw)

                st.success(f"✓ Loaded {len(Ce_exp)} valid data points.")

                # Show the data in a table so users can verify it
                preview_df = pd.DataFrame({
                    f"Ce ({Ce_unit})": Ce_exp,
                    f"qe ({qe_unit})": qe_exp
                })
                st.dataframe(preview_df, use_container_width=True)

            except Exception as e:
                st.error(f"Could not read file: {e}")

    # ── Option B: Manual paste ────────────────────────────────────
    elif input_method == "Paste data manually":
        st.markdown(
            "Enter one data point per line: `Ce, qe`  "
            "(comma-separated, European decimals OK)"
        )

        # st.text_area is a multi-line text input box
        raw_text = st.text_area(
            "Paste your data here",
            height=250,
            placeholder="0.1, 0.9\n0.5, 3.2\n1.0, 4.8\n2.0, 6.5\n...",
        )

        if raw_text.strip():
            try:
                Ce_list, qe_list = [], []
                for line_num, line in enumerate(raw_text.strip().split('\n'), 1):
                    line = line.strip()
                    if not line:
                        continue
                    # Handle both comma and semicolon separators
                    sep = ';' if ';' in line else ','
                    parts = line.split(sep)
                    if len(parts) < 2:
                        st.warning(f"Line {line_num} skipped (expected 2 values).")
                        continue
                    # Replace European decimal comma in each value
                    def to_float(s):
                        s = s.strip()
                        # If there are two commas, treat first as thousands sep
                        return float(s.replace('.', '').replace(',', '.'))

                    Ce_list.append(float(parts[0].strip().replace(',', '.')))
                    qe_list.append(float(parts[1].strip().replace(',', '.')))

                Ce_exp, qe_exp = validate_and_clean(Ce_list, qe_list)

                st.success(f"✓ Parsed {len(Ce_exp)} valid data points.")
                preview_df = pd.DataFrame({
                    f"Ce ({Ce_unit})": Ce_exp,
                    f"qe ({qe_unit})": qe_exp
                })
                st.dataframe(preview_df, use_container_width=True)

            except Exception as e:
                st.error(f"Parsing error: {e}")

    # ── Option C: Example data ────────────────────────────────────
    else:
        st.info("Using built-in Langmuir-like example dataset.")
        Ce_exp = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        qe_exp = np.array([0.9, 3.2, 4.8, 6.5, 8.2,  9.0,  9.5])

        preview_df = pd.DataFrame({
            f"Ce ({Ce_unit})": Ce_exp,
            f"qe ({qe_unit})": qe_exp
        })
        st.dataframe(preview_df, use_container_width=True)

    # ── Store in session_state so other tabs can access it ────────
    # session_state is like a backpack that carries data across re-runs
    if Ce_exp is not None and qe_exp is not None:
        st.session_state['Ce_exp'] = Ce_exp
        st.session_state['qe_exp'] = qe_exp
        st.session_state['ready'] = True
    else:
        st.session_state['ready'] = False


# ═════════════════════════════════════════════════════════════════════
#  TAB 2 — FITTING & RESULTS
# ═════════════════════════════════════════════════════════════════════

with tab_results:

    # Check if data was loaded in Tab 1
    if not st.session_state.get('ready', False):
        st.info("👈 Please load your data in the **Data Input** tab first.")
    
    else:  # ✅ Wrap the rest of the tab content in an else block
            

        Ce_exp = st.session_state['Ce_exp']
        qe_exp = st.session_state['qe_exp']

        if not selected_models:
            st.warning("Select at least one model in the sidebar.")
            st.stop()

        # ── Run fitting ───────────────────────────────────────────────
        # st.spinner shows a loading animation while fitting runs
        with st.spinner("Fitting models…"):
            model = AdsorptionIsotherms()
            all_results = {}
            errors = {}

            for mname in selected_models:
                try:
                    # We cast to tuple so @st.cache_data can hash the arrays
                    res = cached_fit(model, mname,
                                    tuple(Ce_exp), tuple(qe_exp))
                    all_results[mname] = res
                except Exception as e:
                    errors[mname] = str(e)

        # Show any fitting failures
        for mname, err in errors.items():
            st.warning(f"**{mname}** could not be fitted: {err}")

        if not all_results:
            st.error("No models could be fitted. Check your data.")
            st.stop()

        # Sort by R² descending
        all_results = dict(
            sorted(all_results.items(), key=lambda x: x[1]['r_squared'], reverse=True)
        )
        best_model = list(all_results.keys())[0]

        # ── Summary metrics ───────────────────────────────────────────
        # st.columns splits the page into side-by-side columns
        col1, col2, col3 = st.columns(3)
        best_res = all_results[best_model]
        col1.metric("Best model",  best_model.replace('_', ' ').title())
        col2.metric("R²",          f"{best_res['r_squared']:.5f}")
        col3.metric("RMSE",        f"{best_res['rmse']:.5f}")

        st.divider()

        # ── Model selector for individual plot ────────────────────────
        st.subheader("Isotherm plot")

        # st.selectbox is a dropdown; default to best model
        model_to_plot = st.selectbox(
            "Select model to plot",
            options=list(all_results.keys()),
            index=0,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        fig = build_plotly_figure(
            model_to_plot, Ce_exp, qe_exp,
            all_results[model_to_plot],
            Ce_unit, qe_unit, show_ci
        )
        # st.plotly_chart renders an interactive Plotly figure
        st.plotly_chart(fig, use_container_width=True)

        # ── Parameter table ───────────────────────────────────────────
        st.subheader("Fitted parameters")

        selected_res = all_results[model_to_plot]
        param_rows = []
        for pname, pval in selected_res['parameters'].items():
            ci = selected_res['confidence_95'].get(pname, np.nan)
            param_rows.append({
                'Parameter': pname,
                'Value': round(pval, 6),
                '± 95% CI': round(ci, 6)
            })
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True,
                    hide_index=True)

        st.divider()

        # ── Full model comparison table ───────────────────────────────
        st.subheader("Model comparison")
        comparison_df = results_to_dataframe(all_results)

        # Highlight the best model row
        # st.dataframe with column_config lets us customise column display
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'R²': st.column_config.ProgressColumn(
                    'R²', min_value=0, max_value=1, format="%.5f"
                )
            }
        )

        # ── Downloads ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Export results")

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            csv_report = generate_report_csv(
                Ce_exp, qe_exp, all_results, Ce_unit, qe_unit
            )
            st.download_button(
                label="⬇ Download full report (CSV)",
                data=csv_report,
                file_name="isotherm_results.csv",
                mime="text/csv"
            )

        with dl_col2:
            # Export just the comparison table
            csv_table = comparison_df.to_csv(index=False)
            st.download_button(
                label="⬇ Download comparison table (CSV)",
                data=csv_table,
                file_name="model_comparison.csv",
                mime="text/csv"
            )


# ═════════════════════════════════════════════════════════════════════
#  TAB 3 — LINEARIZED FORMS
# ═════════════════════════════════════════════════════════════════════

with tab_linearized:

    # Check if data was loaded in Tab 1
    if not st.session_state.get('ready', False):
        st.info("👈 Please load your data in the **Data Input** tab first.")
    else:    

        Ce_exp = st.session_state['Ce_exp']
        qe_exp = st.session_state['qe_exp']

        st.subheader("Linearized isotherm forms")
        st.markdown("""
        Many classical studies use **linearized** forms of isotherm equations
        to estimate parameters graphically. These are provided for comparison
        with the nonlinear fitting in the Results tab.

        > **Note:** Nonlinear fitting (Results tab) is statistically more
        > rigorous. Linearization distorts the error structure. These plots
        > are here for reference and cross-checking with published literature.
        """)

        lin_col1, lin_col2 = st.columns(2)

        model_obj = AdsorptionIsotherms()

        # ── Langmuir linearization ────────────────────────────────────
        with lin_col1:
            st.markdown("**Langmuir: Ce/qe vs Ce**")
            try:
                lang_lin = model_obj.linearize_langmuir(Ce_exp, qe_exp)

                x_fit = np.linspace(Ce_exp.min(), Ce_exp.max(), 100)
                y_fit = lang_lin['slope'] * x_fit + lang_lin['intercept']

                fig_lang = go.Figure()
                fig_lang.add_trace(go.Scatter(
                    x=lang_lin['x'], y=lang_lin['y'],
                    mode='markers',
                    marker=dict(color='#d62728', size=10),
                    name='Data'
                ))
                fig_lang.add_trace(go.Scatter(
                    x=x_fit, y=y_fit,
                    mode='lines',
                    line=dict(color='#1f77b4', width=2),
                    name='Linear fit'
                ))
                fig_lang.update_layout(
                    xaxis_title=f"Ce ({Ce_unit})",
                    yaxis_title=f"Ce/qe ({Ce_unit}/{qe_unit})",
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=10)
                )
                st.plotly_chart(fig_lang, use_container_width=True)

                st.markdown(f"""
                | Parameter | Linearized | Nonlinear (for reference) |
                |-----------|-----------|--------------------------|
                | qmax      | {lang_lin['qmax']:.4f} {qe_unit} | — |
                | KL        | {lang_lin['KL']:.4f} L/{Ce_unit} | — |
                """)
            except Exception as e:
                st.error(f"Langmuir linearization failed: {e}")

        # ── Freundlich linearization ──────────────────────────────────
        with lin_col2:
            st.markdown("**Freundlich: ln(qe) vs ln(Ce)**")
            try:
                frnd_lin = model_obj.linearize_freundlich(Ce_exp, qe_exp)

                x_fit = np.linspace(frnd_lin['x'].min(), frnd_lin['x'].max(), 100)
                y_fit = frnd_lin['slope'] * x_fit + frnd_lin['intercept']

                fig_frnd = go.Figure()
                fig_frnd.add_trace(go.Scatter(
                    x=frnd_lin['x'], y=frnd_lin['y'],
                    mode='markers',
                    marker=dict(color='#d62728', size=10),
                    name='Data'
                ))
                fig_frnd.add_trace(go.Scatter(
                    x=x_fit, y=y_fit,
                    mode='lines',
                    line=dict(color='#2ca02c', width=2),
                    name='Linear fit'
                ))
                fig_frnd.update_layout(
                    xaxis_title=f"ln(Ce)",
                    yaxis_title=f"ln(qe)",
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=10)
                )
                st.plotly_chart(fig_frnd, use_container_width=True)

                st.markdown(f"""
                | Parameter | Linearized | Description |
                |-----------|-----------|-------------|
                | KF        | {frnd_lin['KF']:.4f} | Freundlich constant |
                | n         | {frnd_lin['n']:.4f}  | Heterogeneity factor |
                | 1/n (slope) | {frnd_lin['slope']:.4f} | Adsorption intensity |
                """)
            except Exception as e:
                st.error(f"Freundlich linearization failed: {e}")


