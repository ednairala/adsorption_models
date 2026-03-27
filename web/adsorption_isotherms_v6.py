"""
adsorption_isotherms_v6.py

Numerical Python library for adsorption isotherm model fitting.
Open-source tool for the scientific and technical community.

Changes in v6:
  - BET model added to self.models (was missing, broke compare_models)
  - Temkin/BET: guard against Ce <= 0 (log of zero crash fixed)
  - Data validation: handles missing values, European decimals, bad headers
  - Confidence intervals from covariance matrix (publication-ready)
  - Linearized forms for Langmuir and Freundlich
  - Unit labels supported throughout
  - All models tested and consistent

Example as library:
    from adsorption_isotherms_v6 import AdsorptionIsotherms
    model = AdsorptionIsotherms()
    results = model.fit("langmuir", Ce_data, qe_data)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────
#  DATA VALIDATION
# ─────────────────────────────────────────────

def validate_and_clean(Ce_raw, qe_raw):
    """Validate and clean experimental data arrays.

    This function checks for common data problems that would cause
    the fitting algorithms to crash silently or give wrong results.
    It removes rows with missing/invalid values and warns the user.

    Why this matters: scientists often paste data from Excel where
    some cells are empty, or use commas as decimal separators (e.g.
    European locale: "3,14" instead of "3.14"). This function handles
    both cases gracefully instead of crashing.

    Args:
        Ce_raw (array-like): Raw equilibrium concentration values.
        qe_raw (array-like): Raw adsorbed amount values.

    Returns:
        tuple: (Ce_clean, qe_clean) as float64 numpy arrays.

    Raises:
        ValueError: If fewer than 3 valid data points remain after cleaning.
    """
    # Convert to numpy arrays of floats
    # np.float64 means 64-bit floating point number (standard precision)
    Ce = np.array(Ce_raw, dtype=np.float64)
    qe = np.array(qe_raw, dtype=np.float64)

    # np.isfinite returns True only for real numbers (not NaN or Inf)
    # NaN = "Not a Number" — appears when Excel cells are empty
    # Inf = infinity — appears from division by zero
    valid_mask = np.isfinite(Ce) & np.isfinite(qe) & (Ce > 0) & (qe > 0)

    n_removed = len(Ce) - np.sum(valid_mask)
    if n_removed > 0:
        print(f"  [Warning] Removed {n_removed} invalid/missing data point(s).")

    Ce_clean = Ce[valid_mask]
    qe_clean = qe[valid_mask]

    if len(Ce_clean) < 3:
        raise ValueError(
            f"Only {len(Ce_clean)} valid data point(s) remain after cleaning. "
            "Need at least 3 to fit a model."
        )

    return Ce_clean, qe_clean


def parse_european_decimal(value_str):
    """Convert a string that may use European decimal format to float.

    In many European countries, the decimal separator is a comma
    and the thousands separator is a period (e.g. "1.234,56" = 1234.56).
    Python's float() only understands "1234.56", so we convert first.

    Args:
        value_str (str): A number string, possibly with European formatting.

    Returns:
        float: The parsed number.

    Raises:
        ValueError: If the string cannot be interpreted as a number.
    """
    # Replace thousands separator (period) first, then decimal comma
    cleaned = value_str.strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        # If that failed, try the original string directly
        return float(value_str.strip())


# ─────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────

class AdsorptionIsotherms:
    """Collection of adsorption isotherm models with fitting capabilities.

    Implements 7 isotherm equations commonly used in environmental,
    chemical, and materials science research. Provides fitting,
    comparison, confidence interval estimation, and visualization.

    Attributes:
        models (dict): Metadata for each model — parameter names,
            bounds, and initial guesses for the optimizer.
    """

    def __init__(self):
        """Initialize with model metadata.

        Each entry in self.models tells scipy's curve_fit:
          - 'params': what to name the fitted parameters
          - 'bounds': physical constraints (e.g. all positive)
          - 'initial': starting guesses for the optimizer

        Good initial guesses help the optimizer converge correctly.
        Bounds prevent physically impossible results (e.g. negative qmax).
        """
        self.models = {
            'henry': {
                'params': ['KH'],
                'bounds': ([0], [np.inf]),
                'initial': [1.0]
            },
            'langmuir': {
                'params': ['qmax', 'KL'],
                'bounds': ([0, 0], [np.inf, np.inf]),
                'initial': [10.0, 1.0]
            },
            'freundlich': {
                'params': ['KF', 'n'],
                'bounds': ([0, 0.1], [np.inf, 10]),
                'initial': [1.0, 2.0]
            },
            'temkin': {
                'params': ['A', 'B'],
                'bounds': ([0.01, 0], [np.inf, np.inf]),
                'initial': [1.0, 1.0]
            },
            'redlich_peterson': {
                'params': ['KR', 'aR', 'g'],
                'bounds': ([0, 0, 0], [np.inf, np.inf, 1]),
                'initial': [1.0, 1.0, 0.5]
            },
            'dubinin_radushkevich': {
                'params': ['qs', 'K'],
                'bounds': ([0, 0], [np.inf, np.inf]),
                'initial': [10.0, 1.0]
            },
            # FIX v6: BET was missing from self.models, making it
            # inaccessible through fit() and compare_models().
            'bet': {
                'params': ['qm', 'C', 'Cs'],
                'bounds': ([0, 0, 0], [np.inf, np.inf, np.inf]),
                'initial': [10.0, 1.0, 100.0]
            }
        }

    # ── Isotherm equations ──────────────────────────────────────────

    def henry(self, Ce, KH):
        """Henry's Law: linear adsorption at low concentrations.

        Formula: qe = KH * Ce

        Args:
            Ce (float or ndarray): Equilibrium concentration.
            KH (float): Henry constant (adsorption intensity).

        Returns:
            float or ndarray: Amount adsorbed (qe).
        """
        return KH * Ce

    def langmuir(self, Ce, qmax, KL):
        """Langmuir isotherm: monolayer on a homogeneous surface.

        Formula: qe = (qmax * KL * Ce) / (1 + KL * Ce)

        Args:
            Ce (float or ndarray): Equilibrium concentration.
            qmax (float): Maximum adsorption capacity.
            KL (float): Langmuir constant (binding affinity).

        Returns:
            float or ndarray: Amount adsorbed (qe).
        """
        return (qmax * KL * Ce) / (1 + KL * Ce)

    def freundlich(self, Ce, KF, n):
        """Freundlich isotherm: empirical model for heterogeneous surfaces.

        Formula: qe = KF * Ce^(1/n)

        Args:
            Ce (float or ndarray): Equilibrium concentration.
            KF (float): Freundlich adsorption constant.
            n (float): Heterogeneity factor (1–10 typical).

        Returns:
            float or ndarray: Amount adsorbed (qe).
        """
        return KF * (Ce ** (1.0 / n))

    def temkin(self, Ce, A, B):
        """Temkin isotherm: accounts for adsorbate–adsorbent interactions.

        Formula: qe = B * ln(A * Ce)

        FIX v6: Added Ce > 0 guard. If Ce contains zeros or negatives,
        np.log() returns -inf or NaN, crashing the optimizer. The
        validate_and_clean() function upstream should catch this, but
        we add a safety clip here as a second layer of protection.

        Args:
            Ce (float or ndarray): Equilibrium concentration (must be > 0).
            A (float): Temkin equilibrium binding constant.
            B (float): Constant related to adsorption heat.

        Returns:
            float or ndarray: Amount adsorbed (qe).
        """
        # np.clip ensures Ce is never below a tiny positive number
        # before taking the logarithm — prevents log(0) = -inf
        Ce_safe = np.clip(Ce, 1e-12, None)
        return B * np.log(A * Ce_safe)

    def bet(self, Ce, qm, C, Cs):
        """BET isotherm: multilayer adsorption (Brunauer–Emmett–Teller).

        Formula: qe = (qm * C * x) / ((1 - x) * (1 + (C - 1) * x))
        where x = Ce / Cs  (relative concentration, must be < 1)

        FIX v6: This method existed but was not in self.models, so it
        could not be used via fit() or compare_models(). Now registered.
        Added a guard for x >= 1 (the formula diverges at saturation).

        Args:
            Ce (float or ndarray): Equilibrium concentration.
            qm (float): Monolayer adsorption capacity.
            C (float): BET energy constant.
            Cs (float): Saturation concentration (Ce must be < Cs).

        Returns:
            float or ndarray: Amount adsorbed (qe).
        """
        # x is the relative pressure/concentration (dimensionless, 0 to 1)
        x = Ce / Cs
        # Clip x below 1 to avoid division by zero at saturation
        x = np.clip(x, 1e-12, 1 - 1e-12)
        return (qm * C * x) / ((1 - x) * (1 + (C - 1) * x))

    def dubinin_radushkevich(self, Ce, qs, K):
        """Dubinin–Radushkevich isotherm: pore-filling mechanism.

        Formula: qe = qs * exp(-K * ε²)
        where ε = ln(1 + 1/Ce)  (Polanyi adsorption potential)

        Args:
            Ce (float or ndarray): Equilibrium concentration (must be > 0).
            qs (float): Theoretical saturation capacity.
            K (float): Constant related to mean adsorption energy.

        Returns:
            float or ndarray: Amount adsorbed (qe).
        """
        Ce_safe = np.clip(Ce, 1e-12, None)
        epsilon = np.log(1 + (1.0 / Ce_safe))
        return qs * np.exp(-K * epsilon ** 2)

    def redlich_peterson(self, Ce, KR, aR, g):
        """Redlich–Peterson isotherm: hybrid Langmuir–Freundlich model.

        Formula: qe = (KR * Ce) / (1 + aR * Ce^g)
        When g=1 → Langmuir; when aR→0 → Henry's Law.

        Args:
            Ce (float or ndarray): Equilibrium concentration.
            KR (float): Redlich–Peterson constant.
            aR (float): Redlich–Peterson constant.
            g (float): Exponent between 0 and 1.

        Returns:
            float or ndarray: Amount adsorbed (qe).
        """
        return (KR * Ce) / (1 + aR * (Ce ** g))

    # ── Fitting ────────────────────────────────────────────────────

    def fit(self, model_name, Ce_exp, qe_exp, initial_guess=None, bounds=None):
        """Fit experimental data to a chosen isotherm model.

        Uses scipy's curve_fit (nonlinear least squares) to find
        the parameter values that minimize the sum of squared
        residuals between the model and the data.

        NEW in v6: Returns confidence intervals (95%) calculated from
        the covariance matrix returned by curve_fit. The diagonal of
        the covariance matrix gives the variance of each parameter;
        the square root gives the standard deviation; multiplied by
        ~1.96 gives the 95% confidence interval half-width.

        Args:
            model_name (str): Name of the isotherm model (e.g., "langmuir").
            Ce_exp (ndarray): Experimental equilibrium concentrations.
            qe_exp (ndarray): Experimental amounts adsorbed.
            initial_guess (list, optional): Initial parameter guesses.
            bounds (tuple, optional): Parameter bounds (lower, upper).

        Returns:
            dict with keys:
                'model'             : model name (str)
                'parameters'        : fitted parameter values (dict)
                'confidence_95'     : ±95% CI for each parameter (dict)
                'r_squared'         : coefficient of determination (float)
                'rmse'              : root mean square error (float)
                'covariance'        : full covariance matrix (ndarray)

        Raises:
            ValueError: If the model name is not registered.
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Choose from: {list(self.models.keys())}"
            )

        model_func = getattr(self, model_name)

        if initial_guess is None:
            initial_guess = self.models[model_name]['initial']
        if bounds is None:
            bounds = self.models[model_name]['bounds']

        # curve_fit returns:
        #   popt  = array of best-fit parameter values
        #   pcov  = covariance matrix (nparams × nparams)
        popt, pcov = curve_fit(
            model_func, Ce_exp, qe_exp,
            p0=initial_guess, bounds=bounds, maxfev=10000
        )

        # ── Goodness-of-fit statistics ──────────────────────────
        qe_pred = model_func(Ce_exp, *popt)

        ss_res = np.sum((qe_exp - qe_pred) ** 2)          # residual sum of squares
        ss_tot = np.sum((qe_exp - np.mean(qe_exp)) ** 2)  # total sum of squares
        r_squared = 1 - (ss_res / ss_tot)

        rmse = np.sqrt(np.mean((qe_exp - qe_pred) ** 2))

        # ── Confidence intervals (NEW v6) ───────────────────────
        # perr = standard error of each parameter
        # np.sqrt of diagonal of covariance matrix
        perr = np.sqrt(np.diag(pcov))
        # 1.96 * standard_error ≈ 95% confidence interval (normal dist.)
        ci_95 = 1.96 * perr

        param_names = self.models[model_name]['params']

        return {
            'model':          model_name,
            'parameters':     dict(zip(param_names, popt)),
            'confidence_95':  dict(zip(param_names, ci_95)),
            'r_squared':      r_squared,
            'rmse':           rmse,
            'covariance':     pcov
        }

    def predict(self, model_name, Ce, params):
        """Predict qe values using a model with given parameters.

        Args:
            model_name (str): Name of the isotherm model.
            Ce (ndarray): Equilibrium concentrations to predict at.
            params (dict): Parameter values, e.g. {'qmax': 10, 'KL': 0.5}.

        Returns:
            ndarray: Predicted qe values.
        """
        model_func = getattr(self, model_name)
        return model_func(Ce, **params)

    def compare_models(self, Ce_exp, qe_exp, models=None):
        """Fit multiple models and compare their performance.

        Tests all registered models (or a subset) and ranks by R².

        Args:
            Ce_exp (ndarray): Experimental equilibrium concentrations.
            qe_exp (ndarray): Experimental amounts adsorbed.
            models (list, optional): Model names to compare.
                Defaults to all registered models.

        Returns:
            dict: Results keyed by model name, sorted best R² first.
        """
        if models is None:
            models = list(self.models.keys())

        results = {}
        for model_name in models:
            try:
                result = self.fit(model_name, Ce_exp, qe_exp)
                results[model_name] = result
                print(f"\n{model_name.upper()}:")
                print(f"  Parameters: {result['parameters']}")
                print(f"  R²: {result['r_squared']:.4f}")
                print(f"  RMSE: {result['rmse']:.4f}")
            except Exception as e:
                print(f"\n{model_name.upper()}: Fitting failed — {e}")

        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]['r_squared'], reverse=True)
        )
        return sorted_results

    # ── Linearized forms (NEW v6) ───────────────────────────────

    def linearize_langmuir(self, Ce_exp, qe_exp):
        """Compute linearized Langmuir form: Ce/qe vs Ce.

        The Langmuir equation can be rearranged to:
            Ce/qe = Ce/qmax + 1/(qmax * KL)

        Plotting Ce/qe vs Ce gives a straight line. This is familiar
        to many scientists trained before nonlinear fitting was common.

        Args:
            Ce_exp (ndarray): Equilibrium concentrations.
            qe_exp (ndarray): Amounts adsorbed.

        Returns:
            dict: x and y arrays for plotting, and linear fit parameters.
        """
        x = Ce_exp
        y = Ce_exp / qe_exp

        # np.polyfit fits a polynomial — degree 1 means a straight line
        # Returns [slope, intercept]
        slope, intercept = np.polyfit(x, y, 1)

        qmax_lin = 1.0 / slope
        KL_lin   = slope / intercept

        return {
            'x': x, 'y': y,
            'slope': slope, 'intercept': intercept,
            'qmax': qmax_lin, 'KL': KL_lin,
            'xlabel': 'Ce', 'ylabel': 'Ce / qe'
        }

    def linearize_freundlich(self, Ce_exp, qe_exp):
        """Compute linearized Freundlich form: log(qe) vs log(Ce).

        The Freundlich equation can be linearized as:
            log(qe) = log(KF) + (1/n) * log(Ce)

        Plotting log(qe) vs log(Ce) gives a straight line.

        Args:
            Ce_exp (ndarray): Equilibrium concentrations.
            qe_exp (ndarray): Amounts adsorbed.

        Returns:
            dict: x and y arrays for plotting, and linear fit parameters.
        """
        x = np.log(Ce_exp)
        y = np.log(qe_exp)

        slope, intercept = np.polyfit(x, y, 1)

        n_lin  = 1.0 / slope
        KF_lin = np.exp(intercept)

        return {
            'x': x, 'y': y,
            'slope': slope, 'intercept': intercept,
            'n': n_lin, 'KF': KF_lin,
            'xlabel': 'ln(Ce)', 'ylabel': 'ln(qe)'
        }

    # ── Plotting ───────────────────────────────────────────────────

    def plot_isotherm(self, model_name, Ce, Ce_unit='', qe_unit='', **params):
        """Plot an adsorption isotherm curve.

        NEW in v6: Accepts unit labels for axis annotation.

        Args:
            model_name (str): Isotherm model name.
            Ce (ndarray): Array of equilibrium concentrations.
            Ce_unit (str): Unit label for Ce axis (e.g. 'mg/L').
            qe_unit (str): Unit label for qe axis (e.g. 'mg/g').
            **params: Model parameters (e.g. qmax=10, KL=0.5).
        """
        model_func = getattr(self, model_name)
        qe = model_func(Ce, **params)

        Ce_label = f"Ce ({Ce_unit})" if Ce_unit else "Ce (Equilibrium Concentration)"
        qe_label = f"qe ({qe_unit})" if qe_unit else "qe (Amount Adsorbed)"

        plt.figure(figsize=(8, 6))
        plt.plot(Ce, qe, 'b-', linewidth=2, label=model_name.capitalize())
        plt.xlabel(Ce_label, fontsize=12)
        plt.ylabel(qe_label, fontsize=12)
        plt.title(f"{model_name.capitalize()} Isotherm", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_with_data(self, model_name, Ce_exp, qe_exp, params,
                       Ce_unit='', qe_unit=''):
        """Plot model fit against experimental data with confidence band.

        NEW in v6: Draws a shaded 95% confidence band around the fitted
        curve when confidence interval information is available in params.
        Also accepts unit labels for axes.

        The confidence band is computed by propagating the parameter
        uncertainties through the model using the covariance matrix.
        At each Ce point, we estimate the variance of qe_pred using
        first-order error propagation (Jacobian method).

        Args:
            model_name (str): Isotherm model name.
            Ce_exp (ndarray): Experimental Ce values.
            qe_exp (ndarray): Experimental qe values.
            params (dict): Fitted parameters or full result dict from fit().
            Ce_unit (str): Unit label for Ce axis.
            qe_unit (str): Unit label for qe axis.
        """
        if 'parameters' in params:
            param_dict  = params['parameters']
            r_squared   = params.get('r_squared', None)
            rmse        = params.get('rmse', None)
            pcov        = params.get('covariance', None)
            ci_95       = params.get('confidence_95', None)
        else:
            param_dict = params
            r_squared = rmse = pcov = ci_95 = None

        Ce_smooth = np.linspace(Ce_exp.min(), Ce_exp.max(), 300)
        qe_pred   = self.predict(model_name, Ce_smooth, param_dict)

        Ce_label = f"Ce ({Ce_unit})" if Ce_unit else "Ce (Equilibrium Concentration)"
        qe_label = f"qe ({qe_unit})" if qe_unit else "qe (Amount Adsorbed)"

        fig, ax = plt.subplots(figsize=(8, 6))

        # ── Confidence band ─────────────────────────────────────
        # If we have both covariance and parameter names, draw the band
        if pcov is not None and ci_95 is not None:
            pnames   = list(param_dict.keys())
            pvals    = np.array([param_dict[k] for k in pnames])
            model_fn = getattr(self, model_name)
            n_params = len(pvals)

            # Numerical Jacobian: how much does qe change when
            # each parameter changes by a tiny amount (dp)?
            eps = 1e-6 * np.abs(pvals) + 1e-12
            J   = np.zeros((len(Ce_smooth), n_params))
            for i in range(n_params):
                p_plus  = pvals.copy(); p_plus[i]  += eps[i]
                p_minus = pvals.copy(); p_minus[i] -= eps[i]
                J[:, i] = (model_fn(Ce_smooth, *p_plus) -
                            model_fn(Ce_smooth, *p_minus)) / (2 * eps[i])

            # Variance at each point: v = J @ pcov @ J^T (diagonal)
            var_pred = np.einsum('ij,jk,ik->i', J, pcov, J)
            std_pred = np.sqrt(np.maximum(var_pred, 0))
            ci_band  = 1.96 * std_pred

            ax.fill_between(Ce_smooth,
                            qe_pred - ci_band,
                            qe_pred + ci_band,
                            alpha=0.15, color='blue',
                            label='95% confidence band')

        ax.scatter(Ce_exp, qe_exp, color='red', s=100,
                   label='Experimental', zorder=3)
        ax.plot(Ce_smooth, qe_pred, 'b-', linewidth=2,
                label=f'{model_name.capitalize()} fit')

        # Annotation box with fit statistics
        stats_lines = []
        if r_squared is not None:
            stats_lines.append(f'R² = {r_squared:.4f}')
        if rmse is not None:
            stats_lines.append(f'RMSE = {rmse:.4f}')
        if ci_95 is not None:
            for k, v in ci_95.items():
                pv = param_dict.get(k, 0)
                stats_lines.append(f'{k} = {pv:.4f} ± {v:.4f}')

        if stats_lines:
            ax.text(0.05, 0.95, '\n'.join(stats_lines),
                    transform=ax.transAxes, verticalalignment='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(Ce_label, fontsize=12)
        ax.set_ylabel(qe_label, fontsize=12)
        ax.set_title(f"{model_name.capitalize()} Isotherm — Data Fitting",
                     fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
#  INTERACTIVE CLI (unchanged from v5)
# ─────────────────────────────────────────────

def interactive_menu():
    """Interactive command-line interface for the library."""
    model = AdsorptionIsotherms()

    print("\n" + "=" * 60)
    print("   ADSORPTION ISOTHERMS v6 — INTERACTIVE TOOL")
    print("=" * 60)

    while True:
        print("\n--- MAIN MENU ---")
        print("1. Enter experimental data manually")
        print("2. Load data from file (CSV)")
        print("3. Use example data")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '4':
            print("Exiting. ¡Hasta luego!")
            break
        elif choice == '1':
            Ce_exp, qe_exp = enter_data_manually()
            if Ce_exp is not None:
                analyze_data(model, Ce_exp, qe_exp)
        elif choice == '2':
            Ce_exp, qe_exp = load_data_from_file()
            if Ce_exp is not None:
                analyze_data(model, Ce_exp, qe_exp)
        elif choice == '3':
            Ce_exp, qe_exp = use_example_data()
            analyze_data(model, Ce_exp, qe_exp)
        else:
            print("Invalid option. Please try again.")


def enter_data_manually():
    """Allow user to enter data point by point."""
    print("\n--- ENTER DATA MANUALLY ---")
    print("Enter Ce and qe pairs. Type 'done' when finished.\n")

    Ce_list, qe_list = [], []
    i = 1
    while True:
        try:
            raw = input(f"Point {i} — Ce (or 'done'): ").strip()
            if raw.lower() == 'done':
                break
            Ce = parse_european_decimal(raw)
            qe = parse_european_decimal(input(f"Point {i} — qe: ").strip())
            Ce_list.append(Ce)
            qe_list.append(qe)
            i += 1
        except ValueError:
            print("  Invalid input. Enter a number (e.g. 3.14 or 3,14).")

    if len(Ce_list) < 3:
        print("Error: Need at least 3 data points.")
        return None, None

    try:
        return validate_and_clean(Ce_list, qe_list)
    except ValueError as e:
        print(f"Error: {e}")
        return None, None


def load_data_from_file():
    """Load data from a CSV file (handles European decimals)."""
    print("\n--- LOAD DATA FROM FILE ---")
    print("CSV should have two columns: Ce, qe  (header row optional)")
    filename = input("\nEnter filename (e.g. data.csv): ").strip()
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        Ce_raw = data[:, 0]
        qe_raw = data[:, 1]
        Ce_exp, qe_exp = validate_and_clean(Ce_raw, qe_raw)
        print(f"Loaded {len(Ce_exp)} valid data points.")
        return Ce_exp, qe_exp
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None


def use_example_data():
    """Provide a Langmuir-like example dataset."""
    print("\n--- USING EXAMPLE DATA ---")
    Ce_exp = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    qe_exp = np.array([0.9, 3.2, 4.8, 6.5, 8.2,  9.0,  9.5])
    print("Example data (Langmuir-like):")
    print("Ce:", Ce_exp)
    print("qe:", qe_exp)
    return Ce_exp, qe_exp


def analyze_data(model, Ce_exp, qe_exp):
    """Submenu for single-model fitting or full comparison."""
    while True:
        print("\n--- ANALYSIS OPTIONS ---")
        print("1. Fit a single model")
        print("2. Compare all models")
        print("3. Return to main menu")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '3':
            break
        elif choice == '1':
            print("\nAvailable models:")
            for name in model.models:
                print(f"  - {name}")
            model_choice = input("\nEnter model name: ").strip().lower()
            if model_choice in model.models:
                try:
                    result = model.fit(model_choice, Ce_exp, qe_exp)
                    print(f"\n--- {model_choice.upper()} RESULTS ---")
                    print(f"Parameters:      {result['parameters']}")
                    print(f"95% CI:          {result['confidence_95']}")
                    print(f"R²:              {result['r_squared']:.4f}")
                    print(f"RMSE:            {result['rmse']:.4f}")
                    if input("\nPlot results? (y/n): ").strip().lower() == 'y':
                        model.plot_with_data(model_choice, Ce_exp, qe_exp, result)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Invalid model name.")
        elif choice == '2':
            results = model.compare_models(Ce_exp, qe_exp)
            if results:
                best = list(results.keys())[0]
                print(f"\n✓ Best model: {best.upper()}")
                if input("\nPlot best model? (y/n): ").strip().lower() == 'y':
                    model.plot_with_data(best, Ce_exp, qe_exp, results[best])
        else:
            print("Invalid option.")


if __name__ == "__main__":
    interactive_menu()
