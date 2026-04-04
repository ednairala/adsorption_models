"""
kinetics/batch.py
=================
Batch (discontinuous / stirred-tank) adsorption kinetics.

Models implemented
------------------
Reaction models (algebraic — fit qt vs time via non-linear regression):
  * Pseudo-first-order  (Lagergren, 1898)
  * Pseudo-second-order (Ho & McKay, 1999)
  * Elovich equation    (Roginsky & Zeldovich, 1934)

Diffusional models (PDE → MOL → stiff ODE system):
  * PVSDM  — Pore Volume AND Surface Diffusion Model
  * PVDM   — Pore Volume Diffusion Model  (Ds = 0)
  * SDM    — Surface Diffusion Model       (Dp = 0)

Numerical approach
------------------
The PVSDM describes intraparticle transport as a PDE in spherical
coordinates.  We discretise the radial dimension r ∈ [0, R] with N
shells using second-order finite differences and integrate the
resulting ODE system with scipy's BDF solver (analogue of MATLAB
ode15s) via ``solve_ivp(..., method='BDF')``.

References
----------
Inglezakis, V.J. & Poulopoulos, S.G. (2006). Adsorption, Ion Exchange
and Catalysis. Elsevier.
Ruthven, D.M. (1984). Principles of Adsorption and Adsorption Processes.
Wiley.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import t as t_dist


# ───────────────────────────────────────────────────────────────────────────
#  Helper: statistical goodness-of-fit metrics
# ───────────────────────────────────────────────────────────────────────────

def _gof_metrics(y_obs: np.ndarray, y_pred: np.ndarray,
                 n_params: int) -> dict:
    """Return R², RMSE, and AIC for a fitted model.

    Args:
        y_obs   : observed values
        y_pred  : model-predicted values
        n_params: number of free parameters in the model

    Returns:
        dict with keys 'r_squared', 'rmse', 'aic'
    """
    n = len(y_obs)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(ss_res / n)

    # AIC = n*ln(SSR/n) + 2*k   (Akaike Information Criterion)
    # Small-sample correction AICc is applied when n/k < ~40
    aic = n * np.log(ss_res / n + 1e-30) + 2 * n_params
    if n - n_params - 1 > 0:
        aic += (2 * n_params * (n_params + 1)) / (n - n_params - 1)  # AICc

    return {"r_squared": round(float(r2), 6),
            "rmse": round(float(rmse), 6),
            "aic": round(float(aic), 4)}


# ───────────────────────────────────────────────────────────────────────────
#  Algebraic reaction models
# ───────────────────────────────────────────────────────────────────────────

class BatchKinetics:
    """Fit and evaluate batch adsorption kinetic models.

    Reaction models
    ~~~~~~~~~~~~~~~
    These are solved algebraically via non-linear regression::

        model = BatchKinetics()
        result = model.fit_pfo(t_exp, qt_exp)

    Diffusional models
    ~~~~~~~~~~~~~~~~~~
    These require the experimental liquid-phase concentration profile
    C(t) and solid-loading profile q(t), plus system parameters::

        result = model.fit_pvsdm(t_exp, C_exp, q_exp,
                                 Cb=50, qe=25, R=5e-4, rho_p=800,
                                 eps_p=0.5, kf=1e-5,
                                 V=0.5, m=1.0)
    """

    # ── Algebraic model equations ───────────────────────────────────────────

    @staticmethod
    def pfo_equation(t: np.ndarray, qe: float, k1: float) -> np.ndarray:
        """Pseudo-first-order (Lagergren) equation.

        qt = qe * (1 - exp(-k1 * t))

        Args:
            t  : time array [min or s]
            qe : adsorption capacity at equilibrium [mg/g]
            k1 : first-order rate constant [1/min]

        Returns:
            qt: solid loading at each time point [mg/g]
        """
        return qe * (1.0 - np.exp(-k1 * t))

    @staticmethod
    def pso_equation(t: np.ndarray, qe: float, k2: float) -> np.ndarray:
        """Pseudo-second-order (Ho & McKay) equation.

        qt = (k2 * qe² * t) / (1 + k2 * qe * t)

        Args:
            t  : time array [min or s]
            qe : equilibrium loading [mg/g]
            k2 : second-order rate constant [g/(mg·min)]

        Returns:
            qt [mg/g]
        """
        return (k2 * qe ** 2 * t) / (1.0 + k2 * qe * t)

    @staticmethod
    def elovich_equation(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Elovich (Roginsky–Zeldovich) equation.

        qt = (1/β) * ln(1 + α·β·t)

        Args:
            t    : time array [min or s]
            alpha: initial adsorption rate [mg/(g·min)]
            beta : desorption constant [g/mg]

        Returns:
            qt [mg/g]
        """
        return (1.0 / beta) * np.log(1.0 + alpha * beta * t)

    # ── Regression wrappers ─────────────────────────────────────────────────

    def _fit_reaction_model(self, model_fn, t: np.ndarray, qt: np.ndarray,
                            p0: list, bounds: tuple,
                            param_names: list) -> dict:
        """Generic non-linear least-squares fitter for reaction models.

        Uses scipy.optimize.least_squares with the 'trf' (Trust Region
        Reflective) algorithm, which handles bounded parameters robustly.

        Args:
            model_fn    : callable model equation (t, *params) → qt
            t, qt       : experimental time and loading arrays
            p0          : initial parameter guesses
            bounds      : (lower, upper) bounds tuple
            param_names : list of parameter names for output dict

        Returns:
            dict with keys: 'parameters', 'covariance', 'confidence_95',
                            'r_squared', 'rmse', 'aic', 'qt_predicted'
        """
        n = len(t)

        def residuals(p):
            return model_fn(t, *p) - qt

        res = least_squares(residuals, p0,
                            bounds=bounds,
                            method='trf',
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)

        # Covariance from the Jacobian: cov ≈ (J^T J)^-1 * MSR
        jac = res.jac
        msr = np.sum(res.fun ** 2) / max(n - len(p0), 1)  # mean squared residual
        try:
            cov = np.linalg.inv(jac.T @ jac) * msr
        except np.linalg.LinAlgError:
            cov = np.full((len(p0), len(p0)), np.nan)

        # 95% confidence interval: t_{α/2, n-k} * sqrt(diag(cov))
        t_crit = t_dist.ppf(0.975, df=max(n - len(p0), 1))
        ci95 = {name: float(t_crit * np.sqrt(max(cov[i, i], 0)))
                for i, name in enumerate(param_names)}

        params = {name: float(v) for name, v in zip(param_names, res.x)}
        qt_pred = model_fn(t, *res.x)
        metrics = _gof_metrics(qt, qt_pred, len(p0))

        return {
            "model": model_fn.__name__.replace("_equation", ""),
            "parameters": params,
            "confidence_95": ci95,
            "covariance": cov,
            **metrics,
            "qt_predicted": qt_pred,
        }

    def fit_pfo(self, t: np.ndarray, qt: np.ndarray,
                qe_init: float | None = None,
                k1_init: float = 0.05) -> dict:
        """Fit the pseudo-first-order (Lagergren) model.

        Args:
            t       : time array [min]
            qt      : measured solid loading at each time [mg/g]
            qe_init : initial guess for qe (defaults to max(qt)*1.1)
            k1_init : initial guess for k1 [1/min]

        Returns:
            Fit result dictionary (see _fit_reaction_model).
        """
        t, qt = np.asarray(t, float), np.asarray(qt, float)
        qe0 = qe_init or float(qt.max() * 1.2)
        return self._fit_reaction_model(
            self.pfo_equation, t, qt,
            p0=[qe0, k1_init],
            bounds=([0, 1e-6], [np.inf, np.inf]),
            param_names=["qe", "k1"]
        )

    def fit_pso(self, t: np.ndarray, qt: np.ndarray,
                qe_init: float | None = None,
                k2_init: float = 1e-3) -> dict:
        """Fit the pseudo-second-order (Ho & McKay) model.

        Args:
            t       : time array [min]
            qt      : measured solid loading [mg/g]
            qe_init : initial guess for qe
            k2_init : initial guess for k2 [g/(mg·min)]

        Returns:
            Fit result dictionary.
        """
        t, qt = np.asarray(t, float), np.asarray(qt, float)
        qe0 = qe_init or float(qt.max() * 1.2)
        return self._fit_reaction_model(
            self.pso_equation, t, qt,
            p0=[qe0, k2_init],
            bounds=([0, 1e-10], [np.inf, np.inf]),
            param_names=["qe", "k2"]
        )

    def fit_elovich(self, t: np.ndarray, qt: np.ndarray,
                    alpha_init: float = 1.0,
                    beta_init: float = 0.1) -> dict:
        """Fit the Elovich equation.

        Args:
            t          : time array [min]
            qt         : measured solid loading [mg/g]
            alpha_init : initial guess for alpha [mg/(g·min)]
            beta_init  : initial guess for beta [g/mg]

        Returns:
            Fit result dictionary.
        """
        t, qt = np.asarray(t, float), np.asarray(qt, float)
        return self._fit_reaction_model(
            self.elovich_equation, t, qt,
            p0=[alpha_init, beta_init],
            bounds=([1e-10, 1e-10], [np.inf, np.inf]),
            param_names=["alpha", "beta"]
        )

    def compare_reaction_models(self, t: np.ndarray,
                                qt: np.ndarray) -> pd.DataFrame:
        """Fit all three reaction models and return a comparison table.

        Args:
            t  : time array [min]
            qt : solid loading array [mg/g]

        Returns:
            pd.DataFrame sorted by AIC (lower = better).
        """
        results = {}
        for name, fn in [("PFO", self.fit_pfo),
                         ("PSO", self.fit_pso),
                         ("Elovich", self.fit_elovich)]:
            try:
                res = fn(t, qt)
                results[name] = {
                    "R²": res["r_squared"],
                    "RMSE": res["rmse"],
                    "AIC": res["aic"],
                    **{f"param_{k}": v for k, v in res["parameters"].items()},
                }
            except Exception as exc:
                results[name] = {"error": str(exc)}

        df = pd.DataFrame(results).T
        df = df[df.get("error", pd.Series(dtype=object)).isna()].copy()
        return df.sort_values("AIC") if "AIC" in df.columns else df

    # ── Diffusional / PDE models via Method of Lines ────────────────────────

    def _build_pvsdm_rhs(self, N: int, r_nodes: np.ndarray,
                         dr: float, R: float,
                         kf: float, Dp: float, Ds: float,
                         eps_p: float, rho_p: float,
                         kf_surf: float,
                         V: float, m: float,
                         Cb: float, isotherm_fn) -> callable:
        """Build the right-hand-side function for the MOL ODE system.

        The PVSDM PDE in spherical coordinates is discretised over
        N internal radial nodes (plus boundary conditions at r=0 and r=R)
        using second-order finite differences.

        State vector layout:
            y[0]        = C_bulk  [mg/L]  liquid-phase bulk concentration
            y[1..N]     = C_p[i]  [mg/L]  pore-fluid conc. at node i
            y[N+1..2N]  = q_s[i]  [mg/g]  surface loading at node i

        Args:
            N           : number of radial shells (typically 10–20)
            r_nodes     : radial positions of interior nodes [m]
            dr          : shell spacing [m]
            R           : particle radius [m]
            kf          : external film mass-transfer coefficient [m/s]
            Dp          : pore diffusivity [m²/s]  (0 to disable)
            Ds          : surface diffusivity [m²/s] (0 to disable)
            eps_p       : particle porosity [-]
            rho_p       : particle density [kg/m³]
            kf_surf     : Henry-like surface-to-pore conversion factor
            V           : solution volume [L]
            m           : adsorbent mass [g]
            Cb          : initial bulk concentration [mg/L]
            isotherm_fn : callable(Cp) → qs_eq  (equilibrium isotherm)

        Returns:
            rhs(t, y) function compatible with scipy solve_ivp.
        """
        r = r_nodes  # shape (N,)

        def rhs(t, y):
            C_bulk = y[0]
            Cp = y[1:N + 1]          # pore concentration, N nodes
            qs = y[N + 1:2 * N + 1]  # surface loading, N nodes

            dCp_dt = np.zeros(N)
            dqs_dt = np.zeros(N)

            # ── Internal nodes (i = 1 … N-2): central differences ─────────
            for i in range(1, N - 1):
                ri = r[i]
                # ∂²Cp/∂r² + (2/r) ∂Cp/∂r
                d2Cp = (Cp[i + 1] - 2 * Cp[i] + Cp[i - 1]) / dr ** 2
                dCp_dr = (Cp[i + 1] - Cp[i - 1]) / (2 * dr)

                # ∂²qs/∂r² + (2/r) ∂qs/∂r
                d2qs = (qs[i + 1] - 2 * qs[i] + qs[i - 1]) / dr ** 2
                dqs_dr = (qs[i + 1] - qs[i - 1]) / (2 * dr)

                flux_p = Dp * (d2Cp + 2 / ri * dCp_dr) if Dp > 0 else 0.0
                flux_s = Ds * rho_p * (d2qs + 2 / ri * dqs_dr) if Ds > 0 else 0.0

                # Local adsorption rate (linear driving force between pore
                # fluid and surface equilibrium)
                qs_eq_i = isotherm_fn(Cp[i])
                ads_rate = kf_surf * (qs_eq_i - qs[i])

                dCp_dt[i] = (flux_p - rho_p * ads_rate) / eps_p
                dqs_dt[i] = ads_rate

            # ── Node 0 (r = 0): symmetry BC — forward FD ─────────────────
            # At the centre: ∂Cp/∂r = 0  →  use forward difference
            d2Cp0 = 2 * (Cp[1] - Cp[0]) / dr ** 2
            d2qs0 = 2 * (qs[1] - qs[0]) / dr ** 2
            qs_eq_0 = isotherm_fn(Cp[0])
            ads_rate_0 = kf_surf * (qs_eq_0 - qs[0])
            dCp_dt[0] = (Dp * d2Cp0 - rho_p * ads_rate_0) / eps_p if Dp > 0 \
                else -rho_p * ads_rate_0 / eps_p
            dqs_dt[0] = ads_rate_0

            # ── Node N-1 (r = R): external film BC — backward FD ─────────
            # Film flux: kf * (C_bulk - Cp[N-1]) = -Dp * ∂Cp/∂r|_{r=R}
            if Dp > 0:
                # Backward difference for ∂Cp/∂r at surface
                dCp_dr_surf = (Cp[N - 1] - Cp[N - 2]) / dr
                # Match film flux to particle-surface gradient
                Cp_surf_bc = Cp[N - 2] + (kf * (C_bulk - Cp[N - 1]) / Dp) * dr
                d2Cp_N = (Cp_surf_bc - 2 * Cp[N - 1] + Cp[N - 2]) / dr ** 2
                flux_p_N = Dp * (d2Cp_N + 2 / R * dCp_dr_surf)
            else:
                flux_p_N = 0.0

            if Ds > 0:
                dqs_dr_surf = (qs[N - 1] - qs[N - 2]) / dr
                d2qs_N = 2 * (qs[N - 2] - qs[N - 1]) / dr ** 2
                flux_s_N = Ds * rho_p * (d2qs_N + 2 / R * dqs_dr_surf)
            else:
                flux_s_N = 0.0

            qs_eq_N = isotherm_fn(Cp[N - 1])
            ads_rate_N = kf_surf * (qs_eq_N - qs[N - 1])
            dCp_dt[N - 1] = (flux_p_N - rho_p * ads_rate_N) / eps_p
            dqs_dt[N - 1] = ads_rate_N

            # ── Bulk mass balance ─────────────────────────────────────────
            # dC/dt = -(m/V) * (3*kf/R) * (C_bulk - Cp[N-1])
            film_flux = kf * (C_bulk - Cp[N - 1])  # [mg/(m²·s)]
            a_s = 3.0 / R  # specific external area of sphere [1/m]
            dC_bulk = -(m / V) * a_s * film_flux   # [mg/(L·s)]

            return np.concatenate([[dC_bulk], dCp_dt, dqs_dt])

        return rhs

    def solve_pvsdm(self, t_span: tuple, t_eval: np.ndarray,
                    Cb: float, qe: float, R: float,
                    rho_p: float, eps_p: float,
                    kf: float, Dp: float, Ds: float,
                    V: float, m: float,
                    isotherm_fn,
                    kf_surf: float = 0.01,
                    N: int = 15) -> dict:
        """Numerically solve the PVSDM using the Method of Lines.

        Solves the PDE system describing intraparticle transport (pore
        volume diffusion + surface diffusion) coupled to the bulk phase.
        Integrates with BDF (stiff solver, analogous to MATLAB ode15s).

        Args:
            t_span      : (t_start, t_end) in seconds
            t_eval      : output time points [s]
            Cb          : initial bulk concentration [mg/L]
            qe          : equilibrium loading [mg/g] (used as initial qs)
            R           : particle radius [m]
            rho_p       : particle bulk density [g/L] (or kg/m³ consistently)
            eps_p       : particle porosity [-]
            kf          : film mass-transfer coefficient [m/s]
            Dp          : pore diffusivity [m²/s] (set to 0 for SDM)
            Ds          : surface diffusivity [m²/s] (set to 0 for PVDM)
            V           : solution volume [L]
            m           : adsorbent mass [g]
            isotherm_fn : callable(Cp [mg/L]) → qs_eq [mg/g]
            kf_surf     : local surface adsorption rate constant [1/s]
            N           : number of radial discretisation nodes

        Returns:
            dict with keys:
              't'       : time array [s]
              'C_bulk'  : bulk concentration profile [mg/L]
              'Cp_surf' : pore conc. at particle surface [mg/L]
              'qs_avg'  : volume-averaged surface loading [mg/g]
              'solver'  : scipy OdeResult object
        """
        dr = R / (N + 1)
        r_nodes = np.linspace(dr, R - dr, N)  # interior nodes

        # Initial conditions: particle initially clean; bulk = Cb
        y0 = np.zeros(1 + 2 * N)
        y0[0] = Cb     # C_bulk
        # y[1..N] = Cp = 0  (clean pore)
        # y[N+1..2N] = qs = 0  (clean surface)

        rhs = self._build_pvsdm_rhs(
            N, r_nodes, dr, R,
            kf, Dp, Ds, eps_p, rho_p, kf_surf,
            V, m, Cb, isotherm_fn
        )

        sol = solve_ivp(
            rhs, t_span, y0,
            method='BDF',       # stiff solver — equivalent to MATLAB ode15s
            t_eval=t_eval,
            rtol=1e-6, atol=1e-8,
            dense_output=False,
            max_step=np.inf
        )

        if not sol.success:
            raise RuntimeError(f"PVSDM solver failed: {sol.message}")

        # Volume-averaged surface loading (Simpson integration over spherical
        # shells): q̄ = (3/R³) ∫₀ᴿ q_s(r) r² dr
        qs_profiles = sol.y[N + 1:2 * N + 1, :]  # shape (N, n_t)
        r_sq = r_nodes ** 2                        # shape (N,)
        qs_avg = 3 / R ** 3 * np.trapz(
            qs_profiles * r_sq[:, np.newaxis], x=r_nodes, axis=0
        )

        return {
            "t": sol.t,
            "C_bulk": sol.y[0],
            "Cp_surf": sol.y[N],         # pore conc. at outermost node
            "qs_avg": qs_avg,
            "solver": sol,
        }

    def solve_pvdm(self, *args, **kwargs) -> dict:
        """Pore Volume Diffusion Model (Ds = 0 limit of PVSDM).

        Accepts the same arguments as solve_pvsdm.  Forces Ds = 0.
        """
        kwargs["Ds"] = 0.0
        return self.solve_pvsdm(*args, **kwargs)

    def solve_sdm(self, *args, **kwargs) -> dict:
        """Surface Diffusion Model (Dp = 0 limit of PVSDM).

        Accepts the same arguments as solve_pvsdm.  Forces Dp = 0.
        """
        kwargs["Dp"] = 0.0
        return self.solve_pvsdm(*args, **kwargs)

    def fit_pvsdm(self, t_exp: np.ndarray, C_exp: np.ndarray,
                  Cb: float, R: float, rho_p: float, eps_p: float,
                  V: float, m: float,
                  isotherm_fn,
                  kf_init: float = 1e-5,
                  Dp_init: float = 1e-11,
                  Ds_init: float = 1e-13,
                  kf_surf: float = 0.01,
                  N: int = 15,
                  model: str = "pvsdm") -> dict:
        """Estimate PVSDM mass-transport parameters from experimental C(t).

        Minimises sum of squared differences between measured and modelled
        bulk-phase concentration profiles using non-linear least-squares
        (scipy.optimize.least_squares, TRF algorithm).

        Args:
            t_exp       : experimental time array [s]
            C_exp       : experimental bulk concentration [mg/L]
            Cb          : initial bulk concentration [mg/L]
            R, rho_p, eps_p, V, m, isotherm_fn, kf_surf, N : see solve_pvsdm
            kf_init     : initial guess for kf [m/s]
            Dp_init     : initial guess for Dp [m²/s]
            Ds_init     : initial guess for Ds [m²/s]
            model       : 'pvsdm' | 'pvdm' | 'sdm'

        Returns:
            dict with 'parameters', 'r_squared', 'rmse', 'aic',
                  'C_predicted', and raw 'least_squares_result'.
        """
        t_exp = np.asarray(t_exp, float)
        C_exp = np.asarray(C_exp, float)
        t_span = (t_exp[0], t_exp[-1])

        # Build parameter list depending on selected model
        if model == "pvsdm":
            p0 = [kf_init, Dp_init, Ds_init]
            p_names = ["kf", "Dp", "Ds"]
            bounds = ([1e-8, 1e-14, 1e-16], [1e-2, 1e-8, 1e-10])
        elif model == "pvdm":
            p0 = [kf_init, Dp_init]
            p_names = ["kf", "Dp"]
            bounds = ([1e-8, 1e-14], [1e-2, 1e-8])
        elif model == "sdm":
            p0 = [kf_init, Ds_init]
            p_names = ["kf", "Ds"]
            bounds = ([1e-8, 1e-16], [1e-2, 1e-10])
        else:
            raise ValueError(f"Unknown model '{model}'. Use 'pvsdm', 'pvdm', or 'sdm'.")

        def residuals(p):
            try:
                kf_ = p[0]
                Dp_ = p[1] if model in ("pvsdm", "pvdm") else 0.0
                Ds_ = p[-1] if model in ("pvsdm", "sdm") else 0.0
                sol = self.solve_pvsdm(
                    t_span, t_exp, Cb, Cb, R, rho_p, eps_p,
                    kf_, Dp_, Ds_, V, m, isotherm_fn, kf_surf, N
                )
                C_mod = np.interp(t_exp, sol["t"], sol["C_bulk"])
                return C_mod - C_exp
            except Exception:
                return np.full_like(C_exp, 1e6)

        res = least_squares(residuals, p0,
                            bounds=bounds,
                            method='trf',
                            ftol=1e-10, xtol=1e-10, gtol=1e-10,
                            verbose=0)

        params = {name: float(v) for name, v in zip(p_names, res.x)}
        C_pred = C_exp + res.fun   # reconstruct
        metrics = _gof_metrics(C_exp, C_pred, len(p0))

        return {
            "model": model.upper(),
            "parameters": params,
            **metrics,
            "C_predicted": C_pred,
            "least_squares_result": res,
        }
