"""
kinetics/fixed_bed.py
=====================
Fixed-bed (continuous / column) adsorption models.

Breakthrough curve models (empirical, fitted via non-linear regression):
  * Bohart–Adams
  * Thomas
  * Yoon–Nelson
  * Wolborska

Design tools:
  * Length of Unused Bed (LUB)
  * Bed Depth Service Time (BDST)

All models predict the dimensionless effluent-to-influent ratio C/C₀ as
a function of time t at a given flow rate and bed geometry.

References
----------
Bohart, G.S. & Adams, E.Q. (1920). J. Am. Chem. Soc., 42, 523–544.
Thomas, H.C. (1944). J. Am. Chem. Soc., 66, 1664–1666.
Yoon, Y.H. & Nelson, J.H. (1984). Am. Ind. Hyg. Assoc. J., 45, 509–516.
Wolborska, A. (1989). Water Res., 23, 85–91.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t as t_dist

from .batch import _gof_metrics


class FixedBedKinetics:
    """Fit empirical breakthrough models and perform column design calculations.

    Basic usage::

        fb = FixedBedKinetics()
        result = fb.fit_thomas(t, C_C0, C0=50, Q=10, m_ads=5)
        print(result['parameters'])

    Design tools::

        fb.lub(t_exp, C_C0, C0, Q, Z)  # → dict
        fb.bdst(t_exp, C_C0, C0, Q, Z, C_break=0.05)  # → dict
    """

    # ── Breakthrough model equations ────────────────────────────────────────

    @staticmethod
    def bohart_adams(t: np.ndarray, kBA: float, N0: float,
                     C0: float, Q: float, Z: float) -> np.ndarray:
        """Bohart–Adams breakthrough model (C/C₀ vs t).

        C/C₀ = 1 / (1 + exp(kBA*N0*Z/Q - kBA*C0*t))

        Args:
            t   : time array [min]
            kBA : kinetic rate constant [L/(mg·min)]
            N0  : maximum volumetric adsorption capacity [mg/L]
            C0  : influent concentration [mg/L]
            Q   : volumetric flow rate [L/min]
            Z   : bed length [cm]

        Returns:
            C/C₀ dimensionless array
        """
        exponent = kBA * N0 * Z / Q - kBA * C0 * t
        return 1.0 / (1.0 + np.exp(exponent))

    @staticmethod
    def thomas(t: np.ndarray, kTh: float, qTh: float,
               C0: float, Q: float, m_ads: float) -> np.ndarray:
        """Thomas breakthrough model (C/C₀ vs t).

        C/C₀ = 1 / (1 + exp(kTh*qTh*m_ads/Q - kTh*C0*t))

        Args:
            t     : time array [min]
            kTh   : Thomas rate constant [L/(mg·min)]
            qTh   : maximum solid-phase loading [mg/g]
            C0    : influent concentration [mg/L]
            Q     : flow rate [L/min]
            m_ads : adsorbent mass [g]

        Returns:
            C/C₀
        """
        exponent = kTh * qTh * m_ads / Q - kTh * C0 * t
        return 1.0 / (1.0 + np.exp(exponent))

    @staticmethod
    def yoon_nelson(t: np.ndarray, kYN: float, tau: float) -> np.ndarray:
        """Yoon–Nelson breakthrough model (C/C₀ vs t).

        C/C₀ = 1 / (1 + exp(kYN*(tau - t)))

        Args:
            t   : time array [min]
            kYN : rate constant [1/min]
            tau : time for 50% breakthrough [min]

        Returns:
            C/C₀
        """
        return 1.0 / (1.0 + np.exp(kYN * (tau - t)))

    @staticmethod
    def wolborska(t: np.ndarray, beta: float, N0: float,
                  C0: float, Q: float, Z: float) -> np.ndarray:
        """Wolborska breakthrough model (C/C₀ vs t).

        ln(C/C₀) = β*C0*t/N0 - β*Z/u
        where u = Q/A (linear velocity).

        Rearranged to explicit C/C₀ for regression:
        C/C₀ = exp(β*C0*t/N0) / (1 + exp(...))  [simplified form]

        For regression we use: C/C₀ = exp(β*C0/N0*(t - N0*Z/(C0*Q)))
        clipped to [0, 1].

        Args:
            t    : time array [min]
            beta : kinetic coefficient [1/min]
            N0   : bed saturation capacity [mg/L]
            C0   : influent concentration [mg/L]
            Q    : flow rate [L/min]
            Z    : bed height [cm]

        Returns:
            C/C₀ (clipped to [0, 1])
        """
        exponent = (beta * C0 / N0) * (t - N0 * Z / (C0 * Q))
        return np.clip(np.exp(exponent), 0.0, 1.0)

    # ── Regression wrappers ─────────────────────────────────────────────────

    def _fit_bt_model(self, model_fn, t, C_C0,
                      p0, bounds, param_names, fixed_kwargs) -> dict:
        """Generic non-linear least-squares fit for breakthrough models."""
        t = np.asarray(t, float)
        C_C0 = np.asarray(C_C0, float)
        n = len(t)

        def residuals(p):
            return model_fn(t, *p, **fixed_kwargs) - C_C0

        res = least_squares(residuals, p0,
                            bounds=bounds,
                            method='trf',
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)

        jac = res.jac
        msr = np.sum(res.fun ** 2) / max(n - len(p0), 1)
        try:
            cov = np.linalg.inv(jac.T @ jac) * msr
        except np.linalg.LinAlgError:
            cov = np.full((len(p0), len(p0)), np.nan)

        t_crit = t_dist.ppf(0.975, df=max(n - len(p0), 1))
        ci95 = {name: float(t_crit * np.sqrt(max(cov[i, i], 0)))
                for i, name in enumerate(param_names)}

        params = {k: float(v) for k, v in zip(param_names, res.x)}
        C_pred = C_C0 + res.fun
        metrics = _gof_metrics(C_C0, C_pred, len(p0))

        return {
            "model": model_fn.__name__,
            "parameters": params,
            "confidence_95": ci95,
            "covariance": cov,
            **metrics,
            "C_C0_predicted": C_pred,
        }

    def fit_bohart_adams(self, t: np.ndarray, C_C0: np.ndarray,
                         C0: float, Q: float, Z: float,
                         kBA_init: float = 1e-4,
                         N0_init: float = 1000.0) -> dict:
        """Fit Bohart–Adams model.

        Args:
            t, C_C0  : experimental time [min] and C/C₀ [-]
            C0       : influent concentration [mg/L]
            Q        : flow rate [L/min]
            Z        : bed height [cm]
            kBA_init : initial guess for kBA [L/(mg·min)]
            N0_init  : initial guess for N0 [mg/L]

        Returns:
            Fit result dictionary.
        """
        return self._fit_bt_model(
            self.bohart_adams, t, C_C0,
            p0=[kBA_init, N0_init],
            bounds=([1e-8, 1.0], [10.0, 1e7]),
            param_names=["kBA", "N0"],
            fixed_kwargs={"C0": C0, "Q": Q, "Z": Z}
        )

    def fit_thomas(self, t: np.ndarray, C_C0: np.ndarray,
                   C0: float, Q: float, m_ads: float,
                   kTh_init: float = 1e-3,
                   qTh_init: float = 10.0) -> dict:
        """Fit Thomas model.

        Args:
            t, C_C0  : experimental time [min] and C/C₀ [-]
            C0       : influent concentration [mg/L]
            Q        : flow rate [L/min]
            m_ads    : adsorbent mass [g]
            kTh_init : initial guess for kTh [L/(mg·min)]
            qTh_init : initial guess for qTh [mg/g]

        Returns:
            Fit result dictionary.
        """
        return self._fit_bt_model(
            self.thomas, t, C_C0,
            p0=[kTh_init, qTh_init],
            bounds=([1e-8, 0.01], [10.0, 1e5]),
            param_names=["kTh", "qTh"],
            fixed_kwargs={"C0": C0, "Q": Q, "m_ads": m_ads}
        )

    def fit_yoon_nelson(self, t: np.ndarray, C_C0: np.ndarray,
                        kYN_init: float = 0.01,
                        tau_init: float | None = None) -> dict:
        """Fit Yoon–Nelson model.

        Args:
            t, C_C0  : experimental time [min] and C/C₀ [-]
            kYN_init : initial guess for kYN [1/min]
            tau_init : initial guess for tau [min] (defaults to median t)

        Returns:
            Fit result dictionary.
        """
        tau0 = tau_init or float(np.median(t))
        return self._fit_bt_model(
            self.yoon_nelson, t, C_C0,
            p0=[kYN_init, tau0],
            bounds=([1e-6, 0.0], [10.0, np.inf]),
            param_names=["kYN", "tau"],
            fixed_kwargs={}
        )

    def fit_wolborska(self, t: np.ndarray, C_C0: np.ndarray,
                      C0: float, Q: float, Z: float,
                      beta_init: float = 0.01,
                      N0_init: float = 1000.0) -> dict:
        """Fit Wolborska model.

        Args:
            t, C_C0   : experimental time [min] and C/C₀ [-]
            C0        : influent concentration [mg/L]
            Q         : flow rate [L/min]
            Z         : bed height [cm]
            beta_init : initial guess for β [1/min]
            N0_init   : initial guess for N0 [mg/L]

        Returns:
            Fit result dictionary.
        """
        return self._fit_bt_model(
            self.wolborska, t, C_C0,
            p0=[beta_init, N0_init],
            bounds=([1e-6, 1.0], [100.0, 1e7]),
            param_names=["beta", "N0"],
            fixed_kwargs={"C0": C0, "Q": Q, "Z": Z}
        )

    def compare_bt_models(self, t: np.ndarray, C_C0: np.ndarray,
                          C0: float, Q: float, Z: float,
                          m_ads: float) -> pd.DataFrame:
        """Fit all four breakthrough models and return a ranked comparison.

        Args:
            t, C_C0 : experimental time [min] and C/C₀ [-]
            C0, Q, Z, m_ads : column operating parameters

        Returns:
            pd.DataFrame sorted by AIC (best model first).
        """
        results = {}
        fits = {
            "Bohart-Adams": lambda: self.fit_bohart_adams(t, C_C0, C0, Q, Z),
            "Thomas":       lambda: self.fit_thomas(t, C_C0, C0, Q, m_ads),
            "Yoon-Nelson":  lambda: self.fit_yoon_nelson(t, C_C0),
            "Wolborska":    lambda: self.fit_wolborska(t, C_C0, C0, Q, Z),
        }
        for name, fn in fits.items():
            try:
                res = fn()
                results[name] = {
                    "R²": res["r_squared"],
                    "RMSE": res["rmse"],
                    "AIC": res["aic"],
                    **{f"param_{k}": v for k, v in res["parameters"].items()},
                }
            except Exception as exc:
                results[name] = {"error": str(exc)}

        df = pd.DataFrame(results).T
        return df.sort_values("AIC") if "AIC" in df.columns else df

    # ── Design tools ────────────────────────────────────────────────────────

    def lub(self, t_exp: np.ndarray, C_C0: np.ndarray,
            C0: float, Q: float, Z: float,
            C_break: float = 0.05,
            C_sat: float = 0.95) -> dict:
        """Calculate the Length of Unused Bed (LUB).

        LUB = Z * (1 - t_b / t_s)

        where:
          t_b = time at breakthrough (C/C₀ = C_break)
          t_s = time at saturation   (C/C₀ = C_sat)

        Args:
            t_exp   : time array [min]
            C_C0    : C/C₀ array [-]
            C0      : influent concentration [mg/L]
            Q       : flow rate [L/min]
            Z       : total bed height [cm]
            C_break : breakthrough fraction (default 0.05 = 5%)
            C_sat   : saturation fraction (default 0.95 = 95%)

        Returns:
            dict with 'LUB' [cm], 't_breakthrough', 't_saturation',
            'utilization_fraction', and 'throughput_at_break' [L].
        """
        t_exp = np.asarray(t_exp, float)
        C_C0 = np.asarray(C_C0, float)

        # Interpolate to find t_b and t_s
        if np.max(C_C0) < C_break:
            raise ValueError(
                f"C/C₀ never reaches breakthrough fraction {C_break}. "
                "Extend the experiment."
            )
        t_b = float(np.interp(C_break, C_C0, t_exp))
        t_s = float(np.interp(C_sat, C_C0, t_exp)) \
            if np.max(C_C0) >= C_sat else float(t_exp[-1])

        lub_val = Z * (1.0 - t_b / t_s)
        utilization = 1.0 - lub_val / Z
        throughput_b = Q * t_b  # [L] of solution processed at breakthrough

        return {
            "LUB": round(lub_val, 4),
            "t_breakthrough": round(t_b, 2),
            "t_saturation": round(t_s, 2),
            "utilization_fraction": round(utilization, 4),
            "throughput_at_break_L": round(throughput_b, 2),
            "Z_total_cm": Z,
        }

    def bdst(self, t_exp: np.ndarray, C_C0: np.ndarray,
             C0: float, Q: float, A: float,
             bed_heights: np.ndarray | None = None,
             C_break: float = 0.05) -> dict:
        """Bed Depth Service Time (BDST) analysis.

        Fits the linear BDST equation:
          t_b = (N0 / (C0 * u)) * Z - (1 / kBA) * ln(C0/C_b - 1)

        where u = Q/A (superficial linear velocity).

        This function accepts a single experiment and fits the BDST line
        using multiple C/C₀ cut-offs if bed_heights is None, or uses
        provided bed_heights.

        Args:
            t_exp       : time array [min]
            C_C0        : C/C₀ array from one experiment
            C0          : influent concentration [mg/L]
            Q           : volumetric flow rate [mL/min or L/min, consistent with A]
            A           : cross-sectional area of bed [cm²]
            bed_heights : array of Z values [cm] for multi-column BDST.
                          If None, uses a single column (analysis via C/C₀ cuts)
            C_break     : breakthrough cut-off fraction (default 0.05)

        Returns:
            dict with 'N0' [mg/L], 'kBA' [L/(mg·min)], 'slope', 'intercept',
            'r_squared' of the BDST line, and 'service_times' DataFrame.
        """
        t_exp = np.asarray(t_exp, float)
        C_C0 = np.asarray(C_C0, float)
        u = Q / A  # linear velocity [cm/min or L/(min·cm²)]

        if bed_heights is not None:
            # Multi-column mode: user provides t_b at each Z
            raise NotImplementedError(
                "Multi-column BDST requires per-column breakthrough times. "
                "Pass t_b_array and Z_array directly to bdst_multicolumn()."
            )

        # Single-column mode: sweep C_break from 5% to 85% in steps of 5%,
        # treating each fraction as a pseudo-data point
        fractions = np.arange(0.05, 0.90, 0.05)
        t_breaks = []
        for f in fractions:
            if np.max(C_C0) >= f:
                tb = float(np.interp(f, C_C0, t_exp))
                t_breaks.append((f, tb))

        if len(t_breaks) < 3:
            raise ValueError(
                "Insufficient breakthrough data to build BDST curve."
            )

        frac_arr = np.array([x[0] for x in t_breaks])
        tb_arr = np.array([x[1] for x in t_breaks])

        # BDST intercept (at Z = 0):  b = -(1/kBA) * ln(C0/Cb - 1)
        # For a single Z experiment, BDST is generated as t_b vs ln(C0/Cb-1)
        ln_term = np.log(C0 / (C0 * frac_arr) - 1)

        # Linear fit: t_b = a * ln_term + b  → kBA = -1/a, t0 = b
        coeffs = np.polyfit(ln_term, tb_arr, deg=1)
        slope_bt, intercept_bt = coeffs

        t_b_fit = np.polyval(coeffs, ln_term)
        ss_res = np.sum((tb_arr - t_b_fit) ** 2)
        ss_tot = np.sum((tb_arr - tb_arr.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        kBA_est = -1.0 / slope_bt if abs(slope_bt) > 1e-12 else np.nan

        service_df = pd.DataFrame({
            "C/C0_fraction": np.round(frac_arr, 2),
            "t_break_min": np.round(tb_arr, 2),
            "t_break_fit_min": np.round(t_b_fit, 2),
        })

        return {
            "N0_estimate": None,   # requires known Z in multi-col mode
            "kBA_estimate": round(float(kBA_est), 8) if not np.isnan(kBA_est) else None,
            "slope": round(float(slope_bt), 6),
            "intercept": round(float(intercept_bt), 4),
            "r_squared": round(float(r2), 5),
            "service_times": service_df,
            "note": "Single-column BDST: kBA estimated from slope of t_b vs ln(C0/Cb-1). "
                    "Use bdst_multicolumn() for full N0 and kBA estimation."
        }

    def bdst_multicolumn(self, Z_array: np.ndarray,
                         t_break_array: np.ndarray,
                         C0: float, Q: float, A: float,
                         C_break: float = 0.05) -> dict:
        """Full BDST analysis using breakthrough times from multiple columns.

        Fits:  t_b = (N0 / (C0 * u)) * Z  -  (1/kBA) * ln(C0/Cb - 1)

        Args:
            Z_array        : bed heights [cm], one per column experiment
            t_break_array  : breakthrough times [min] at C_break fraction
            C0             : influent concentration [mg/L]
            Q              : flow rate [L/min or mL/min, consistent with A]
            A              : cross-sectional area [cm²]
            C_break        : breakthrough fraction

        Returns:
            dict with 'N0' [mg/L], 'kBA' [L/(mg·min)], 'slope', 'intercept',
            'r_squared'.
        """
        Z = np.asarray(Z_array, float)
        tb = np.asarray(t_break_array, float)
        u = Q / A

        # Linear fit: tb = slope * Z + intercept
        coeffs = np.polyfit(Z, tb, deg=1)
        slope, intercept = coeffs

        # slope = N0 / (C0 * u)  →  N0 = slope * C0 * u
        N0 = slope * C0 * u
        # intercept = -(1/kBA) * ln(C0/Cb - 1)  →  kBA = -ln(...) / intercept
        ln_term = np.log(C0 / (C0 * C_break) - 1.0)
        kBA = ln_term / (-intercept) if abs(intercept) > 1e-12 else np.nan

        tb_fit = np.polyval(coeffs, Z)
        ss_res = np.sum((tb - tb_fit) ** 2)
        ss_tot = np.sum((tb - tb.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "N0": round(float(N0), 4),
            "kBA": round(float(kBA), 8) if not np.isnan(kBA) else None,
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r_squared": round(float(r2), 5),
            "linear_velocity_u": round(float(u), 6),
        }
