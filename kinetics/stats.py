"""
kinetics/stats.py
=================
Statistical validation utilities for kinetic model fitting.

Implements:
  * Student t-test — test if model predictions differ significantly from
    experimental observations
  * Fisher exact F-test — compare two model fits (are the residuals
    significantly different?)
  * Model selection summary (R², RMSE, AIC comparison)

References
----------
Press, W.H. et al. (2007). Numerical Recipes (3rd ed.). Cambridge.
Motulsky, H. & Christopoulos, A. (2004). Fitting Models to Biological Data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, f as f_dist


class KineticsStats:
    """Statistical tests for comparing model fits to experimental data.

    Usage::

        ks = KineticsStats()

        # t-test: are model predictions significantly different from data?
        result = ks.ttest(y_obs, y_pred, n_params=2)

        # F-test: is model A significantly better than model B?
        result = ks.ftest(y_obs, y_pred_A, n_params_A=2,
                                 y_pred_B, n_params_B=3)
    """

    @staticmethod
    def ttest(y_obs: np.ndarray, y_pred: np.ndarray,
              n_params: int,
              alpha: float = 0.05) -> dict:
        """One-sample Student t-test: residuals vs zero.

        Tests H₀: mean residual = 0 (model fits data without systematic bias).
        A significant result (p < alpha) indicates that model predictions are
        systematically different from observations.

        Args:
            y_obs    : experimental observations
            y_pred   : model-predicted values
            n_params : number of free model parameters (for degrees of freedom)
            alpha    : significance level (default 0.05)

        Returns:
            dict with:
              't_statistic'   : computed t value
              't_critical'    : critical value at alpha/2 (two-tailed)
              'p_value'       : two-tailed p-value
              'df'            : degrees of freedom (n - n_params)
              'mean_residual' : mean of (y_pred - y_obs)
              'se_residual'   : standard error of residuals
              'reject_H0'     : True if p < alpha
              'interpretation': human-readable conclusion
        """
        y_obs = np.asarray(y_obs, float)
        y_pred = np.asarray(y_pred, float)
        residuals = y_pred - y_obs
        n = len(residuals)
        df = max(n - n_params, 1)

        mean_res = float(np.mean(residuals))
        se_res = float(np.std(residuals, ddof=1) / np.sqrt(n))

        if se_res < 1e-15:
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat = mean_res / se_res
            p_val = float(2 * t_dist.sf(abs(t_stat), df=df))

        t_crit = float(t_dist.ppf(1 - alpha / 2, df=df))
        reject = bool(p_val < alpha)

        if reject:
            interpretation = (
                f"REJECT H₀ (p = {p_val:.4f} < α = {alpha}): the model "
                "predictions are systematically biased relative to the data. "
                "Consider a more appropriate model."
            )
        else:
            interpretation = (
                f"FAIL TO REJECT H₀ (p = {p_val:.4f} ≥ α = {alpha}): no "
                "evidence of systematic bias. The model is statistically "
                "consistent with the experimental data."
            )

        return {
            "t_statistic": round(t_stat, 5),
            "t_critical": round(t_crit, 5),
            "p_value": round(p_val, 6),
            "df": df,
            "mean_residual": round(mean_res, 6),
            "se_residual": round(se_res, 6),
            "reject_H0": reject,
            "significance_level": alpha,
            "interpretation": interpretation,
        }

    @staticmethod
    def ftest(y_obs: np.ndarray,
              y_pred_A: np.ndarray, n_params_A: int,
              y_pred_B: np.ndarray, n_params_B: int,
              alpha: float = 0.05) -> dict:
        """Fisher F-test: compare two nested or non-nested model fits.

        Tests H₀: both models explain the data equally well.
        A significant result (p < alpha) means that model B (with more
        parameters) is significantly better than model A.

        The F statistic is:
          F = [(SSR_A - SSR_B) / (df_B - df_A)] / [SSR_B / df_B]

        where SSR is the sum of squared residuals, df = n - k.

        Args:
            y_obs      : experimental observations (same for both models)
            y_pred_A   : predictions from model A (fewer parameters / simpler)
            n_params_A : number of free parameters in model A
            y_pred_B   : predictions from model B (more parameters / complex)
            n_params_B : number of free parameters in model B
            alpha      : significance level

        Returns:
            dict with F statistic, p-value, degrees of freedom,
            critical F value, reject_H0 flag, and interpretation.
        """
        y_obs = np.asarray(y_obs, float)
        y_pred_A = np.asarray(y_pred_A, float)
        y_pred_B = np.asarray(y_pred_B, float)
        n = len(y_obs)

        ssr_A = float(np.sum((y_obs - y_pred_A) ** 2))
        ssr_B = float(np.sum((y_obs - y_pred_B) ** 2))

        df_A = n - n_params_A
        df_B = n - n_params_B
        d_df = abs(df_B - df_A)  # difference in degrees of freedom

        if ssr_B < 1e-30 or d_df == 0:
            return {
                "F_statistic": np.nan,
                "F_critical": np.nan,
                "p_value": 1.0,
                "df_numerator": d_df,
                "df_denominator": df_B,
                "SSR_A": round(ssr_A, 6),
                "SSR_B": round(ssr_B, 6),
                "reject_H0": False,
                "interpretation": "Cannot compute F-test (degenerate case).",
            }

        F_stat = ((ssr_A - ssr_B) / d_df) / (ssr_B / df_B)
        p_val = float(f_dist.sf(F_stat, dfn=d_df, dfd=df_B))
        F_crit = float(f_dist.ppf(1 - alpha, dfn=d_df, dfd=df_B))
        reject = bool(p_val < alpha)

        if reject:
            interpretation = (
                f"REJECT H₀ (p = {p_val:.4f} < α = {alpha}): model B "
                f"({n_params_B} params) is significantly better than model A "
                f"({n_params_A} params). The additional parameters are justified."
            )
        else:
            interpretation = (
                f"FAIL TO REJECT H₀ (p = {p_val:.4f} ≥ α = {alpha}): model B "
                f"({n_params_B} params) is NOT significantly better than model A "
                f"({n_params_A} params). Prefer the simpler model (Occam's razor)."
            )

        return {
            "F_statistic": round(float(F_stat), 5),
            "F_critical": round(F_crit, 5),
            "p_value": round(p_val, 6),
            "df_numerator": d_df,
            "df_denominator": df_B,
            "SSR_A": round(ssr_A, 6),
            "SSR_B": round(ssr_B, 6),
            "reject_H0": reject,
            "significance_level": alpha,
            "interpretation": interpretation,
        }

    @staticmethod
    def compare_models_table(results_dict: dict,
                             y_obs: np.ndarray,
                             alpha: float = 0.05) -> pd.DataFrame:
        """Build a full model comparison table including statistical tests.

        Args:
            results_dict : dict mapping model name → fit result dict
                           (each must have 'parameters', 'r_squared',
                            'rmse', 'aic', and a predicted array key)
            y_obs        : experimental observations
            alpha        : significance level for t-tests

        Returns:
            pd.DataFrame with one row per model, sorted by AIC.
        """
        rows = []
        ks = KineticsStats()

        # Identify the predicted-values key (varies by model family)
        def _get_pred(res):
            for key in ("qt_predicted", "C_predicted", "C_C0_predicted"):
                if key in res:
                    return res[key]
            return None

        for name, res in results_dict.items():
            y_pred = _get_pred(res)
            n_params = len(res.get("parameters", {}))

            row = {
                "Model": name,
                "R²": res.get("r_squared", np.nan),
                "RMSE": res.get("rmse", np.nan),
                "AIC": res.get("aic", np.nan),
                "n_params": n_params,
            }

            if y_pred is not None:
                t_res = ks.ttest(y_obs, y_pred, n_params=n_params, alpha=alpha)
                row["t_p_value"] = t_res["p_value"]
                row["bias_H0_rejected"] = t_res["reject_H0"]

            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")
        return df.sort_values("AIC")

    @staticmethod
    def residual_analysis(y_obs: np.ndarray,
                          y_pred: np.ndarray,
                          model_name: str = "Model") -> dict:
        """Compute a suite of residual diagnostics.

        Args:
            y_obs       : experimental observations
            y_pred      : model predictions
            model_name  : label for the output dict

        Returns:
            dict with mean, std, max absolute error, normality flag
            (Shapiro-Wilk if n ≤ 50), and a residuals array.
        """
        from scipy.stats import shapiro

        y_obs = np.asarray(y_obs, float)
        y_pred = np.asarray(y_pred, float)
        residuals = y_obs - y_pred

        diag = {
            "model": model_name,
            "n": len(residuals),
            "mean_residual": round(float(np.mean(residuals)), 6),
            "std_residual": round(float(np.std(residuals, ddof=1)), 6),
            "max_abs_error": round(float(np.max(np.abs(residuals))), 6),
            "residuals": residuals,
        }

        if len(residuals) >= 3:
            try:
                stat, p_sw = shapiro(residuals)
                diag["shapiro_wilk_stat"] = round(float(stat), 5)
                diag["shapiro_wilk_p"] = round(float(p_sw), 5)
                diag["residuals_normal"] = bool(p_sw >= 0.05)
            except Exception:
                diag["shapiro_wilk_stat"] = None
                diag["shapiro_wilk_p"] = None
                diag["residuals_normal"] = None

        return diag
