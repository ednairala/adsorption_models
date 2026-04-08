# Code Review & Suggestions

## Overall Assessment
Your codebase is **well-structured, well-documented, and implements sophisticated numerical methods correctly**. The code is production-ready for a research tool. Below are targeted suggestions to improve robustness, maintainability, and user experience.

---

## 1. **Streamlit App (`app.py`) — Error Handling & UX**

### 1.1 Add Try-Except Around Fitting Operations
Currently, fitting functions are called without wrapping. If fitting fails, users see a Streamlit traceback instead of a helpful message.

```python
# Current (line ~500-600):
result = bk.fit_pso(t_exp, qt_exp)

# Suggested:
try:
    result = bk.fit_pso(t_exp, qt_exp)
except Exception as e:
    st.error(f"Fitting failed: {str(e)}. Try adjusting initial parameters or bounds.")
    return
```

**Why:** Users uploading ill-conditioned data see cryptic scipy errors. A friendly message + recovery path is better.

---

### 1.2 Refactor `build_plotly_figure()` — Too Long & Complex
The confidence band calculation (lines ~110–150) is embedded in the plotting function. Extract to a separate function.

```python
# Suggested: add to a new utilities module or within app.py
def compute_confidence_bands(Ce_smooth, pvals, pcov, model_fn, ci=0.95):
    """Compute prediction confidence bands via Jacobian propagation."""
    eps = 1e-6 * np.abs(pvals) + 1e-12
    J = np.zeros((len(Ce_smooth), len(pvals)))
    for i in range(len(pvals)):
        p_plus = pvals.copy()
        p_plus[i] += eps[i]
        p_minus = pvals.copy()
        p_minus[i] -= eps[i]
        J[:, i] = (model_fn(Ce_smooth, *p_plus) -
                   model_fn(Ce_smooth, *p_minus)) / (2 * eps[i])
    var_pred = np.einsum('ij,jk,ik->i', J, pcov, J)
    std_pred = np.sqrt(np.maximum(var_pred, 0))
    z_crit = 1.96 if ci == 0.95 else 2.576
    return z_crit * std_pred

def build_plotly_figure(model_name, Ce_exp, qe_exp, result, Ce_unit, qe_unit, show_ci):
    """Build Plotly figure for isotherm."""
    model = AdsorptionIsotherms()
    Ce_smooth = np.linspace(Ce_exp.min(), Ce_exp.max(), 300)
    qe_smooth = model.predict(model_name, Ce_smooth, result['parameters'])
    fig = go.Figure()

    # Add fitted curve
    fig.add_trace(go.Scatter(x=Ce_smooth, y=qe_smooth, ...))

    # Add confidence bands if valid
    if show_ci and _is_covariance_valid(result.get('covariance')):
        ci_band = compute_confidence_bands(
            Ce_smooth,
            np.array([result['parameters'][k] for k in result['parameters'].keys()]),
            result['covariance'],
            getattr(model, model_name)
        )
        fig.add_trace(go.Scatter(...))  # confidence band

    # ... rest of plotting
    return fig
```

**Why:** Easier to test, reuse, and understand. Makes the plotting logic clearer.

---

### 1.3 Add Input Validation Before Fitting
Users can upload malformed CSVs or provide mismatched parameter counts. Validate early.

```python
# Suggested: add to a shared validation function
def validate_fitting_inputs(Ce, qe, model_name, n_params_expected):
    """Validate inputs before fitting."""
    Ce, qe = np.asarray(Ce), np.asarray(qe)

    if len(Ce) != len(qe):
        raise ValueError(f"Ce and qe must have same length. Got {len(Ce)} vs {len(qe)}.")

    if len(Ce) < 3:
        raise ValueError(f"Need ≥3 data points. Got {len(Ce)}.")

    if not np.all(np.isfinite(Ce)) or not np.all(np.isfinite(qe)):
        raise ValueError("Data contains NaN or Inf values. Check CSV format.")

    if np.any(Ce <= 0) or np.any(qe <= 0):
        raise ValueError("Ce and qe must be strictly positive.")

    return Ce, qe
```

---

## 2. **Adsorption Isotherms (`adsorption_isotherms_v6.py`) — Numerical Robustness**

### 2.1 Protect Log-Based Models from Edge Cases
The Temkin and BET models use logarithms. Protect against Ce → 0 or Ce → Cs.

```python
# Current (somewhere in your BET model):
# qe = (qm * C * Ce) / ((1 - Ce/Cs) * (1 + (C - 1) * Ce/Cs))

# Better: add guards
def bet(Ce, qm, C, Cs):
    """BET isotherm with numerical guards."""
    Ce = np.asarray(Ce, dtype=float)

    # Prevent division by zero
    Ce_clipped = np.clip(Ce, 1e-12, Cs * 0.9999)
    denom = (1 - Ce_clipped / Cs) * (1 + (C - 1) * Ce_clipped / Cs)

    # Guard against negative/zero denominator
    result = np.zeros_like(Ce)
    valid = denom > 1e-15
    result[valid] = (qm * C * Ce_clipped[valid]) / denom[valid]
    result[~valid] = 0  # or np.nan, depending on intent

    return result
```

**Why:** Users may upload edge-case data (very low concentrations, etc.). Graceful degradation is better than NaN propagation.

---

### 2.2 Use Named Constants for Tolerances
Scattered magic numbers (1e-30, 1e-12, etc.) reduce readability.

```python
# Suggested: add at module level
EPSILON = 1e-12  # Numerical tolerance for small numbers
ZERO_DIV_PROTECTION = 1e-30  # Prevent division by zero
CONFIDENCE_LEVEL = 0.95  # Default confidence interval level
CONFIDENCE_ALPHA = 1 - CONFIDENCE_LEVEL  # For t-distribution

# Then use:
if se_res < EPSILON:
    t_stat = 0.0
    ...
aic = n * np.log(ss_res / n + ZERO_DIV_PROTECTION) + 2 * n_params
```

---

### 2.3 Add Logging Instead of Print Statements
`print()` statements are fine for scripts, but production tools should use logging.

```python
# Current:
print(f"  [Warning] Removed {n_removed} invalid/missing data point(s).")

# Better:
import logging
logger = logging.getLogger(__name__)

logger.warning(f"Removed {n_removed} invalid/missing data point(s).")
```

**In Streamlit**, you can configure logging to display in the sidebar or suppress for cleaner UI:
```python
# In app.py
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adsorption_library")
```

---

## 3. **Kinetics Modules (`batch.py`, `fixed_bed.py`) — Code Organization**

### 3.1 Create a Results Dataclass Instead of Dicts
Dicts are flexible but lack type safety. Dataclasses improve IDE autocomplete and catch typos early.

```python
# Suggested: add to kinetics/__init__.py or new kinetics/results.py
from dataclasses import dataclass
import numpy as np

@dataclass
class FitResult:
    """Result of a kinetic model fit."""
    model: str
    parameters: dict
    confidence_95: dict
    covariance: np.ndarray
    r_squared: float
    rmse: float
    aic: float
    qt_predicted: np.ndarray  # or C_predicted, depending on context

    def __getitem__(self, key):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)

    def get(self, key, default=None):
        """Dict.get() compatibility."""
        return getattr(self, key, default)

# Usage:
# result = FitResult(
#     model="pso",
#     parameters={"qe": 45.2, "k2": 0.008},
#     ...
# )
# print(result.parameters)   # IDE autocomplete works!
# print(result["parameters"])  # backward compatibility
```

**Why:** Type hints + IDE support reduce bugs. You can gradually migrate without breaking backward compatibility.

---

### 3.2 Extract Common Validation Logic
Both batch.py and fixed_bed.py validate time and loading arrays. DRY principle suggests shared utilities.

```python
# Suggested: add to kinetics/validation.py
def validate_time_series(t, y, model_type="batch"):
    """Validate time-series data for kinetics fitting.

    Args:
        t: time array [min for batch, hours for fixed-bed]
        y: measured values array (qt for batch, C/C0 for fixed-bed)
        model_type: 'batch' or 'fixed_bed'

    Returns:
        (t_clean, y_clean) as float arrays

    Raises:
        ValueError: If validation fails
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) != len(y):
        raise ValueError(f"t and y must have same length. Got {len(t)} vs {len(y)}.")

    if len(t) < 3:
        raise ValueError(f"Need ≥3 data points for fitting. Got {len(t)}.")

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Data contains NaN or Inf.")

    if np.any(t < 0):
        raise ValueError("Time values must be non-negative.")

    if model_type == "batch":
        if np.any(y < 0):
            raise ValueError("qt must be non-negative.")
    elif model_type == "fixed_bed":
        if np.any((y < 0) | (y > 1)):
            raise ValueError("C/C0 must be in [0, 1].")

    # Sort by time
    sort_idx = np.argsort(t)
    return t[sort_idx], y[sort_idx]

# Usage in batch.py:
def fit_pfo(self, t, qt, qe_init=None, k1_init=0.05):
    t, qt = validate_time_series(t, qt, model_type="batch")
    # ... rest of fit
```

---

### 3.3 Covariance Validation Utility
Your covariance checking code (lines ~122–135 in app.py) repeats a pattern. Abstract it.

```python
# Suggested: add to kinetics/stats.py
def is_covariance_valid(cov: np.ndarray, max_condition=1e12) -> bool:
    """Check if covariance matrix is well-conditioned for confidence intervals.

    A covariance matrix is valid if:
      - Shape matches (k, k) for k parameters
      - All entries are finite (no NaN/Inf)
      - Condition number < max_condition (well-conditioned)

    Args:
        cov: covariance matrix
        max_condition: maximum acceptable condition number

    Returns:
        True if cov is valid for confidence interval computation
    """
    if cov is None or cov.size == 0:
        return False

    cov = np.asarray(cov, dtype=float)

    if not np.all(np.isfinite(cov)):
        return False

    try:
        cond = np.linalg.cond(cov)
        return np.isfinite(cond) and cond < max_condition
    except np.linalg.LinAlgError:
        return False

# Usage in app.py:
from kinetics import is_covariance_valid

if show_ci and is_covariance_valid(result.get('covariance')):
    # compute and add confidence bands
    ...
```

---

## 4. **Testing & Validation**

### 4.1 Add Unit Tests for Numerical Models
Your models work, but tests catch regressions and document expected behavior.

```python
# tests/test_batch_kinetics.py
import pytest
import numpy as np
from kinetics import BatchKinetics

def test_pfo_equation_sanity():
    """PFO should approach qe as t → ∞."""
    bk = BatchKinetics()
    t = np.array([0, 10, 100, 1000])
    qe, k1 = 50.0, 0.1
    qt = bk.pfo_equation(t, qe, k1)

    assert qt[0] == 0  # qt(0) = 0
    assert qt[-1] < qe * 1.001  # qt(t→∞) → qe
    assert np.all(np.diff(qt) > 0)  # monotonic increasing

def test_fit_pso_with_synthetic_data():
    """Fit PSO to synthetic data and recover parameters."""
    bk = BatchKinetics()
    t = np.logspace(0, 3, 20)  # 1 to 1000 min
    qe_true, k2_true = 40.0, 0.01
    qt_true = bk.pso_equation(t, qe_true, k2_true)

    # Add 2% noise
    qt_noisy = qt_true + 0.02 * np.random.randn(len(t)) * qt_true

    result = bk.fit_pso(t, qt_noisy)

    # Recovered parameters should be close to truth
    assert abs(result['parameters']['qe'] - qe_true) / qe_true < 0.05
    assert abs(result['parameters']['k2'] - k2_true) / k2_true < 0.10
    assert result['r_squared'] > 0.98

def test_fit_raises_on_bad_data():
    """Fitting should fail gracefully on bad input."""
    bk = BatchKinetics()

    with pytest.raises(ValueError):
        bk.fit_pso([1, 2], [3, 4])  # Too few points

    with pytest.raises(ValueError):
        bk.fit_pso(np.array([1, 2, 3]), np.array([1, np.nan, 3]))  # NaN
```

---

### 4.2 Add Integration Tests for the Streamlit App
Streamlit has testing utilities that simulate user interactions.

```python
# tests/test_app.py
from streamlit.testing.v1 import AppTest

def test_homepage_loads():
    """Homepage should load without errors."""
    at = AppTest.from_file("app.py")
    at.run()
    assert at.session_state['page'] == 'home'
    # Check that buttons exist
    assert len(at.button) >= 2  # At least 2 cards

def test_equilibrium_upload_and_fit():
    """User uploads equilibrium data and auto-fits."""
    at = AppTest.from_file("app.py")
    at.run()

    # Simulate clicking "Equilibrium" card
    at.session_state['page'] = 'equilibrium'
    at.run()

    # Upload a test CSV (mock this with fixtures)
    # ...
```

---

## 5. **Documentation**

### 5.1 Add Docstring Examples
Your docstrings are great, but examples improve discoverability.

```python
# Current docstring (in batch.py, fit_pso method):
"""Fit the pseudo-second-order (Ho & McKay) model.

Args:
    t       : time array [min]
    qt      : measured solid loading at each time [mg/g]
    ...

Returns:
    Fit result dictionary (see _fit_reaction_model).
"""

# Better: add Examples section
"""Fit the pseudo-second-order (Ho & McKay) model.

Args:
    t       : time array [min]
    qt      : measured solid loading at each time [mg/g]
    ...

Returns:
    Fit result dictionary (see _fit_reaction_model).

Examples:
    >>> from kinetics import BatchKinetics
    >>> import numpy as np
    >>> bk = BatchKinetics()
    >>> t = np.array([0, 5, 10, 20, 30, 60, 120, 240])
    >>> qt = np.array([0.5, 10.2, 18.5, 28.3, 34.1, 39.2, 42.5, 44.8])
    >>> result = bk.fit_pso(t, qt)
    >>> print(result['parameters'])
    {'qe': 45.2, 'k2': 0.008}
"""
```

---

### 5.2 Add a CONTRIBUTING.md
Help future contributors understand the codebase structure and conventions.

```markdown
# Contributing

## Code Style
- Follow PEP 8 (use black formatter)
- Type hints for all functions
- Docstrings for all public methods
- 100-character line limit

## Adding a New Isotherm Model
1. Define the model equation in `adsorption_isotherms_v6.py`
2. Add parameter validation & guards against numerical issues
3. Add to the `models` dict and `fit()` method
4. Test with synthetic data
5. Add to `README.md` documentation
6. Submit PR with example notebook

## Testing
Run tests before submitting PR:
```bash
pytest tests/ -v
```
```

---

## 6. **Performance & Scalability**

### 6.1 Implement Fitting Timeout
Long-running fits can hang the Streamlit app.

```python
# In batch.py or app.py
from scipy.optimize import least_squares
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Fitting exceeded 30 seconds. Try simpler model or better initial guesses.")

# Wrap fitting with timeout (Unix only; Windows needs different approach)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30-second timeout

try:
    result = least_squares(...)
finally:
    signal.alarm(0)  # Cancel timeout
```

**Better for Streamlit:** Use `st.cache_data` to cache results and avoid re-fitting on reruns.

```python
@st.cache_data(show_spinner=False)
def fit_isotherm_cached(model_name, Ce_tuple, qe_tuple):
    model = AdsorptionIsotherms()
    return model.fit(model_name, np.array(Ce_tuple), np.array(qe_tuple))
```

---

## 7. **Minor Issues & Polish**

### 7.1 Type Hints Everywhere
Your code uses `float | None` (Python 3.10+ syntax). Ensure consistent.

```python
# Good:
def fit_pfo(self, t: np.ndarray, qt: np.ndarray,
            qe_init: float | None = None,
            k1_init: float = 0.05) -> dict:

# Also good (pre-3.10 compatible):
from typing import Optional, Dict
def fit_pfo(self, t: np.ndarray, qt: np.ndarray,
            qe_init: Optional[float] = None,
            k1_init: float = 0.05) -> Dict[str, Any]:
```

---

### 7.2 Consider Pydantic for Configuration
If users specify many parameters, validate with Pydantic.

```python
from pydantic import BaseModel, Field

class FittingConfig(BaseModel):
    """Configuration for kinetics fitting."""
    initial_qe: float = Field(None, gt=0, description="Initial qe guess [mg/g]")
    k_init: float = Field(0.05, gt=0, description="Initial k guess")
    lower_qe: float = Field(0.1, gt=0, description="Lower bound on qe")
    upper_qe: float = Field(1e6, gt=0, description="Upper bound on qe")
    tolerance: float = Field(1e-12, gt=0, description="Optimization tolerance")

# Validation happens automatically
config_dict = {
    "initial_qe": -5,  # Invalid!
    "k_init": 0.01,
}
config = FittingConfig(**config_dict)  # Raises ValidationError
```

---

### 7.3 Make Equation Display More Readable
In PDF/HTML output, use proper LaTeX rendering.

```python
# Instead of:
eq_str = "qe = (qm·b·Ce)/(1+b·Ce)"

# Use:
eq_latex = r"$q_e = \frac{q_m b C_e}{1 + b C_e}$"

# For Streamlit:
st.markdown(f"**Langmuir:** {eq_latex}")

# For PDF (reportlab):
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
styles = getSampleStyleSheet()
para = Paragraph(eq_latex, styles['Normal'])
```

---

## Summary Table

| Category | Suggestion | Priority | Effort |
|----------|-----------|----------|--------|
| Error Handling | Wrap fitting calls in try-except | High | Low |
| Code Quality | Extract confidence band computation | Medium | Low |
| Robustness | Protect log-based models with guards | Medium | Medium |
| Testing | Add unit tests for models | Medium | Medium |
| Documentation | Add example usage in docstrings | Low | Low |
| Performance | Cache Streamlit fitting results | Medium | Low |
| Type Safety | Use dataclasses for results | Low | Medium |
| Testing | Integration tests for Streamlit | Medium | High |

---

## Final Notes

✅ **Strengths:**
- Excellent numerical implementations (BDF solver, confidence intervals)
- Comprehensive docstrings
- Good separation of concerns (models, stats, UI)
- Well-designed UX for non-experts

⚠️ **Next Steps:**
1. Add error handling to Streamlit app (quick win)
2. Refactor `build_plotly_figure()` (improves maintainability)
3. Add unit tests (improves reliability)
4. Document contribution process (helps adoption)

The codebase is in good shape for a research tool. These suggestions would push it toward production robustness. 🚀
