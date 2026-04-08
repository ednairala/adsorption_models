# Integration & GitHub Update Guide

## What you received

```
app.py                         ← replace your existing file
kinetics/
    __init__.py
    batch.py
    fixed_bed.py
    stats.py
```

All four kinetics files and the new `app.py` go into the **same folder** as your
existing `adsorption_isotherms_v6.py`. Nothing else changes in that file.

---

## Step 1 — Folder structure

Your repository should look like this after integration:

```
your-repo/
│
├── app.py                         ← replace with the new version
├── adsorption_isotherms_v6.py     ← unchanged
│
├── kinetics/
│   ├── __init__.py
│   ├── batch.py
│   ├── fixed_bed.py
│   └── stats.py
│
├── requirements.txt               ← update (see below)
├── README.md                      ← update (see below)
└── .gitignore                     ← no change needed
```

**That is all.** Drop the four kinetics files into a new `kinetics/` subfolder
and replace `app.py`. No configuration files, no build step.

---

## Step 2 — Install dependencies

The kinetics module uses packages already required by the equilibrium module,
so the only addition to your environment is none — `numpy`, `scipy`, and `pandas`
are already listed. Just make sure your `requirements.txt` includes:

```txt
streamlit>=1.32
numpy>=1.24
scipy>=1.11
pandas>=2.0
plotly>=5.18
```

Re-install if needed:

```bash
pip install -r requirements.txt
```

---

## Step 3 — Test locally before pushing

```bash
# From inside your repo folder:
streamlit run app.py
```

The app opens in your browser. Verify:

1. Homepage loads with two cards (Equilibrium / Kinetics).
2. "Open Isotherm Fitting" opens the existing equilibrium workflow unchanged.
3. "Open Kinetics Fitting" → Batch → "Use example data" → Reaction Models tab
   → fitting runs and shows PFO/PSO/Elovich comparison.
4. "Back to Home" returns to the homepage.

If you see an import error like `ModuleNotFoundError: No module named 'kinetics'`,
check that the `kinetics/` folder is in the **same directory** as `app.py`, not
inside a subdirectory.

---

## Step 4 — Update requirements.txt (if not already correct)

```bash
pip freeze > requirements.txt
```

Or edit manually to keep it minimal (recommended for a clean open-source repo).

---

## Step 5 — Push to GitHub

```bash
# Stage all new and changed files
git add app.py kinetics/ requirements.txt README.md

# Commit with a clear message
git commit -m "feat: add kinetics module (batch + fixed-bed) and unified homepage"

# Push to your main branch (adjust branch name if needed)
git push origin main
```

If you use a feature-branch workflow:

```bash
git checkout -b feature/kinetics-module
git add app.py kinetics/ requirements.txt README.md
git commit -m "feat: add kinetics module (batch + fixed-bed) and unified homepage"
git push origin feature/kinetics-module
# Then open a Pull Request on GitHub
```

---

## Step 6 — Update README.md

Add a section like this to your existing README:

```markdown
## Modules

### Equilibrium — `adsorption_isotherms_v6.py`
Fits (Ce, qe) data to seven classical isotherm models (Henry, Langmuir,
Freundlich, Temkin, BET, Dubinin–Radushkevich, Redlich–Peterson) using
nonlinear least squares with 95% confidence intervals.

### Kinetics — `kinetics/`
Time-resolved fitting for batch and fixed-bed systems.

**Batch models**
- Reaction: Pseudo-first-order (Lagergren), Pseudo-second-order (Ho & McKay), Elovich
- Diffusional: PVSDM, PVDM, SDM via Method of Lines + BDF stiff solver

**Fixed-bed models**
- Breakthrough: Bohart–Adams, Thomas, Yoon–Nelson, Wolborska
- Design: LUB (Length of Unused Bed), BDST (Bed Depth Service Time)

**Statistics**: R², RMSE, AIC, Student t-test, Fisher F-test

## Usage as a Python library

```python
from adsorption_isotherms_v6 import AdsorptionIsotherms
from kinetics import BatchKinetics, FixedBedKinetics, KineticsStats

# Batch reaction model
bk = BatchKinetics()
result = bk.fit_pso(t_exp, qt_exp)
print(result['parameters'])   # {'qe': ..., 'k2': ...}
print(result['r_squared'])

# Fixed-bed breakthrough
fb = FixedBedKinetics()
result = fb.fit_thomas(t, C_C0, C0=50, Q=0.01, m_ads=5)

# Design calculation
lub = fb.lub(t, C_C0, C0=50, Q=0.01, Z=10)
print(lub['LUB'], lub['utilization_fraction'])
```
```

---

## Step 7 — Deploy on Streamlit Community Cloud (optional)

If you are hosting on [share.streamlit.io](https://share.streamlit.io):

1. Push your changes to GitHub (Step 5).
2. In the Streamlit Cloud dashboard, your app will **auto-redeploy** from the
   connected repository. No manual action needed.
3. If it does not redeploy automatically, click **"Reboot app"** in the dashboard.

Make sure `requirements.txt` is at the repository root — Streamlit Cloud reads it
automatically to install dependencies.

---

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: kinetics` | `kinetics/` folder not next to `app.py` | Move folder to repo root |
| `ImportError: cannot import name 'BatchKinetics'` | Old `__init__.py` missing | Check `kinetics/__init__.py` exists |
| PVSDM solver very slow | Too many radial nodes | Reduce N (default 12 is fine for most cases) |
| Breakthrough model fails to converge | Initial guesses too far from truth | Adjust `kBA_init`, `kTh_init`, etc. in function calls |
| Streamlit Cloud: package missing | `requirements.txt` outdated | Run `pip freeze > requirements.txt` and push |

---

## How navigation works (for developers)

The app uses `st.session_state['page']` as a simple router:

```
'home'        → page_home()
'equilibrium' → page_equilibrium()
'kinetics'    → page_kinetics()
```

Each page function is self-contained. Adding a new module means:
1. Writing a `page_newmodule()` function in `app.py`.
2. Adding a card on the homepage that sets `st.session_state['page'] = 'newmodule'`.
3. Adding an `elif page == 'newmodule': page_newmodule()` in the router at the bottom.

---

*MIT License — free to use, modify, and distribute.*
