"""
kinetics — Adsorption Kinetics Module
======================================
Sub-packages:
  kinetics.batch      — Batch (discontinuous) kinetic models
  kinetics.fixed_bed  — Fixed-bed (continuous) breakthrough models
  kinetics.stats      — Statistical validation utilities
"""

from .batch import BatchKinetics
from .fixed_bed import FixedBedKinetics
from .stats import KineticsStats

__all__ = ["BatchKinetics", "FixedBedKinetics", "KineticsStats"]
__version__ = "1.0.0"
