"""
Pokemon Red RL — analysis utilities.

Reusable, importable analysis logic that powers both ``scripts/analyze.py``
(post-hoc rliable figures for the paper) and ``scripts/compare.py`` (live
Streamlit treatment comparisons).
"""

from .comparison import (
    TREATMENT_COLORS,
    TREATMENT_DISPLAY,
    KNOWN_TREATMENTS,
    detect_treatment,
    group_runs_by_treatment,
    learning_curves_with_bands,
    treatment_summary_table,
    milestone_first_episode,
    final_performance,
    setup_publication_style,
    export_figure,
)

__all__ = [
    "TREATMENT_COLORS",
    "TREATMENT_DISPLAY",
    "KNOWN_TREATMENTS",
    "detect_treatment",
    "group_runs_by_treatment",
    "learning_curves_with_bands",
    "treatment_summary_table",
    "milestone_first_episode",
    "final_performance",
    "setup_publication_style",
    "export_figure",
]
