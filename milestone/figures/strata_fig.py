"""Milestone Figure 2: the same propagation_strata chart from the val tables, with larger fonts so it stays legible at
half the text width. Numbers come from results/step8/val/tables.json unchanged.

Usage (laptop venv): python milestone/figures/strata_fig.py results/step8/val/tables.json \
    milestone/figures/propagation_strata_val.png"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skier.report.figures import propagation_strata

tables, out = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({"font.size": 17, "axes.titlesize": 17, "legend.fontsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14})
strata = json.loads(tables.read_text())["propagation"]["strata"]
print(propagation_strata(strata, out))
