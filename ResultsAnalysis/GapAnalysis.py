#!/usr/bin/env python3
"""
Single-panel figure:
(A) Tour cost (CRISP instances scaled by ×1000)

Colors are fully user-modifiable via the COLORS dictionary.
"""

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Flags
# -----------------------------
INCLUDE_GUROBI_GENERAL_HEURISTICS = False  # toggle 4th bar

# -----------------------------
# Experiments & algorithms
# -----------------------------
experiments = ["Bridge-1000", "Bridge-2000", "Crisp-1000", "Crisp-2000"]
CRISP_EXPS = {"Crisp-1000", "Crisp-2000"}

algorithms_base = [
    "Group-Cutset Formulation",
    "SCF Formulation",
    "Charge Formulation (Mizutani et. al.)",
]
EXTRA_ALG = "Gurobi general heuristics"

algorithms = (
    algorithms_base + [EXTRA_ALG]
    if INCLUDE_GUROBI_GENERAL_HEURISTICS
    else algorithms_base
)

# -----------------------------
# MODIFIABLE COLORS
# -----------------------------
COLORS = {
    "Group-Cutset Formulation": '#1f77b4',   # blue
    "SCF Formulation": "#2ca02c", # orange
    "Charge Formulation (Mizutani et. al.)": "#ff7f0e"            # green
}

# -----------------------------
# Tour cost values (lower is better)
# -----------------------------
# --- Gap
# values = {
#     groups_sampling[0]: [42.8, 51.04, 48, 62.8],
#     groups_sampling[1]: [6.49, 32.7, 46.3, 86.6],
#     groups_sampling[2]: [12.3, 34.1, 51.3, 77.9],
# }

# --- LB
values = {
    algorithms_base[0]: [575, 462, 0.50, 0.43],
    algorithms_base[1]: [574, 431, 0.44, 0.16],
    algorithms_base[2]: [556, 440, 0.36, 0.27],
}


# -----------------------------
# Scale CRISP tour costs by 1000
# -----------------------------
scaled_values = {alg: [] for alg in algorithms}
for j, exp in enumerate(experiments):
    scale = 1000 if exp in CRISP_EXPS else 1
    for alg in algorithms:
        scaled_values[alg].append(values[alg][j] * scale)

# -----------------------------
# Plot
# -----------------------------
x = np.arange(len(experiments))
bar_width = 0.18 if INCLUDE_GUROBI_GENERAL_HEURISTICS else 0.25

fig, ax = plt.subplots(figsize=(6.5, 4.5))

for i, alg in enumerate(algorithms):
    bars = ax.bar(
        x + i * bar_width,
        scaled_values[alg],
        width=bar_width,
        color=COLORS[alg],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.8,
        label=alg,
    )

    # Value annotations
    for k, b in enumerate(bars):
        v = scaled_values[alg][k]
        ax.annotate(
            f"{v:.1f}" if v >= 10 else f"{v:.3f}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

# -----------------------------
# Formatting
# -----------------------------
# ax.set_title("Final Optimality Gap (T = 1000s)")
# ax.set_ylabel("Gap [Percent]")


ax.set_title("Final Lower Bound Values (CRISP x1000)")
ax.set_ylabel("GIP tour cose lower bound")

ax.set_xticks(x + bar_width * (len(algorithms) - 1) / 2)
ax.set_xticklabels(experiments)
ax.grid(axis="y", linestyle="--", alpha=0.6)

# ax.legend(
#     loc="upper left",
#     frameon=True,
#     fontsize=9,
# )

plt.tight_layout()
plt.show()
print("Done.")
