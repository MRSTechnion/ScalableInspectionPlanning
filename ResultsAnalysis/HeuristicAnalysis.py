#!/usr/bin/env python3
"""
Split the original 3-panel figure into 3 separate PDF figures (recommended for papers).

Exports:
  - PanelA_TourCost.pdf
  - PanelB_Runtime.pdf
  - PanelC_Anytime_Bridge1000.pdf

Optionally also exports the combined 3-panel version:
  - Combined_ABC.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import Utils.paperstyle as ps

# -----------------------------
# Paper style + PDF font embedding
# -----------------------------
ps.use_paper_style()
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# -----------------------------
# Output
# -----------------------------
output_dest = "/home/adir/Desktop/IP-results/pdf_results"
os.makedirs(output_dest, exist_ok=True)

# -----------------------------
# Flags
# -----------------------------
INCLUDE_GUROBI_GENERAL_HEURISTICS = False   # toggle this (Panel A only)
USE_LOG_TIME = True                         # toggle this (Panel B y-scale)
EXPORT_COMBINED_FIGURE = True              # optional convenience export

# -----------------------------
# Experiments & algorithms
# -----------------------------
experiments = ["Bridge-1000", "Bridge-2000", "Crisp-1000", "Crisp-2000"]
CRISP_EXPS = {"Crisp-1000", "Crisp-2000"}

algorithms_base = [
    "Covering Tree + Greedy Matching",
    "Covering Tree + Perfect Matching",
    "ST Heuristic",
]
EXTRA_ALG = "Gurobi general heuristics"

algorithms_panelA = (
    algorithms_base + [EXTRA_ALG]
    if INCLUDE_GUROBI_GENERAL_HEURISTICS
    else algorithms_base
)

# -----------------------------
# Fixed colors (consistent across all panels)
# -----------------------------
COLORS = {
    algorithms_base[0]: "powderblue",
    algorithms_base[1]: "palevioletred",
    algorithms_base[2]: "lightsalmon",
    EXTRA_ALG:          "#d62728",
}

# -----------------------------
# Tour cost values (lower is better)
# -----------------------------
values = {
    algorithms_base[0]: [1030.858697, 1067.758323, 1.0736472473, 1.32223787616],
    algorithms_base[1]: [1000.768093, 1037.684464, 0.9746528724, 1.20125329467],
    algorithms_base[2]: [1424.60072, 1603.900669, 1.21952137587, 1.51330701857],
    EXTRA_ALG:          [144726.564, 245249.813, 5.50834, 27.37700],
}

# -----------------------------
# Runtimes (seconds; only the base 3 methods)
# -----------------------------
times = {
    algorithms_base[0]: [0.8011, 2.6737, 0.9794, 3.4974],
    algorithms_base[1]: [5.3740, 11.2588, 3.6447, 11.3328],
    algorithms_base[2]: [1.8399, 4.7004, 4.8746, 14.8978],
}

# -----------------------------
# Anytime curves (Bridge1000)
# -----------------------------
T_END = 500.0
pts_map = {
    algorithms_base[0]: [
        (3.0, 1054.4), (6.0, 1047.4), (12.0, 1022.7),
        (16.0, 1001.9), (19.0, 983.1), (34.0, 979.1),
        (83.0, 976.9), (102.0, 970.3), (139.0, 948.7),
        (291.0, 946.7), (346.0, 934.8), (500.0, 934.8),
    ],
    algorithms_base[1]: [
        (2.0, 1008.7), (27.0, 983.9), (41.0, 957.3),
        (76.0, 944.8), (135.0, 941.0), (163.0, 932.5),
        (295.0, 922.4), (304.0, 915.8), (493.0, 903.2),
        (500.0, 903.2),
    ],
    algorithms_base[2]: [
        (3.0, 1347.4), (7.0, 1343.4),
        (439.0, 1325.4), (500.0, 1325.4),
    ],
}


# ============================================================
# Standalone Legend Export
# ============================================================
def export_legend(savepath):
    # Create a small figure specifically for the legend
    fig = plt.figure(figsize=(10, 1))
    ax = fig.add_subplot(111)

    # Hide the axes completely so they don't bleed through
    ax.axis('off')

    from matplotlib.patches import Patch

    # Create handles based on your current algorithms_panelA
    handles = [
        Patch(facecolor=COLORS[alg], edgecolor='black', label=alg)
        for alg in algorithms_panelA
    ]

    # Create the legend
    legend = fig.legend(
        handles=handles,
        labels=algorithms_panelA,
        loc='center',
        frameon=True,
        ncol=len(handles),
        handletextpad=0.5,
        columnspacing=1.5
    )

    # Force the legend background to be opaque white and on top
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_alpha(1.0)  # Ensure no transparency
    frame.set_zorder(10)  # Move it to the front

    # Draw and export only the legend area
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(savepath, format="pdf", bbox_inches=bbox, transparent=False)
    plt.close(fig)

def stepify_and_extend(points, t_end):
    points = sorted(points)
    xs, ys = [points[0][0]], [points[0][1]]
    for t, v in points[1:]:
        xs += [t, t]
        ys += [ys[-1], v]
    if xs[-1] < t_end:
        xs.append(t_end)
        ys.append(ys[-1])
    return xs, ys


# -----------------------------
# Scale CRISP tour costs by 1000 for Panel A
# -----------------------------
scaled_values = {alg: [] for alg in algorithms_panelA}
for j, exp in enumerate(experiments):
    scale = 1000 if exp in CRISP_EXPS else 1
    for alg in algorithms_panelA:
        scaled_values[alg].append(values[alg][j] * scale)


def centered_bar_positions(n_groups, n_methods):
    """
    Returns x positions for each method so bars are centered around each group tick.
    """
    x = np.arange(n_groups)
    # width chosen so that total bar span is reasonable
    # If 3 methods -> width ~0.22; If 4 -> width ~0.17
    width = 0.22 if n_methods == 3 else 0.17
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width
    return x, width, offsets


# ============================================================
# Panel A (Tour cost)
# ============================================================
def plot_panel_a(savepath):
    n_methods = len(algorithms_panelA)
    x, bar_width, offsets = centered_bar_positions(len(experiments), n_methods)

    fig, ax = plt.subplots(figsize=(8, 6)) # Smaller, standard size

    for i, alg in enumerate(algorithms_panelA):
        bars = ax.bar(
            x + offsets[i],
            scaled_values[alg],
            width=bar_width,
            color=COLORS[alg],
            edgecolor="black",
            linewidth=0.8,
            label=alg,
        )

        # # annotations (kept from your original, but now larger/cleaner)
        # for k, b in enumerate(bars):
        #     v = scaled_values[alg][k]
        #     ax.annotate(
        #         f"{v:.1f}" if v >= 10 else f"{v:.3f}",
        #         (b.get_x() + b.get_width() / 2, b.get_height()),
        #         xytext=(0, 3),
        #         textcoords="offset points",
        #         ha="center",
        #         va="bottom",
        #         fontsize=8,
        #     )

    # ax.set_title("Tour cost (CRISP ×1000)")
    # ax.set_ylabel("Tour cost (CRISP ×1000)")
    ax.set_ylabel("Tour cost ")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # legend local to this panel (since we split)
    ncol = 1 if n_methods <= 2 else 2
    # ax.legend(frameon=True, ncol=ncol)

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    fig.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Panel B (Runtime)
# ============================================================
def plot_panel_b(savepath):
    n_methods = len(algorithms_base)
    x, bar_width, offsets = centered_bar_positions(len(experiments), n_methods)

    fig, ax = plt.subplots(figsize=(8, 6)) # Smaller, standard size

    for i, alg in enumerate(algorithms_base):
        bars = ax.bar(
            x + offsets[i],
            times[alg],
            width=bar_width,
            color=COLORS[alg],
            edgecolor="black",
            linewidth=0.8,
            label=alg,
        )
        # for k, b in enumerate(bars):
        #     ax.annotate(
        #         f"{times[alg][k]:.2f}s",
        #         (b.get_x() + b.get_width() / 2, b.get_height()),
        #         xytext=(0, 3),
        #         textcoords="offset points",
        #         ha="center",
        #         va="bottom",
        #         fontsize=8,
        #     )

    # ax.set_title("Heuristic Runtime")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.6)
    if USE_LOG_TIME:
        ax.set_yscale("log")

    # ax.legend(
    #     frameon=True,
    #     ncol=3,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.18),
    # )

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    fig.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Panel C (Anytime curve)
# ============================================================
def plot_panel_c(savepath):
    fig, ax = plt.subplots(figsize=(8, 6)) # Smaller, standard size

    for alg in algorithms_base:
        xs, ys = stepify_and_extend(pts_map[alg], T_END)
        ax.plot(xs, ys, color=COLORS[alg], linewidth=2.2, label=alg)

    # ax.set_title("Primal Heuristic performance (Bridge-1000)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tour cost")
    ax.set_xlim(1, 500)
    ax.set_ylim(800, 1500)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(500, color="black", linestyle=":", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.legend(
    #     frameon=True,
    #     ncol=1,
    #     # loc="upper center",
    #     # bbox_to_anchor=(0.5, -0.18),
    # )

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    fig.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Optional combined figure (kept close to your original layout)
# ============================================================
def plot_combined(savepath):
    # Panel A width depends on whether extra alg exists
    xA, bar_widthA, offsetsA = centered_bar_positions(len(experiments), len(algorithms_panelA))
    xB, bar_widthB, offsetsB = centered_bar_positions(len(experiments), len(algorithms_base))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.0, 4.8))

    # Panel A
    bars_by_alg = {}
    for i, alg in enumerate(algorithms_panelA):
        bars = ax1.bar(
            xA + offsetsA[i],
            scaled_values[alg],
            width=bar_widthA,
            color=COLORS[alg],
            edgecolor="black",
            linewidth=0.8,
            label=alg,
        )
        bars_by_alg[alg] = bars[0]
        for k, b in enumerate(bars):
            v = scaled_values[alg][k]
            ax1.annotate(
                f"{v:.1f}" if v >= 10 else f"{v:.3f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_title("Tour cost (CRISP ×1000)")
    ax1.set_ylabel("Tour cost")
    ax1.set_xticks(np.arange(len(experiments)))
    ax1.set_xticklabels(experiments)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel B
    for i, alg in enumerate(algorithms_base):
        bars = ax2.bar(
            xB + offsetsB[i],
            times[alg],
            width=bar_widthB,
            color=COLORS[alg],
            edgecolor="black",
            linewidth=0.8,
        )
        for k, b in enumerate(bars):
            ax2.annotate(
                f"{times[alg][k]:.2f}s",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax2.set_title("Heuristic Runtime")
    ax2.set_ylabel("Seconds")
    ax2.set_xticks(np.arange(len(experiments)))
    ax2.set_xticklabels(experiments)
    ax2.grid(axis="y", which="both", linestyle="--", alpha=0.6)
    if USE_LOG_TIME:
        ax2.set_yscale("log")

    # Panel C
    for alg in algorithms_base:
        xs, ys = stepify_and_extend(pts_map[alg], T_END)
        ax3.plot(xs, ys, color=COLORS[alg], linewidth=2.2)

    ax3.set_title("Primal Heuristic performance (Bridge-1000)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Incumbent objective")
    ax3.set_xlim(1, 500)
    ax3.set_ylim(800, 1500)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.axvline(500, color="black", linestyle=":", linewidth=1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Shared legend on top
    handles = [bars_by_alg[a] for a in algorithms_panelA]
    labels = algorithms_panelA
    ncol = 3 if len(labels) <= 3 else 2
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
        borderaxespad=0.2,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Run exports
# -----------------------------
plot_panel_a(os.path.join(output_dest, "PanelA_TourCost.pdf"))
plot_panel_b(os.path.join(output_dest, "PanelB_Runtime.pdf"))
plot_panel_c(os.path.join(output_dest, "PanelC_Anytime_Bridge1000.pdf"))
export_legend(os.path.join(output_dest, "Legend_Horizontal.pdf"))
if EXPORT_COMBINED_FIGURE:
    plot_combined(os.path.join(output_dest, "Combined_ABC.pdf"))

print("Done.")
