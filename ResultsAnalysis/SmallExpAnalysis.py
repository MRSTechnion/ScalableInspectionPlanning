#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import Utils.paperstyle as ps

# ==========================================
# Paper style + PDF embedding
# ==========================================
ps.use_paper_style()
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUTPUT_DEST = "/home/adir/Desktop/IP-results/pdf_results"
os.makedirs(OUTPUT_DEST, exist_ok=True)

# ==========================================
# 1. DATA & STYLE CONFIGURATION
# ==========================================
K_VALUES = [100, 250, 350, 500]

DATA = {
    250: {
        "Charge": [64.3, 55.3, 46.1, 51.2],
        "SCF": [62.4, 53.4, 45.7, 46.4],  # Shortened for consistency
        "Group-Cutset": [41.2, 33.7, 27.3, 26.8],
        "MCF": [0, 0, 0, 100],
    },
    500: {
        "Charge": [65.1, 63.5, 58.4, 54.4],
        "SCF": [58.3, 55, 52.6, 49.1],
        "Group-Cutset": [57.1, 32.1, 26.5, 30.4],
        "MCF": [0, 100, 100, 100],
    },
    750: {
        "Charge": [76.1, 70, 65, 64.6],
        "SCF": [70.7, 66.8, 57.1, 57.7],
        "Group-Cutset": [34.8, 36.6, 39.4, 40.5],
        "MCF": [100, 100, 100, 100],
    },
    1000: {
        "Charge": [78, 73.4, 74.6, 75.8],
        "SCF": [76.2, 75.8, 68.1, 64.3],
        "Group-Cutset": [39, 43.5, 39.4, 42.2],
        "MCF": [100, 100, 100, 100],
    }
}

STYLES = {
    "SCF": {"color": "#2ca02c", "marker": "o", "linestyle": "-"},
    "MCF": {"color": "fuchsia", "marker": "o", "linestyle": "-"},
    "Charge": {"color": "red", "marker": "o", "linestyle": "-"},
    "Group-Cutset": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
}

X_LABEL = "k"
Y_LABEL = "Optimality Gap (%)"


# ==========================================
# 2. PLOTTING INDIVIDUAL FIGURES
# ==========================================
def plot_individual_results_small():
    first_n = list(DATA.keys())[0]

    for n_val, methods in DATA.items():
        # Fixed figsize for consistency
        fig, ax = plt.subplots(figsize=(7, 8))

        for method_name, vals in methods.items():
            style = STYLES.get(method_name, {"color": "black", "marker": "x"})
            ax.plot(
                K_VALUES, vals,
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=3.0,
                markersize=8.0,
            )

        # Formatting
        ax.set_xlabel(X_LABEL)
        ax.set_ylim(-5, 105)
        ax.set_xticks(K_VALUES)
        ax.tick_params(axis="x", labelsize=30)
        ax.grid(True, linestyle="--", alpha=0.35)

        # --- Y-Axis Logic for Identical Sizing ---
        ax.set_ylabel(Y_LABEL)
        if n_val == first_n:
            ax.tick_params(axis="y", labelsize=30)
        else:
            # Reserve space but hide text
            ax.yaxis.label.set_color('none')
            ax.tick_params(axis="y", labelcolor='none')

        out_path = os.path.join(OUTPUT_DEST, f"Gap_vs_K_Small_N{n_val}.pdf")
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    # --- Generate Shared Legend ---
    fig_leg = plt.figure(figsize=(22, 2.0))
    # Match the order you prefer in the legend
    legend_order = ["Group-Cutset", "SCF", "MCF", "Charge"]

    handles = [
        Line2D([0], [0], color=STYLES[m]["color"], marker='o', lw=6.0, markersize=12)
        for m in legend_order
    ]

    leg = fig_leg.legend(
        handles, legend_order,
        loc="center",
        ncol=4,
        columnspacing=3.0,
        handlelength=4.0,
        fontsize=35,
        frameon=True,
        edgecolor='black'
    )
    if leg:
        leg.get_frame().set_linewidth(2.0)

    out_leg = os.path.join(OUTPUT_DEST, "Gap_vs_K_Small_Legend.pdf")
    fig_leg.savefig(out_leg, format="pdf", bbox_inches="tight")
    plt.close(fig_leg)
    print("Individual plots and shared legend saved.")


if __name__ == "__main__":
    plot_individual_results_small()