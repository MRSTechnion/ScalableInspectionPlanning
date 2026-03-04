#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
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
# 1. DATA CONFIGURATION
# ==========================================
K_VALUES = [1000, 2500, 5000, 7500]

DATA = {
    3500: {
        "Charge": [86.6, 84.3, 81.7, 81.6],
        "Single Commodity Flow": [79.6, 76.8, 75.4, 75.2],
        "Group-Cutset": [48.8, 52.7, 51.7, 53.5],
    },
    5000: {
        "Charge": [88.1, 87.0, 84.9, 84.1],
        "Single Commodity Flow": [86.6, 84.8, 83.2, 83.1],
        "Group-Cutset": [55.8, 61.8, 60.0, 60.6],
    },
    10000: {
        "Charge": [94.7, 95.5, 95.2, 95.2],
        "Single Commodity Flow": [94.5, 95.1, 94.9, 94.8],
        "Group-Cutset": [68.9, 72.3, 75.2, 77.3],
    },
    15000: {
        "Charge": [98.2, 95.8, 96.0, 96.6],
        "Single Commodity Flow": [96.8, 96.5, 96.6, 96.6],
        "Group-Cutset": [75.5, 78.2, 83.6, 84.3],
    },
}

STYLES = {
    "Charge": {"color": 'red', "marker": "o", "linestyle": "-"},
    "Single Commodity Flow": {"color": "#2ca02c", "marker": "o", "linestyle": "-"},
    "Group-Cutset": {"color": '#1f77b4', "marker": "o", "linestyle": "-"},
}

X_LABEL = "k"
Y_LABEL = "Optimality Gap (%)"

# ==========================================
# 2. PLOTTING INDIVIDUAL FIGURES
# ==========================================
def plot_individual_results():
    # Identify the first key to handle Y-axis logic
    first_n = list(DATA.keys())[0]

    for n_val, methods in DATA.items():
        # Create a single figure for each N
        fig, ax = plt.subplots(figsize=(7, 8))

        for method_name, values in methods.items():
            style = STYLES.get(method_name, {"color": "black", "marker": "x", "linestyle": "-"})
            ax.plot(
                K_VALUES, values,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=3.0,      # Slightly thicker for individual plots
                markersize=8.0,     # Larger markers for clarity
            )

        # Basic formatting
        ax.set_xlabel(X_LABEL)
        ax.set_ylim(40, 100)
        ax.set_xticks(K_VALUES)
        ax.tick_params(axis="x", labelsize=30)
        ax.grid(True, linestyle="--", alpha=0.35)

        # --- Y-Axis Logic for Identical Sizing ---
        ax.set_ylabel(Y_LABEL)
        ax.set_ylim(40, 100)

        if n_val == first_n:
            ax.tick_params(axis="y", labelsize=30)
            # Label is visible by default
        else:
            # Make the label and tick numbers invisible but keep the SPACE reserved
            ax.yaxis.label.set_color('none')
            ax.tick_params(axis="y", labelcolor='none')

        # Save without title and legend
        out_path = os.path.join(OUTPUT_DEST, f"Gap_vs_K_N{n_val}.pdf")
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_individual_results()