import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import gurobi_logtools as glt

import Utils.paperstyle as ps
from matplotlib.lines import Line2D

# ---------- Paper style + PDF embedding ----------
ps.use_paper_style()
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ---------- Configuration ----------
FILL_REGION = True  # optional shading between bounds
OUTPUT_DEST = "/home/adir/Desktop/IP-results/pdf_results"
os.makedirs(OUTPUT_DEST, exist_ok=True)

# -----------------------------------
# ---------- loading ----------
def load_nodelog(log_file: str) -> pd.DataFrame:
    _, timelines = glt.get_dataframe([log_file], timelines=True)
    nl = timelines["nodelog"].copy()
    return nl

# ---------- cleaning / alignment ----------
def infer_sense(nodelog: pd.DataFrame) -> str:
    nl = nodelog.dropna(subset=["Incumbent", "BestBd"])
    if nl.empty:
        return "min"
    last = nl.iloc[-1]
    inc, bd = float(last["Incumbent"]), float(last["BestBd"])
    # If BestBd <= Incumbent at end, typical minimization
    if bd <= inc:
        return "min"
    if bd >= inc:
        return "max"
    return "min"

def clean_bestbd_min(nl: pd.DataFrame) -> pd.Series:
    bd = nl["BestBd"].astype(float).copy()
    inc = nl["Incumbent"].astype(float).copy()
    mask_inc = inc.notna()
    bd[mask_inc & (bd > inc)] = np.nan
    bd[~np.isfinite(bd)] = np.nan
    return bd

def enforce_monotone_progress(nodelog: pd.DataFrame, sense: str) -> pd.DataFrame:
    nl = nodelog.copy()
    nl = nl.sort_values("Time", kind="mergesort")
    nl = nl.drop_duplicates(subset=["Time"], keep="last").reset_index(drop=True)

    for c in ["Time", "Incumbent", "BestBd", "Gap"]:
        if c in nl.columns:
            nl[c] = pd.to_numeric(nl[c], errors="coerce")

    t0 = nl["Time"].min()
    nl["t"] = nl["Time"] - (t0 if pd.notna(t0) else 0.0)

    # Monotone tracks
    if sense == "min":
        nl["Incumbent_m"] = nl["Incumbent"].cummin()

        bd_clean = clean_bestbd_min(nl)
        # BestBd for min should be non-decreasing (tightening LB)
        nl["BestBd_m"] = bd_clean.cummax().ffill()

        # (Optional) clip extreme bound spikes
        bd = nl["BestBd"].astype(float).copy()
        bd[~np.isfinite(bd)] = np.nan
        cap = bd.quantile(0.95)
        bd = bd.clip(upper=cap)
        nl["BestBd_m"] = bd.cummax().ffill()

    else:  # max
        nl["Incumbent_m"] = nl["Incumbent"].cummax()
        bd = nl["BestBd"].astype(float).copy()
        bd[~np.isfinite(bd)] = np.nan
        # BestBd for max should be non-increasing (tightening UB)
        nl["BestBd_m"] = bd.cummin().ffill()

    # Recompute a clean gap
    eps = 1e-12
    inc = nl["Incumbent_m"]
    bd = nl["BestBd_m"]
    denom = inc.abs().clip(lower=eps)
    nl["Gap_m"] = ((inc - bd).abs() / denom) * 100.0
    nl["Gap_m"] = nl["Gap_m"].fillna(100.0)

    return nl

def align_on_common_time_grid(processed_by_file: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    all_t = np.unique(
        np.concatenate([df["t"].dropna().to_numpy() for df in processed_by_file.values()])
    )
    all_t.sort()

    aligned = {}
    for lf, df in processed_by_file.items():
        tmp = df.set_index("t")[["Incumbent_m", "BestBd_m", "Gap_m"]].sort_index()
        tmp = tmp.reindex(all_t).ffill()
        tmp = tmp.reset_index().rename(columns={"index": "t"})
        aligned[lf] = tmp

    return aligned

# ---------- plotting helpers ----------
def nice_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def bounds_from_inc_bd(inc: np.ndarray, bd: np.ndarray, sense: str):
    """
    Returns (upper_bound, lower_bound) arrays following standard BnB terminology:
      - minimization: upper = incumbent, lower = best bound
      - maximization: upper = best bound, lower = incumbent
    """
    if sense == "min":
        ub = inc
        lb = bd
    else:
        ub = bd
        lb = inc
    return ub, lb

def plot_objective_bounds(ax, df_aligned: pd.DataFrame, label: str, color: str, sense: str):
    t = df_aligned["t"].to_numpy()
    inc = df_aligned["Incumbent_m"].to_numpy()
    bd = df_aligned["BestBd_m"].to_numpy()
    ub, lb = bounds_from_inc_bd(inc, bd, sense)

    # Requested styles:
    #   Upper bound: solid
    #   Lower bound: dashed
    ax.step(t, ub, where="post", color=color, linestyle="-",  linewidth=2.6, label=f"{label}")
    ax.step(t, lb, where="post", color=color, linestyle="--", linewidth=2.6, label="_nolegend_")

    if FILL_REGION:
        lower = np.minimum(ub, lb)
        upper = np.maximum(ub, lb)
        ax.fill_between(t, lower, upper, step="post", color=color, alpha=0.12, label="_nolegend_")

def plot_gap(ax, df_aligned: pd.DataFrame, label: str, color: str):
    t = df_aligned["t"].to_numpy()
    gap = df_aligned["Gap_m"].to_numpy()

    stride = max(1, len(t) // 15)
    ax.step(
        t, gap, where="post",
        color=color, linestyle="-", linewidth=2.4,
        marker="o", markersize=6, markevery=stride,
        label=label
    )

if __name__ == "__main__":
    from matplotlib.lines import Line2D

    for Experiment in ["Drone1000", "Drone2000", "Crisp1000", "Crisp2000"]:

        log_files = [
            f"/home/adir/Desktop/IP-results/grb_logs_final/Cutset_{Experiment}_TL1000.log",
            f"/home/adir/Desktop/IP-results/grb_logs_final/SCF_{Experiment}_TL1000.log",
            f"/home/adir/Desktop/IP-results/grb_logs_final/Charge_{Experiment}_TL1000.log",
        ]

        legend_names = ["Group-Cutset", "SCF", "Charge"]
        file_colors = ['#1f77b4', '#2ca02c', 'red']

        # ---------- Load + preprocess ----------
        processed = {}
        senses = {}
        for lf in log_files:
            nl = load_nodelog(lf)
            sense = infer_sense(nl)
            senses[lf] = sense
            processed[lf] = enforce_monotone_progress(nl, sense=sense)

        aligned = align_on_common_time_grid(processed)

        # ---------- Common axis settings ----------
        def apply_obj_axes(ax):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Cost")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1005)
            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)

            if Experiment == "Drone1000":
                ax.set_ylim(400, 1000)
            elif Experiment == "Drone2000":
                ax.set_ylim(400, 1200)
            elif Experiment in ["Crisp1000", "Crisp2000"]:
                ax.set_ylim(0, 2)

        def apply_gap_axes(ax):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Gap (%)")
            ax.set_ylim(0, 105)
            ax.set_xlim(0, 1005)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)

        # ---------- 1) One PDF per scenario (Objective+Bounds) ----------
        for i, lf in enumerate(log_files):
            fig_obj, ax_obj = plt.subplots(figsize=(13, 9))
            c = file_colors[i % len(file_colors)]

            plot_objective_bounds(ax_obj, aligned[lf], legend_names[i], c, senses[lf])
            apply_obj_axes(ax_obj)

            # IMPORTANT: no legend on the per-scenario figures
            fig_obj.tight_layout()
            out = os.path.join(OUTPUT_DEST, f"{Experiment}_{legend_names[i]}_ObjBounds.pdf")
            fig_obj.savefig(out, format="pdf", bbox_inches="tight")
            plt.close(fig_obj)

        # ---------- (Optional) One PDF per scenario (Gap) ----------
        # If you also want 3 separate gap PDFs, keep this block; otherwise remove it.
        for i, lf in enumerate(log_files):
            fig_gap, ax_gap = plt.subplots(figsize=(13, 9))
            c = file_colors[i % len(file_colors)]

            plot_gap(ax_gap, aligned[lf], legend_names[i], c)
            apply_gap_axes(ax_gap)

            fig_gap.tight_layout()
            out = os.path.join(OUTPUT_DEST, f"{Experiment}_{legend_names[i]}_Gap.pdf")
            fig_gap.savefig(out, format="pdf", bbox_inches="tight")
            plt.close(fig_gap)

        method_handles = [
            Line2D([0], [0], color=file_colors[i], lw=2.6, linestyle="-")
            for i in range(len(legend_names))
        ]
        style_handles = [
            Line2D([0], [0], color="black", lw=2.6, linestyle="-"),
            Line2D([0], [0], color="black", lw=2.6, linestyle="--"),
        ]
        style_labels = [r"Upper bound ($c_{UB}$)", r"Lower bound ($c_{LB}$)"]

        # ----- Method-Only Legend (Enlarged) -----
        fig_leg = plt.figure(figsize=(12, 1.2))  # Adjusted width for 3 items
        fig_leg.patch.set_alpha(0.0)

        # Only use the method handles and names
        # Ensure 'lw' is high so the lines are visible even when large
        method_handles = [
            Line2D([0], [0], color=file_colors[i], lw=4.0, linestyle="-")
            for i in range(len(legend_names))
        ]

        fig_leg.legend(
            method_handles,
            legend_names,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=fig_leg.transFigure,
            frameon=True,
            ncol=len(legend_names),  # Exactly 3 columns
            columnspacing=2.5,  # Increased spacing for readability
            handlelength=3.0,  # Longer lines
            fontsize=30,  # Large font for visibility
            borderpad=0.6,
            edgecolor='black'
        )

        out_leg = os.path.join(OUTPUT_DEST, f"{Experiment}_Method_Legend.pdf")
        fig_leg.savefig(out_leg, format="pdf", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_leg)

        # ----- Enlarged Method-Only Legend -----
        fig_leg = plt.figure(figsize=(20, 2.0))
        fig_leg.patch.set_alpha(0.0)

        # Increase 'lw' here to make the lines inside the legend thicker
        method_handles_large = [
            Line2D([0], [0], color=file_colors[i], lw=6.0, linestyle="-")
            for i in range(len(legend_names))
        ]

        leg = fig_leg.legend(
            method_handles_large,
            legend_names,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=fig_leg.transFigure,
            frameon=True,
            ncol=len(legend_names),
            columnspacing=3.0,
            handlelength=4.5,
            fontsize=40,
            borderpad=1.0,
            edgecolor='black'  # This sets the border color
        )

        # To enlarge the border thickness (the frame), do this:
        if leg:
            frame = leg.get_frame()
            frame.set_linewidth(2.5)  # This is the correct way to set border thickness

        out_leg = os.path.join(OUTPUT_DEST, f"{Experiment}_Method_Legend_Enlarged.pdf")
        fig_leg.savefig(out_leg, format="pdf", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig_leg)
        print("Done. Saved PDFs to:", OUTPUT_DEST)

