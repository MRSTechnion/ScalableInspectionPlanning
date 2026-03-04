#!/usr/bin/env python3
"""
Make two compact 2x2 figures:
  1) Drone/Bridge family: (Drone1000, Drone2000)  -> ObjBounds + Gap
  2) Crisp family:        (Crisp1000, Crisp2000)  -> ObjBounds + Gap

Each 2x2 figure layout:
  Row 1: *-1000  (left=ObjBounds, right=Gap)
  Row 2: *-2000  (left=ObjBounds, right=Gap)

Saves:
  OUTPUT_DEST/Drone_2x2.pdf
  OUTPUT_DEST/Crisp_2x2.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import gurobi_logtools as glt

import Utils.paperstyle as ps

# ----------------- Paper style + PDF embedding -----------------
ps.use_paper_style()
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ----------------- Configuration -----------------
FILL_REGION = True  # optional shading between bounds
OUTPUT_DEST = "/home/adir/Desktop/IP-results/pdf_results"
LOG_DIR = "/home/adir/Desktop/IP-results/grb_logs_final"
TIME_LIMIT = 1000  # seconds
os.makedirs(OUTPUT_DEST, exist_ok=True)

# ----------------- loading -----------------
def load_nodelog(log_file: str) -> pd.DataFrame:
    _, timelines = glt.get_dataframe([log_file], timelines=True)
    return timelines["nodelog"].copy()

# ----------------- cleaning / alignment -----------------
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

    if sense == "min":
        nl["Incumbent_m"] = nl["Incumbent"].cummin()

        bd_clean = clean_bestbd_min(nl)
        nl["BestBd_m"] = bd_clean.cummax().ffill()

        # Optional clip of extreme spikes
        bd = nl["BestBd"].astype(float).copy()
        bd[~np.isfinite(bd)] = np.nan
        if bd.notna().any():
            cap = bd.quantile(0.95)
            bd = bd.clip(upper=cap)
            nl["BestBd_m"] = bd.cummax().ffill()
    else:
        nl["Incumbent_m"] = nl["Incumbent"].cummax()
        bd = nl["BestBd"].astype(float).copy()
        bd[~np.isfinite(bd)] = np.nan
        nl["BestBd_m"] = bd.cummin().ffill()

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

# ----------------- plotting helpers -----------------
def bounds_from_inc_bd(inc: np.ndarray, bd: np.ndarray, sense: str):
    # standard BnB terminology
    # min: upper=incumbent, lower=best bound
    # max: upper=best bound, lower=incumbent
    if sense == "min":
        return inc, bd
    return bd, inc

def plot_objective_bounds(ax, df_aligned: pd.DataFrame, label: str, color: str, sense: str):
    t = df_aligned["t"].to_numpy()
    inc = df_aligned["Incumbent_m"].to_numpy()
    bd = df_aligned["BestBd_m"].to_numpy()
    ub, lb = bounds_from_inc_bd(inc, bd, sense)

    ax.step(t, ub, where="post", color=color, linestyle="-",  linewidth=2.6, label=label)
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

# ----------------- experiment preparation -----------------
def prepare_experiment(experiment_name: str) -> dict:
    log_files = [
        os.path.join(LOG_DIR, f"Cutset_{experiment_name}_TL{TIME_LIMIT}.log"),
        os.path.join(LOG_DIR, f"SCF_{experiment_name}_TL{TIME_LIMIT}.log"),
        os.path.join(LOG_DIR, f"Charge_{experiment_name}_TL{TIME_LIMIT}.log"),
    ]

    legend_names = ["Group-Cutset", "SCF", "Charge"]
    file_colors = ['#1f77b4', '#2ca02c', 'red']

    processed, senses = {}, {}
    for lf in log_files:
        nl = load_nodelog(lf)
        sense = infer_sense(nl)
        senses[lf] = sense
        processed[lf] = enforce_monotone_progress(nl, sense=sense)

    aligned = align_on_common_time_grid(processed)

    # y-lims for your existing settings
    ylim_obj = None
    if experiment_name == "Drone1000":
        ylim_obj = (300, 1000)
    elif experiment_name == "Drone2000":
        ylim_obj = (300, 1200)
    elif experiment_name in ("Crisp1000", "Crisp2000"):
        ylim_obj = (0, 2)

    return {
        "name": experiment_name,
        "log_files": log_files,
        "legend_names": legend_names,
        "file_colors": file_colors,
        "senses": senses,
        "aligned": aligned,
        "ylim_obj": ylim_obj,
    }


def make_family_2x2(family_label: str, exp_1000: dict, exp_2000: dict, output_pdf: str):
    fig, axs = plt.subplots(
        2, 2,
        figsize=(12.0, 8.0),
        sharex="col",
        sharey=False
    )

    ax_cost_1000, ax_cost_2000 = axs[0, 0], axs[0, 1]
    ax_gap_1000, ax_gap_2000 = axs[1, 0], axs[1, 1]

    # --- Plotting helpers ---
    def draw_cost(ax, exp):
        for i, lf in enumerate(exp["log_files"]):
            c = exp["file_colors"][i % len(exp["file_colors"])]
            plot_objective_bounds(ax, exp["aligned"][lf], exp["legend_names"][i], c, exp["senses"][lf])
        if exp.get("ylim_obj") is not None:
            ax.set_ylim(exp["ylim_obj"])
        ax.set_xlim(0, TIME_LIMIT + 5)
        ax.grid(True, alpha=0.2)

    def draw_gap(ax, exp):
        for i, lf in enumerate(exp["log_files"]):
            c = exp["file_colors"][i % len(exp["file_colors"])]
            plot_gap(ax, exp["aligned"][lf], exp["legend_names"][i], c)
        ax.set_xlim(0, TIME_LIMIT + 5)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.2)

    draw_cost(ax_cost_1000, exp_1000)
    draw_cost(ax_cost_2000, exp_2000)
    draw_gap(ax_gap_1000, exp_1000)
    draw_gap(ax_gap_2000, exp_2000)

    # --- FONTS ---
    title_fs = 24
    label_fs = 22
    tick_fs = 18
    leg_fs = 19

    # Reduced title pad slightly so titles stay with their plots
    ax_cost_1000.set_title(f"{family_label}-1000", fontsize=title_fs, pad=12)
    ax_cost_2000.set_title(f"{family_label}-2000", fontsize=title_fs, pad=12)

    ax_cost_1000.set_ylabel("Cost", fontsize=label_fs)
    ax_gap_1000.set_ylabel("Gap (%)", fontsize=label_fs)
    ax_gap_1000.set_xlabel("Time (s)", fontsize=label_fs)
    ax_gap_2000.set_xlabel("Time (s)", fontsize=label_fs)

    for ax in axs.ravel():
        ax.tick_params(axis="both", labelsize=tick_fs)

        # --------- TWO-ROW LEGEND (METHODS OVER BOUNDS) ---------
        method_handles, method_labels = ax_gap_1000.get_legend_handles_labels()

        style_handles = [
            Line2D([0], [0], color="black", lw=2.5, linestyle="-"),
            Line2D([0], [0], color="black", lw=2.5, linestyle="--"),
        ]
        style_labels = [r"Upper bound ($c_{UB}$)", r"Lower bound ($c_{LB}$)"]

        # Invisible spacer to force row-wise appearance
        blank = Line2D([], [], linestyle="none")

        # Column-wise fill order:
        # Col 1: Group-Cutset | Upper
        # Col 2: SCF          | Lower
        # Col 3: Charge       | blank
        handles = [
            method_handles[0], style_handles[0],
            method_handles[1], style_handles[1],
            method_handles[2], blank,
        ]
        labels = [
            method_labels[0], style_labels[0],
            method_labels[1], style_labels[1],
            method_labels[2], "",
        ]

        # leg = fig.legend(
        #     handles,
        #     labels,
        #     loc="lower center",
        #     ncol=3,
        #     frameon=True,
        #     bbox_to_anchor=(0.5, 0.04),
        #     fontsize=leg_fs,
        #     handletextpad=0.6,
        #     columnspacing=2.2,
        #     alignment="center",
        # )
        #
        # # Hide the blank entry
        # for h, t in zip(leg.legend_handles, leg.texts):
        #     if t.get_text() == "":
        #         h.set_visible(False)
        #         t.set_visible(False)

    # --------- UPDATED SPACING ---------
    # top=0.82 provides the "tiny bit more space" from the upper legend
    # hspace=0.15 keeps the rows tight
    plt.subplots_adjust(
        left=0.10,
        right=0.96,
        top=0.86,
        bottom=0.26,  # room for two legend rows
        wspace=0.18,
        hspace=0.15
    )

    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# ----------------- main -----------------
def main():
    # Build experiments
    drone1000 = prepare_experiment("Drone1000")
    drone2000 = prepare_experiment("Drone2000")
    crisp1000 = prepare_experiment("Crisp1000")
    crisp2000 = prepare_experiment("Crisp2000")

    # Save two compact 2x2 PDFs
    drone_out = os.path.join(OUTPUT_DEST, "Drone_2x2.pdf")
    crisp_out = os.path.join(OUTPUT_DEST, "Crisp_2x2.pdf")

    make_family_2x2("Bridge", drone1000, drone2000, drone_out)
    make_family_2x2("Crisp",  crisp1000, crisp2000, crisp_out)

    print("Done. Saved:")
    print("  ", drone_out)
    print("  ", crisp_out)

if __name__ == "__main__":
    main()
