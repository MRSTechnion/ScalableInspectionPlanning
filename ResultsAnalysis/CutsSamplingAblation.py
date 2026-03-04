import matplotlib.pyplot as plt
import numpy as np
import os, sys
import Utils.paperstyle as ps


ps.use_paper_style()

experiments = ["Bridge-1000", "Bridge-2000", "Crisp-1000", "Crisp-2000"]

groups_sampling = [
    "50",
    "100",
    # "250",
    "500",
    "1000"
]

COLORS = {
    "50": 'palegreen',
    "100": "limegreen",
    # "250": "forestgreen",
    "500": "darkgreen",
    "1000": "darkolivegreen",
}

values = {
    groups_sampling[0]: [38.2, 54.6, 61.8, 74],
    groups_sampling[1]: [37.4, 52.6, 51.3, 71.7],
    # groups_sampling[2]: [32.9, 51.0, 51.7, 71.4],
    groups_sampling[2]: [39.3, 52.4, 52.2, 70.4],
    groups_sampling[3]: [38.0, 55.7, 52.3, 71.5]
}

x = np.arange(len(experiments))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(13, 6))

for i, alg in enumerate(groups_sampling):
    bars = ax.bar(
        x + i * bar_width,
        values[alg],
        width=bar_width,
        color=COLORS[alg],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.8,
        label=alg,
    )

    # Value annotations
    # for k, b in enumerate(bars):
    #     v = values[alg][k]
    #     ax.annotate(
    #         f"{v:.0f}" if v >= 10 else f"{v:.1f}",
    #         (b.get_x() + b.get_width() / 2, b.get_height()),
    #         xytext=(0, 3),
    #         textcoords="offset points",
    #         ha="center",
    #         va="bottom",
    #     )

# ax.set_ylabel("Optimality Gap [Percent]")
# ax.set_xlabel("Experiment")

ax.set_xticks(x + bar_width * (len(groups_sampling) - 1) / 2)
ax.set_xticklabels(experiments)
ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(
    title="Groups sample size:",
    # loc="upper left",
    frameon=True,
)
ax.set_ylim([0, 107])

output_dest = "/home/adir/Desktop/IP-results/pdf_results"
plt.savefig(os.path.join(output_dest, "GroupsSamplingAblation.pdf"), format="pdf", bbox_inches='tight')


plt.tight_layout()
plt.show()
print("Done.")
