import matplotlib.pyplot as plt

def use_paper_style():
    plt.rcParams.update({
        "figure.figsize": (3.4, 2.4),
        "font.size": 30,
        "axes.labelsize": 30,
        "legend.fontsize": 25,
        "lines.linewidth": 1.8,
        # "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "mathtext.fontset": "cm",
    })