import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "qaoa": "#FF6B35",  # Vibrant orange
    "enumerate": "#00D9C0",  # Teal/cyan
    "grover": "#7B68EE",  # Medium slate blue
}

ALGORITHM_LABELS = {
    "qaoa": "QAOA",
    "enumerate": "Enumeration",
    "grover": "Grover's Algorithm",
}


def process_benchmark_data(data):
    processed = {}

    for key, trials in data.items():
        if isinstance(key, str):
            n, m = map(int, key.strip("()").split(","))
        else:
            n, m = key

        stats = {}
        for algo in ["qaoa", "enumerate", "grover"]:
            times = [trial[algo]["elapsed"] for trial in trials]
            solutions = [trial[algo]["solution"][1] for trial in trials]
            stats[algo] = {
                "times": times,
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "mean_objective": np.mean(solutions),
                "solutions": solutions,
            }

        processed[(n, m)] = stats

    return processed


def draw_header(fig, title, subtitle):
    """Draw a beautiful header with title and subtitle."""
    fig.text(
        0.5,
        0.97,
        title,
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.935,
        subtitle,
        fontsize=11,
        ha="center",
        va="top",
    )


def draw_legend(ax):
    """Draw a custom legend."""
    patches = [
        mpatches.Patch(color=COLORS["qaoa"], label="QAOA"),
        mpatches.Patch(color=COLORS["grover"], label="Grover's Algorithm"),
        mpatches.Patch(color=COLORS["enumerate"], label="Classical Enumeration"),
    ]
    ax.legend(
        handles=patches,
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
    )


def plot_time_comparison_bars(ax, processed_data):
    keys = sorted(processed_data.keys())
    x_labels = [f"({n},{m})" for n, m in keys]
    x = np.arange(len(keys))
    width = 0.25

    qaoa_times = [processed_data[k]["qaoa"]["mean_time"] * 1000 for k in keys]
    grover_times = [processed_data[k]["grover"]["mean_time"] * 1000 for k in keys]
    enum_times = [processed_data[k]["enumerate"]["mean_time"] * 1000 for k in keys]

    # Error bars
    qaoa_err = [processed_data[k]["qaoa"]["std_time"] * 1000 for k in keys]
    grover_err = [processed_data[k]["grover"]["std_time"] * 1000 for k in keys]
    enum_err = [processed_data[k]["enumerate"]["std_time"] * 1000 for k in keys]

    _bars1 = ax.bar(
        x - width,
        qaoa_times,
        width,
        label="QAOA",
        color=COLORS["qaoa"],
        alpha=0.9,
        yerr=qaoa_err,
        capsize=3,
        error_kw={"elinewidth": 1, "capthick": 1, "alpha": 0.5},
    )
    _bars2 = ax.bar(
        x,
        grover_times,
        width,
        label="Grover's",
        color=COLORS["grover"],
        alpha=0.9,
        yerr=grover_err,
        capsize=3,
        error_kw={"elinewidth": 1, "capthick": 1, "alpha": 0.5},
    )
    _bars3 = ax.bar(
        x + width,
        enum_times,
        width,
        label="Enumeration",
        color=COLORS["enumerate"],
        alpha=0.9,
        yerr=enum_err,
        capsize=3,
        error_kw={"elinewidth": 1, "capthick": 1, "alpha": 0.5},
    )

    ax.set_xlabel("Problem Size (n, m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Average Execution Time by Problem Size",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    draw_legend(ax)


def visualize_benchmark_results(data, output_path="benchmark_results.png", dpi=150):
    processed = process_benchmark_data(data)
    fig = plt.figure(figsize=(19.20, 10.80))
    draw_header(
        fig, "Comparing QAOA, Grover's Algorithm, and Classical Enumeration", ""
    )
    ax = fig.add_subplot(2, 3, (1, 3))
    plot_time_comparison_bars(ax, processed)

    # Save figure
    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )
    print(f"Visualization saved to: {output_path}")
    return fig


if __name__ == "__main__":
    with open("results.dict") as f:
        dict_str = f.read()
    data = eval(dict_str)
    visualize_benchmark_results(data)
