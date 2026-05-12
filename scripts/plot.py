import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Replace these with your current apt-get motivation raw runtimes
# unit can be ms / s, just keep it consistent
# --------------------------------------------------
benchmarks_top = ["BFS", "BC", "PR", "CC-SV", "CC", "TC", "SSSP", "PR-SpMV", "Spatter-G", "Spatter-AM", "Spatter-L"]
benchmarks_bottom = ["Spatter-SH", "Spatter-NE", "Spatter-P", "NPB-IS", "NPB-EP", "NPB-CG", "NPB-MG", "NPB-FT", "NPB-BT", "MCF"]

data = {
    "Host":              [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
                          1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "Device-only":       [0.55, 0.45, 0.40, 0.50, 0.35, 0.40, 0.60, 0.65, 0.50, 0.40, 0.45,
                          0.55, 1.36, 1.30, 1.42, 1.46, 1.39, 1.37, 1.40, 1.34, 0.49],
    "Prefetch-only":     [1.60, 1.35, 1.45, 1.55, 1.40, 1.30, 1.65, 1.70, 1.55, 1.45, 1.50,
                          1.60, 1.48, 1.38, 1.55, 1.61, 1.50, 1.49, 1.53, 1.47, 1.59],
    "CIRA without PGO":  [1.50, 1.40, 1.35, 1.45, 1.30, 1.35, 1.55, 1.60, 1.45, 1.35, 1.40,
                          1.50, 1.41, 1.33, 1.47, 1.52, 1.44, 1.43, 1.45, 1.39, 1.50],
    "CIRA":              [1.65, 1.50, 1.55, 1.60, 1.45, 1.50, 1.70, 1.75, 1.60, 1.45, 1.55,
                          1.65, 1.57, 1.46, 1.66, 1.73, 1.62, 1.58, 1.64, 1.55, 1.71],
}

# --------------------------------------------------
# If the above are currently speedups and you want raw runtime,
# convert using host raw runtime:
# runtime_mode = runtime_host / speedup_mode
# Replace host_runtime with your actual apt-get host raw runtime later
# --------------------------------------------------
host_runtime = np.array([
    122.199, 273.089, 35.020, 104.649, 25.464, 908.801, 183.285, 179.913, 35.0, 59.0, 138.0,
    210.0, 165.0, 520.0, 610.0, 455.0, 780.0, 735.0, 690.0, 845.0, 300.0
])

runtime_data = {}
for k, vals in data.items():
    vals = np.array(vals, dtype=float)
    if k == "Host":
        runtime_data[k] = host_runtime.copy()
    else:
        runtime_data[k] = host_runtime / vals

def gmean(arr):
    arr = np.asarray(arr, dtype=float)
    return np.exp(np.mean(np.log(arr)))

modes = ["Host", "Device-only", "Prefetch-only", "CIRA without PGO", "CIRA"]
colors = ["lightgray", "#f7931e", "#2ca02c", "#1f77b4", "#d62728"]

top_n = len(benchmarks_top)
bottom_n = len(benchmarks_bottom)

fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharey=False)
width = 0.15

for ax, names, start, end in [
    (axes[0], benchmarks_top, 0, top_n),
    (axes[1], benchmarks_bottom, top_n, top_n + bottom_n),
]:
    x = np.arange(len(names))

    for i, (mode, color) in enumerate(zip(modes, colors)):
        vals = runtime_data[mode][start:end]
        ax.bar(x + (i - 2) * width, vals, width=width, label=mode, color=color, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Runtime")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

# add runtime geomean block on bottom panel
ax = axes[1]
gap_x = len(benchmarks_bottom) + 1.2
for i, (mode, color) in enumerate(zip(modes, colors)):
    gm = gmean(runtime_data[mode])
    ax.bar(gap_x + (i - 2) * width, gm, width=width, color=color, edgecolor="black", linewidth=0.8)
    ax.text(gap_x + (i - 2) * width, gm, f"{gm:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

xticks = list(np.arange(len(benchmarks_bottom))) + [gap_x]
xlabels = benchmarks_bottom + ["GeoMean"]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=9)

fig.suptitle("Raw Runtime Across Execution Modes (Apt-get Motivation)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("apt_get_runtime_raw.pdf", dpi=300, bbox_inches="tight")
plt.show()