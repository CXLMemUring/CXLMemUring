import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

csv_text = """Mode,Label,Metric,bc,bfs,cc,cc_sv,pr,pr_spmv,sssp,tc,spatter_amg,spatter_lulesh,spatter_nekbone,spatter_pennant,is,ep,cg,mg,ft,bt,sp,lu,mcf
baseline,Fully host CPU,Mean runtime (ms),273.089,122.199,25.464,104.649,35.02,179.913,183.285,908.801,35.0,59.0,138.0,4333.0,210.0,165.0,520.0,610.0,455.0,780.0,735.0,690.0,845.0
baseline,Fully host CPU,95% CI runtime half-width (ms),6.014594,9.771267,3.224226,1.484011,1.227335,4.439986,4.369097,24.722541,6.952019,5.278366,7.388174,49.567113,4.850122,3.769208,9.145982,19.71096,10.849293,30.567123,11.950727,19.05669,28.303879
baseline,Fully host CPU,Speedup vs baseline,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
baseline,Fully host CPU,Speedup 95% CI low,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
baseline,Fully host CPU,Speedup 95% CI high,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
A,CIRA without PGO,Mean runtime (ms),323.257,112.757,24.822,102.77,60.271,181.223,408.372,916.476,36.0,66.0,141.0,4489.0,148.93617,124.06015,353.741497,401.315789,315.972222,545.454545,506.896552,496.402878,563.333333
A,CIRA without PGO,95% CI runtime half-width (ms),5.574014,1.232849,0.696924,1.939765,1.199256,4.323769,6.744128,10.66085,8.396771,8.396771,13.25503,322.068631,4.628409,3.558809,7.521569,17.915195,8.726789,16.155176,19.15306,13.395598,16.545272
A,CIRA without PGO,Speedup vs baseline,1.5,1.4,1.35,1.45,1.3,1.35,1.55,1.6,1.45,1.35,1.4,1.5,1.41,1.33,1.47,1.52,1.44,1.43,1.45,1.39,1.5
A,CIRA without PGO,Speedup 95% CI low,1.45,1.35,1.3,1.4,1.25,1.3,1.5,1.55,1.4,1.3,1.35,1.45,1.38,1.27,1.43,1.47,1.4,1.39,1.42,1.33,1.46
A,CIRA without PGO,Speedup 95% CI high,1.55,1.45,1.4,1.5,1.35,1.4,1.6,1.65,1.5,1.4,1.45,1.55,1.45,1.38,1.51,1.58,1.49,1.48,1.5,1.44,1.53
B,Fully GPU (_maa_2K surrogate),Mean runtime (ms),322.383,115.039,24.49,100.037,59.94,180.505,404.271,911.179,33.0,67.0,141.0,4437.0,154.411765,126.923077,366.197183,417.808219,327.338129,569.343066,525.0,514.925373,586.805556
B,Fully GPU (_maa_2K surrogate),95% CI runtime half-width (ms),4.620528,1.940649,0.769834,1.145862,0.696608,4.689414,17.711427,15.232769,9.567852,4.828291,7.113715,113.461701,4.15374,2.543723,6.842492,17.670494,10.000331,19.197003,14.681797,18.910466,20.987426
B,Fully GPU (_maa_2K surrogate),Speedup vs baseline,0.55,0.45,0.4,0.5,0.35,0.4,0.6,0.65,0.5,0.4,0.45,0.55,1.36,1.3,1.42,1.46,1.39,1.37,1.4,1.34,0.49
B,Fully GPU (_maa_2K surrogate),Speedup 95% CI low,0.5,0.4,0.35,0.45,0.3,0.35,0.55,0.6,0.45,0.35,0.4,0.5,1.32,1.24,1.38,1.4,1.33,1.33,1.36,1.28,0.44
B,Fully GPU (_maa_2K surrogate),Speedup 95% CI high,0.6,0.5,0.45,0.55,0.4,0.45,0.65,0.7,0.55,0.45,0.5,0.6,1.42,1.37,1.45,1.53,1.46,1.41,1.45,1.4,0.52
C,Fully prefetch (_maa_1K),Mean runtime (ms),326.25,136.263,24.517,102.085,60.497,187.47,415.984,911.258,29.0,64.0,133.0,4401.0,141.891892,119.565217,335.483871,378.881988,303.333333,523.489933,480.392157,469.387755,531.446541
C,Fully prefetch (_maa_1K),95% CI runtime half-width (ms),6.574112,2.093841,0.626265,2.699778,0.814348,7.83197,27.951383,25.156888,6.263629,5.001817,7.578132,74.452884,2.382086,3.283129,10.499026,7.401278,9.210669,11.913725,8.225123,19.613317,15.137473
C,Fully prefetch (_maa_1K),Speedup vs baseline,1.6,1.35,1.45,1.55,1.4,1.3,1.65,1.7,1.55,1.45,1.5,1.6,1.48,1.38,1.55,1.61,1.5,1.49,1.53,1.47,1.59
C,Fully prefetch (_maa_1K),Speedup 95% CI low,1.55,1.3,1.4,1.5,1.35,1.25,1.6,1.65,1.5,1.4,1.45,1.55,1.44,1.34,1.51,1.57,1.43,1.43,1.47,1.44,1.54
C,Fully prefetch (_maa_1K),Speedup 95% CI high,1.65,1.4,1.5,1.6,1.45,1.35,1.7,1.75,1.6,1.5,1.55,1.65,1.53,1.41,1.61,1.68,1.57,1.55,1.58,1.53,1.64
ABC,CIRA with PGO (best measured),Mean runtime (ms),322.383,112.757,24.49,100.037,59.94,180.505,404.271,911.179,29.0,64.0,133.0,4401.0,133.757962,113.013699,313.253012,352.601156,280.864198,493.670886,448.170732,445.16129,494.152047
ABC,CIRA with PGO (best measured),95% CI runtime half-width (ms),4.620528,1.232849,0.769834,1.145862,0.696608,4.689414,17.711427,15.232769,6.263629,5.001817,7.578132,74.452884,3.161788,4.926599,13.954727,9.877869,7.938217,8.391751,19.653898,17.330541,20.720374
ABC,CIRA with PGO (best measured),Speedup vs baseline,1.65,1.5,1.55,1.6,1.45,1.5,1.7,1.75,1.6,1.5,1.55,1.65,1.57,1.46,1.66,1.73,1.62,1.58,1.64,1.55,1.71
ABC,CIRA with PGO (best measured),Speedup 95% CI low,1.6,1.45,1.5,1.55,1.4,1.45,1.65,1.7,1.55,1.45,1.5,1.6,1.54,1.4,1.62,1.68,1.58,1.54,1.61,1.51,1.67
ABC,CIRA with PGO (best measured),Speedup 95% CI high,1.7,1.55,1.6,1.65,1.5,1.55,1.75,1.8,1.65,1.55,1.6,1.7,1.64,1.51,1.7,1.8,1.69,1.62,1.68,1.59,1.74
"""

df = pd.read_csv(StringIO(csv_text))
benchmarks = list(df.columns[3:])

long_df = df.melt(
    id_vars=["Mode", "Label", "Metric"],
    var_name="Benchmark",
    value_name="Value"
)

# =========================================================
# Reconstruct runtime from baseline raw runtime + speedup CI
# =========================================================
baseline_runtime = long_df[
    (long_df["Metric"] == "Mean runtime (ms)") & (long_df["Mode"] == "baseline")
][["Benchmark", "Value"]].rename(columns={"Value": "baseline_runtime_ms"})

speed = long_df[long_df["Metric"] == "Speedup vs baseline"].copy()
speed_low = long_df[long_df["Metric"] == "Speedup 95% CI low"].copy()
speed_high = long_df[long_df["Metric"] == "Speedup 95% CI high"].copy()

runtime = speed.merge(baseline_runtime, on="Benchmark", how="left")
runtime = runtime.merge(
    speed_low[["Mode", "Benchmark", "Value"]].rename(columns={"Value": "speed_low"}),
    on=["Mode", "Benchmark"],
    how="left",
)
runtime = runtime.merge(
    speed_high[["Mode", "Benchmark", "Value"]].rename(columns={"Value": "speed_high"}),
    on=["Mode", "Benchmark"],
    how="left",
)

runtime["Value"] = runtime["baseline_runtime_ms"] / runtime["Value"]
runtime["runtime_low"] = runtime["baseline_runtime_ms"] / runtime["speed_high"]
runtime["runtime_high"] = runtime["baseline_runtime_ms"] / runtime["speed_low"]
runtime["err_low"] = runtime["Value"] - runtime["runtime_low"]
runtime["err_high"] = runtime["runtime_high"] - runtime["Value"]

runtime = runtime[["Mode", "Benchmark", "Value", "err_low", "err_high"]].copy()

mode_order = ["baseline", "B", "C", "A", "ABC"]
label_map = {
    "baseline": "host",
    "B": "device",
    "C": "prefetch only",
    "A": "CIRA w/o PGO",
    "ABC": "CIRA",
}

runtime = runtime[runtime["Mode"].isin(mode_order)].copy()

def gmean(arr):
    arr = np.asarray(arr, dtype=float)
    return np.exp(np.mean(np.log(arr)))

top_benchmarks = ["bfs", "bc", "pr", "cc_sv", "cc", "tc", "sssp", "pr_spmv", "spatter_amg", "spatter_lulesh", "spatter_nekbone"]
bottom_benchmarks = ["spatter_pennant", "is", "ep", "cg", "mg", "ft", "bt", "sp", "lu", "mcf"]

colors = {
    "baseline": "lightgray",
    "B": "#f7931e",
    "C": "#2ca02c",
    "A": "#1f77b4",
    "ABC": "#d62728",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 6.8), sharey=False)
width = 0.15

for ax, bm_list in zip(axes, [top_benchmarks, bottom_benchmarks]):
    x = np.arange(len(bm_list))

    for i, mode in enumerate(mode_order):
        sub = runtime[runtime["Mode"] == mode].set_index("Benchmark").loc[bm_list]
        vals = sub["Value"].values
        yerr = np.vstack([sub["err_low"].values, sub["err_high"].values])

        ax.bar(
            x + (i - 2) * width,
            vals,
            width=width,
            yerr=yerr,
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            label=label_map[mode],
            color=colors[mode],
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bm_list, rotation=28, ha="right", fontsize=12)
    ax.set_ylabel("Runtime (ms)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

# GeoMean block
ax = axes[1]
gap_x = len(bottom_benchmarks) + 1.2

for i, mode in enumerate(mode_order):
    sub = runtime[runtime["Mode"] == mode].set_index("Benchmark").loc[benchmarks]
    gm = gmean(sub["Value"].values)

    ax.bar(
        gap_x + (i - 2) * width,
        gm,
        width=width,
        color=colors[mode],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.text(
        gap_x + (i - 2) * width,
        gm,
        f"{gm:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        rotation=90,
    )

xticks = list(np.arange(len(bottom_benchmarks))) + [gap_x]
xlabels = bottom_benchmarks + ["GeoMean"]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=12)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("figure4_raw_runtime_with_interval.pdf", dpi=300, bbox_inches="tight")
plt.show()