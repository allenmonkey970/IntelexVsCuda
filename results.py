import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'PerformanceSpeeds.csv'
df = pd.read_csv(file_path)
df = df[df["Speed(milliseconds)"].notna()]

datasets = df["Dataset"].unique()
algorithms = df["Algorithm"].unique()
implementations = df["Implementation"].unique()

def dynamic_y_axis_ticks(data):
    max_value = max(data)
    if max_value > 60000:
        step = 60000
        ticks = list(range(0, int(max_value) + step, step))
        labels = [f"{int(tick/60000)}min" if tick >= 60000 else f"{tick}ms" for tick in ticks]
    else:
        step = 10000
        ticks = list(range(0, int(max_value) + step, step))
        labels = [f"{tick}ms" for tick in ticks]
    return ticks, labels

for dataset in datasets:
    plt.figure(figsize=(10, 6))
    plt.title(f"Algorithm Speeds for {dataset}", fontsize=16)

    dataset_df = df[df["Dataset"] == dataset]
    speeds_all = []
    for i, impl in enumerate(implementations):
        impl_data = dataset_df[dataset_df["Implementation"] == impl]
        speeds = [impl_data[impl_data["Algorithm"] == algo]["Speed(milliseconds)"].values[0]
                  if algo in impl_data["Algorithm"].values else 0
                  for algo in algorithms]
        speeds_all.extend(speeds)

        plt.bar(np.arange(len(algorithms)) + i * 0.2, speeds, 0.2, label=impl)

    custom_ticks, custom_labels = dynamic_y_axis_ticks(speeds_all)

    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Speed", fontsize=12)
    plt.xticks(np.arange(len(algorithms)) + (0.2 * (len(implementations) - 1) / 2),
               algorithms, rotation=45, fontsize=10)
    plt.yticks(custom_ticks, custom_labels)
    plt.ylim(0, custom_ticks[-1])
    plt.legend(title="Implementation", fontsize=10)
    plt.tight_layout()
    plt.show()
