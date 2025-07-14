import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

def calculate_wavefront_reduction(non_sparsified_wavefront, sparsified_wavefront):
    if non_sparsified_wavefront > 0:
        return (non_sparsified_wavefront - sparsified_wavefront) / non_sparsified_wavefront
    return 0

def extract_data_for_correlation(speedup_data, wavefront_data):
    x_values = []
    y_values = []

    for matrix_name, group in speedup_data.groupby("Matrix Name"):
        max_speedup_row = group.loc[group["Per-iteration Speedup"].idxmax()]
        max_speedup = max_speedup_row["Per-iteration Speedup"]
        best_ratio = max_speedup_row["Sparsification Ratio"]
        
        wavefront_group = wavefront_data[wavefront_data["Matrix Name"] == matrix_name]
        if wavefront_group.empty:
            print(f"Matrix {matrix_name} is missing wavefront data.")
            continue

        non_sparsified_wavefront = wavefront_group[wavefront_group["Sparsification Ratio"] == 0]["Wavefront Count"].values
        if len(non_sparsified_wavefront) == 0:
            print(f"Matrix {matrix_name} is missing non-sparsified wavefront data.")
            continue

        # Get corresponding wavefront count for this sparsification ratio
        sparsified_wavefront_data = wavefront_group[wavefront_group["Sparsification Ratio"] == best_ratio]["Wavefront Count"].values
        if len(sparsified_wavefront_data) == 0:
            print(f"Matrix {matrix_name} is missing wavefront data for ratio {best_ratio}")
            continue
        sparsified_wavefront = sparsified_wavefront_data[0]
        
        reduction_ratio = calculate_wavefront_reduction(non_sparsified_wavefront[0], sparsified_wavefront)

        if 0 <= max_speedup <= 6:
            x_values.append(max_speedup)
            y_values.append(reduction_ratio)

    return np.array(x_values), np.array(y_values)

def plot_correlation(x_values, y_values, output_path):
    plt.figure(figsize=(12, 9))
    plt.scatter(x_values, y_values, color="blue", alpha=0.6)

    # Trend line
    slope, intercept, r_value, _, _ = linregress(x_values, y_values)
    print(f"ILUK Coefficient of Determination (R²): {r_value ** 2:.4f}")
    trend_x = np.linspace(min(x_values), max(x_values), 500)
    trend_y = slope * trend_x + intercept
    plt.plot(trend_x, trend_y, color="red", label=f"Trend Line ($R^2$ = {r_value**2:.2f})")

    # Apply labels
    plt.xlabel("Per-iteration Speedup", fontsize=18, fontweight="bold")
    plt.ylabel("Wavefront Reduction Ratio", fontsize=18, fontweight="bold")
    plt.xlim(0, 5)

    # Remove x=0 and x=6 from ticks
    ax = plt.gca()
    current_ticks = ax.get_xticks()
    new_ticks = [tick for tick in current_ticks if tick not in [0, 6]]
    ax.set_xticks(new_ticks)

    # Tick formatting
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ILUK correlation plot saved to: {output_path}")

def main():
    # Load data files
    speedup_file = "../../../logs/iluk_speedups_best_fill_factor.csv"
    wavefront_file = "../../../logs/wavefronts.csv"
    output_dir = "../../../results/plots"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    speedup_data = pd.read_csv(speedup_file)
    wavefront_data = pd.read_csv(wavefront_file)
    
    print(f"Loaded {len(speedup_data)} ILUK speedup records")
    print(f"Loaded {len(wavefront_data)} wavefront records")
    
    # Extract correlation data
    x_values, y_values = extract_data_for_correlation(speedup_data, wavefront_data)
    
    if len(x_values) > 0:
        output_path = os.path.join(output_dir, "iluk_correlation.png")
        plot_correlation(x_values, y_values, output_path)
    else:
        print("No valid data to plot for ILUK.")

if __name__ == "__main__":
    main() 