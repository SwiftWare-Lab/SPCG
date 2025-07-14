import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

def calculate_wavefront_reduction(non_sparsified_wavefront, sparsified_wavefront):
    """Calculate wavefront reduction."""
    if non_sparsified_wavefront > 0:
        return (non_sparsified_wavefront - sparsified_wavefront) / non_sparsified_wavefront
    return 0

def process_speedup(speedup_data, wavefront_data):
    """Process speedup and calculate wavefront reductions."""
    best_speedups = []

    for matrix_name, group in speedup_data.groupby("Matrix Name"):
        wavefront_group = wavefront_data[wavefront_data["Matrix Name"] == matrix_name]

        if wavefront_group.empty:
            print(f"Matrix {matrix_name} is missing wavefront data.")
            continue

        non_sparsified_wavefront = wavefront_group[wavefront_group["Sparsification Ratio"] == 0]["Wavefront Count"].values
        if len(non_sparsified_wavefront) == 0:
            print(f"Matrix {matrix_name} is missing non-sparsified wavefront data.")
            continue
        non_sparsified_wavefront = non_sparsified_wavefront[0]

        # Find the best speedup for this matrix
        max_speedup_row = group.loc[group["Per-iteration Speedup"].idxmax()]
        max_speedup = max_speedup_row["Per-iteration Speedup"]
        best_ratio = max_speedup_row["Sparsification Ratio"]
        
        # Get corresponding wavefront count for this sparsification ratio
        sparsified_wavefront_data = wavefront_group[wavefront_group["Sparsification Ratio"] == best_ratio]["Wavefront Count"].values
        if len(sparsified_wavefront_data) == 0:
            print(f"Matrix {matrix_name} is missing wavefront data for ratio {best_ratio}")
            continue
        sparsified_wavefront = sparsified_wavefront_data[0]
        
        wavefront_reduction = calculate_wavefront_reduction(non_sparsified_wavefront, sparsified_wavefront)

        # Only include data points within the valid speedup range
        if 0 <= max_speedup <= 6:
            best_speedups.append({
                "Matrix Name": matrix_name,
                "Per-iteration Speedup": max_speedup,
                "Wavefront Reduction": wavefront_reduction
            })

    return pd.DataFrame(best_speedups)

def plot_scatter_with_trendline(data, output_path):
    """Plot scatter plot with trend line."""
    x_values = data["Per-iteration Speedup"]
    y_values = data["Wavefront Reduction"]

    # Scatter plot
    plt.figure(figsize=(12, 9))
    plt.scatter(x_values, y_values, color="blue", alpha=0.6, label="Matrices")

    # Trendline
    slope, intercept, r_value, _, _ = linregress(x_values, y_values)
    print(f"ILU0 Coefficient of Determination (R²): {r_value ** 2:.4f}")
    trendline = slope * x_values + intercept
    plt.plot(x_values, trendline, color="red", label=f"Trendline (R²={r_value**2:.2f})")

    # Labels and legend
    plt.xlabel("Per-iteration Speedup", fontsize=18, fontweight="bold")
    plt.ylabel("Wavefront Reduction Ratio", fontsize=18, fontweight="bold")

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tick formatting
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ILU0 correlation plot saved to: {output_path}")

def main():
    # Load data files
    speedup_file = "../../../logs/ilu0_speedups.csv"
    wavefront_file = "../../../logs/wavefronts.csv"
    output_dir = "../../../results/plots"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    speedup_data = pd.read_csv(speedup_file)
    wavefront_data = pd.read_csv(wavefront_file)
    
    print(f"Loaded {len(speedup_data)} speedup records")
    print(f"Loaded {len(wavefront_data)} wavefront records")
    
    # Process data
    processed_data = process_speedup(speedup_data, wavefront_data)
    
    if not processed_data.empty:
        output_path = os.path.join(output_dir, "ilu0_correlation.png")
        plot_scatter_with_trendline(processed_data, output_path)
    else:
        print("No valid data to plot for ILU0.")

if __name__ == "__main__":
    main() 