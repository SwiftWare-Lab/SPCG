import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def prepare_algorithm_selected_data(speedup_file, algorithm_file, output_file, method_name):
    """Prepare data with algorithm selected speedups"""
    speedup_data = pd.read_csv(speedup_file)
    
    try:
        algorithm_results = pd.read_csv(algorithm_file)
        
        # Merge algorithm results with speedup data
        merged = pd.merge(
            algorithm_results, 
            speedup_data, 
            left_on=["Matrix Name", "Selected Sparsification Ratio"],
            right_on=["Matrix Name", "Sparsification Ratio"],
            how="inner"
        )
        
        if len(merged) > 0:
            # Create the required columns for histogram plotting
            result_data = merged[["Matrix Name", "Selected Sparsification Ratio"]].copy()
            result_data["Selected Per-iteration Speedup"] = merged["Per-iteration Speedup"]
            result_data["Selected End-to-end Speedup"] = merged["End-to-end Speedup"]
            
            result_data.to_csv(output_file, index=False)
            print(f"Prepared {len(result_data)} algorithm-selected records for {method_name}")
            return result_data
        else:
            print(f"Warning: No algorithm-selected data found for {method_name}")
            return None
    except FileNotFoundError:
        print(f"Warning: Algorithm results file not found for {method_name}")
        return None

def prepare_cpu_data(speedup_file, algorithm_file, output_file):
    """Prepare CPU data with algorithm selected speedups"""
    try:
        algorithm_results = pd.read_csv(algorithm_file)
        
        if len(algorithm_results) > 0:
            # CPU algorithm results already have the correct column names
            # Just rename them to match the expected format
            result_data = algorithm_results.copy()
            result_data["Selected Per-iteration Speedup"] = algorithm_results["Per-iteration Speedup"]
            result_data["Selected End-to-end Speedup"] = algorithm_results["End-to-end Speedup"]
            
            result_data.to_csv(output_file, index=False)
            print(f"Prepared {len(result_data)} algorithm-selected records for CPU")
            return result_data
        else:
            print(f"Warning: No algorithm-selected data found for CPU")
            return None
    except FileNotFoundError:
        print(f"Warning: Algorithm results file not found for CPU")
        return None

def plot_histogram(data, method_name, output_dir):
    """Plot histogram for speedup distribution"""
    speedup_columns = [
        'Selected Per-iteration Speedup',
        'Selected End-to-end Speedup'
    ]

    for speedup_type in speedup_columns:
        if speedup_type not in data.columns:
            print(f"Column '{speedup_type}' not found in {method_name} data. Skipping plot.")
            continue

        speedup_data = data[speedup_type]

        bins = np.arange(0, 5.25, 0.25)

        filtered_speedup_data = speedup_data[(speedup_data >= 0) & (speedup_data <= 5)]

        hist, bin_edges = np.histogram(filtered_speedup_data, bins=bins)
        hist_percentage = hist / hist.sum() * 100  # Normalize to percentages

        plt.figure(figsize=(10, 6))
        plt.bar(
            bin_edges[:-1], hist_percentage, width=0.25, color='#1f77b4', edgecolor='black', align='edge'
        )
        plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label="x=1")

        if method_name.upper() == "CPU":
            axis_label = speedup_type.replace("Selected ", "SPCG-ILU(0) ")
        else:
            axis_label = speedup_type.replace("Selected ", f"SPCG-{method_name.upper()} ")
        
        plt.xlabel(axis_label, fontsize=18, fontweight="bold")
        plt.ylabel('Distribution (%)', fontsize=18, fontweight="bold")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        
        # Save plot
        speedup_type_clean = speedup_type.lower().replace(' ', '_').replace('-', '_')
        output_path = os.path.join(output_dir, f"{method_name.lower()}_{speedup_type_clean}_histogram.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{method_name} {speedup_type} histogram saved to: {output_path}")

def main():
    # Setup paths
    output_dir = "../../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare and plot ILU0 histograms
    print("Processing ILU0 histograms...")
    ilu0_data = prepare_algorithm_selected_data(
        "../../../logs/ilu0_speedups.csv",
        "../../../script_src/python_scripts/algorithm/best_oracle_selection_gpu_pi_from_prediction.csv",
        "../../../results/ilu0_histogram_data.csv",
        "ILU0"
    )
    if ilu0_data is not None:
        plot_histogram(ilu0_data, "ILU0", output_dir)
    
    # Prepare and plot ILUK histograms
    print("\nProcessing ILUK histograms...")
    iluk_data = prepare_algorithm_selected_data(
        "../../../logs/iluk_speedups_best_fill_factor.csv",
        "../../../script_src/python_scripts/algorithm/best_oracle_selection_gpu_pi_from_prediction.csv",
        "../../../results/iluk_histogram_data.csv",
        "ILUK"
    )
    if iluk_data is not None:
        plot_histogram(iluk_data, "ILUK", output_dir)
    
    # Prepare and plot CPU histograms
    print("\nProcessing CPU histograms...")
    cpu_data = prepare_cpu_data(
        "../../../logs/cpu_speedups.csv",
        "../../../script_src/python_scripts/algorithm/best_oracle_selection_gpu_pi_from_prediction_cpu.csv",
        "../../../results/cpu_histogram_data.csv"
    )
    if cpu_data is not None:
        plot_histogram(cpu_data, "CPU", output_dir)

if __name__ == "__main__":
    main() 