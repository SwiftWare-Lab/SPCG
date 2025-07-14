import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # Load data
    raw_file = "../../../logs/ilu0_raw.csv"
    output_dir = "../../../results/plots"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ILU0 raw data
    raw_data = pd.read_csv(raw_file)
    
    print(f"Loaded {len(raw_data)} raw ILU0 records")
    
    # Prepare data for plotting
    plot_data = []
    colors = {'0.01': '#ff7f0e', '0.05': '#2ca02c', '0.1': '#d62728'}  # Orange, Green, Red
    labels = {'0.01': 'Ratio 0.01', '0.05': 'Ratio 0.05', '0.1': 'Ratio 0.10'}
    
    # Process each matrix
    for matrix_name in raw_data["Matrix Name"].unique():
        matrix_data = raw_data[raw_data["Matrix Name"] == matrix_name]
        
        # Find baseline (sparsification ratio = 0)
        baseline = matrix_data[matrix_data["Sparsification Ratio"] == 0]
        if len(baseline) == 0:
            print(f"Warning: No baseline found for matrix {matrix_name}")
            continue
        
        baseline_row = baseline.iloc[0]
        baseline_factorization_time = baseline_row["Preconditioning Time (ms)"]
        nonzeros = baseline_row["Nonzeros"]
        
        # Process sparsified versions
        for ratio in [0.01, 0.05, 0.1]:
            sparsified = matrix_data[matrix_data["Sparsification Ratio"] == ratio]
            if len(sparsified) == 0:
                print(f"Warning: No data for ratio {ratio} in matrix {matrix_name}")
                continue
            
            sparsified_row = sparsified.iloc[0]
            sparsified_factorization_time = sparsified_row["Preconditioning Time (ms)"]
            
            # Calculate factorization speedup
            if sparsified_factorization_time > 0:
                factorization_speedup = baseline_factorization_time / sparsified_factorization_time
                
                plot_data.append({
                    "Matrix Name": matrix_name,
                    "Sparsification Ratio": str(ratio),
                    "Nonzeros": nonzeros,
                    "Factorization Speedup": factorization_speedup
                })
    
    if len(plot_data) == 0:
        print("No data available for plotting")
        return
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    print(f"Prepared {len(plot_df)} data points for plotting")
    print("\nData summary:")
    for ratio in ['0.01', '0.05', '0.1']:
        ratio_data = plot_df[plot_df["Sparsification Ratio"] == ratio]
        if len(ratio_data) > 0:
            mean_speedup = ratio_data["Factorization Speedup"].mean()
            print(f"  Ratio {ratio}: {len(ratio_data)} points, mean speedup = {mean_speedup:.3f}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    for ratio in ['0.01', '0.05', '0.1']:
        ratio_data = plot_df[plot_df["Sparsification Ratio"] == ratio]
        if len(ratio_data) > 0:
            plt.scatter(ratio_data["Nonzeros"], ratio_data["Factorization Speedup"], 
                       color=colors[ratio], label=labels[ratio], alpha=0.7, s=60)
    
    # Add reference line at y=1
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.xlabel('Number of Nonzeros', fontsize=14, fontweight='bold')
    plt.ylabel('Factorization Speedup', fontsize=14, fontweight='bold')
    plt.title('ILU0 Factorization Speedup vs Matrix Size', fontsize=16, fontweight='bold')
    
    # Use log scale for x-axis since nonzeros can vary widely
    plt.xscale('log')
    
    # Style
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "ilu0_factorization_speedup.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nILU0 factorization speedup plot saved to: {output_path}")

if __name__ == "__main__":
    main() 