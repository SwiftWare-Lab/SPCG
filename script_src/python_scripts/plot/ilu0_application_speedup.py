import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
import os

def main():
    # Load data
    speedup_file = "../../../logs/ilu0_speedups.csv"
    application_file = "../../../logs/matrix_application.csv"
    algorithm_file = "../../../script_src/python_scripts/algorithm/best_oracle_selection_gpu_pi_from_prediction.csv"
    output_dir = "../../../results/plots"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    speedup_data = pd.read_csv(speedup_file)
    application_data = pd.read_csv(application_file)
    algorithm_results = pd.read_csv(algorithm_file)
    
    print(f"Loaded {len(speedup_data)} speedup records")
    print(f"Loaded {len(application_data)} application records")
    print(f"Loaded {len(algorithm_results)} algorithm results")
    
    # Merge algorithm results with speedup data to get algorithm-selected speedups
    algo_merged = pd.merge(
        algorithm_results, 
        speedup_data, 
        left_on=["Matrix Name", "Selected Sparsification Ratio"],
        right_on=["Matrix Name", "Sparsification Ratio"],
        how="inner"
    )
    
    # Merge with application data
    final_data = pd.merge(
        algo_merged,
        application_data,
        on="Matrix Name",
        how="inner"
    )
    
    print(f"Final merged data: {len(final_data)} records")
    
    if len(final_data) == 0:
        print("No data available for plotting")
        return
    
    # Group by application and compute geometric mean
    app_speedups = final_data.groupby("Application")["Per-iteration Speedup"].agg(gmean).reset_index()
    app_speedups.columns = ["Application", "Gmean Per Iteration Speedup"]
    
    # Sort by speedup for better visualization
    app_speedups = app_speedups.sort_values("Gmean Per Iteration Speedup", ascending=True)
    
    print("\nApplication Speedups:")
    print(app_speedups.to_string(index=False))
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(app_speedups["Application"], app_speedups["Gmean Per Iteration Speedup"], 
                    color='#1f77b4', edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    # Add reference line at x=1
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.xlabel('Gmean Per Iteration Speedup', fontsize=14, fontweight='bold')
    plt.ylabel('Application', fontsize=14, fontweight='bold')
    plt.title('ILU0 Performance by Matrix Application Type', fontsize=16, fontweight='bold')
    
    # Style
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "ilu0_application_speedup.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nILU0 application speedup plot saved to: {output_path}")

if __name__ == "__main__":
    main() 