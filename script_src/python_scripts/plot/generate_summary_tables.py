import pandas as pd
import numpy as np
from scipy.stats import gmean
import os

def calculate_metrics(speedup_data, ratio_column, speedup_column):
    """Calculate geometric mean and percentage accelerated for given data"""
    if len(speedup_data) == 0:
        return 0.0, 0.0
    
    # Calculate geometric mean
    geo_mean = gmean(speedup_data[speedup_column])
    
    # Calculate percentage accelerated (speedup > 1)
    accelerated_count = (speedup_data[speedup_column] > 1).sum()
    percentage_accelerated = (accelerated_count / len(speedup_data)) * 100
    
    return geo_mean, percentage_accelerated

def get_oracle_selection(speedup_data):
    """For each matrix, select the ratio that gives the best speedup (oracle)"""
    oracle_results = []
    
    for matrix_name in speedup_data["Matrix Name"].unique():
        matrix_data = speedup_data[speedup_data["Matrix Name"] == matrix_name]
        # Find the row with maximum per-iteration speedup for this matrix
        best_row = matrix_data.loc[matrix_data["Per-iteration Speedup"].idxmax()]
        oracle_results.append(best_row)
    
    return pd.DataFrame(oracle_results)

def generate_ilu0_table():
    """Generate ILU0 summary table"""
    print("Generating ILU0 summary table...")
    
    # Load data
    speedup_data = pd.read_csv("../../../logs/ilu0_speedups.csv")
    algorithm_results = pd.read_csv("../../../script_src/python_scripts/algorithm/best_oracle_selection_gpu_pi_from_prediction.csv")
    
    results = []
    
    # Individual sparsification ratios
    for ratio in [0.01, 0.05, 0.10]:
        ratio_data = speedup_data[speedup_data["Sparsification Ratio"] == ratio]
        if len(ratio_data) > 0:
            geo_mean, pct_acc = calculate_metrics(ratio_data, "Sparsification Ratio", "Per-iteration Speedup")
            results.append({
                "Selection Strategy": f"Fixed {ratio}",
                "Geometric Mean": f"{geo_mean:.6f}",
                "% Accelerated": f"{pct_acc:.1f}%"
            })
    
    # Algorithm selected ratios
    if len(algorithm_results) > 0:
        # Merge algorithm results with speedup data
        algo_merged = pd.merge(
            algorithm_results, 
            speedup_data, 
            left_on=["Matrix Name", "Selected Sparsification Ratio"],
            right_on=["Matrix Name", "Sparsification Ratio"],
            how="inner"
        )
        if len(algo_merged) > 0:
            geo_mean, pct_acc = calculate_metrics(algo_merged, "Selected Sparsification Ratio", "Per-iteration Speedup")
            results.append({
                "Selection Strategy": "Algorithm Selected",
                "Geometric Mean": f"{geo_mean:.6f}",
                "% Accelerated": f"{pct_acc:.1f}%"
            })
    
    # Oracle selected ratios
    oracle_data = get_oracle_selection(speedup_data)
    if len(oracle_data) > 0:
        geo_mean, pct_acc = calculate_metrics(oracle_data, "Sparsification Ratio", "Per-iteration Speedup")
        results.append({
            "Selection Strategy": "Oracle Selected",
            "Geometric Mean": f"{geo_mean:.6f}",
            "% Accelerated": f"{pct_acc:.1f}%"
        })
    
    return pd.DataFrame(results)

def generate_iluk_table():
    """Generate ILUK summary table"""
    print("Generating ILUK summary table...")
    
    # Load data
    speedup_data = pd.read_csv("../../../logs/iluk_speedups_best_fill_factor.csv")
    algorithm_results = pd.read_csv("../../../script_src/python_scripts/algorithm/best_oracle_selection_gpu_pi_from_prediction.csv")
    
    results = []
    
    # Individual sparsification ratios
    for ratio in [0.01, 0.05, 0.10]:
        ratio_data = speedup_data[speedup_data["Sparsification Ratio"] == ratio]
        if len(ratio_data) > 0:
            geo_mean, pct_acc = calculate_metrics(ratio_data, "Sparsification Ratio", "Per-iteration Speedup")
            results.append({
                "Selection Strategy": f"Fixed {ratio}",
                "Geometric Mean": f"{geo_mean:.6f}",
                "% Accelerated": f"{pct_acc:.1f}%"
            })
    
    # Algorithm selected ratios (note: ILUK uses same algorithm file as ILU0)
    if len(algorithm_results) > 0:
        # Merge algorithm results with speedup data
        algo_merged = pd.merge(
            algorithm_results, 
            speedup_data, 
            left_on=["Matrix Name", "Selected Sparsification Ratio"],
            right_on=["Matrix Name", "Sparsification Ratio"],
            how="inner"
        )
        if len(algo_merged) > 0:
            geo_mean, pct_acc = calculate_metrics(algo_merged, "Selected Sparsification Ratio", "Per-iteration Speedup")
            results.append({
                "Selection Strategy": "Algorithm Selected",
                "Geometric Mean": f"{geo_mean:.6f}",
                "% Accelerated": f"{pct_acc:.1f}%"
            })
    
    # Oracle selected ratios
    oracle_data = get_oracle_selection(speedup_data)
    if len(oracle_data) > 0:
        geo_mean, pct_acc = calculate_metrics(oracle_data, "Sparsification Ratio", "Per-iteration Speedup")
        results.append({
            "Selection Strategy": "Oracle Selected",
            "Geometric Mean": f"{geo_mean:.6f}",
            "% Accelerated": f"{pct_acc:.1f}%"
        })
    
    return pd.DataFrame(results)

def save_tables_to_file(ilu0_table, iluk_table, output_file):
    """Save both tables to a single text file"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPARSE PRECONDITIONING PERFORMANCE SUMMARY TABLES\n")
        f.write("="*80 + "\n\n")
        
        f.write("ILU0 PERFORMANCE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(ilu0_table.to_string(index=False))
        f.write("\n\n")
        
        f.write("ILUK PERFORMANCE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(iluk_table.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("NOTES:\n")
        f.write("- Geometric Mean: Geometric mean of per-iteration speedups\n")
        f.write("- % Accelerated: Percentage of matrices with speedup > 1.0\n")
        f.write("- Fixed X.XX: Using fixed sparsification ratio for all matrices\n")
        f.write("- Algorithm Selected: Ratios chosen by prediction algorithm\n")
        f.write("- Oracle Selected: Best ratio per matrix (theoretical upper bound)\n")
        f.write("="*80 + "\n")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("../../../results", exist_ok=True)
    
    # Generate tables
    ilu0_table = generate_ilu0_table()
    iluk_table = generate_iluk_table()
    
    # Save to file
    output_file = "../../../results/performance_summary_tables.txt"
    save_tables_to_file(ilu0_table, iluk_table, output_file)
    
    print(f"\nSummary tables saved to: {output_file}")
    print("\nILU0 Table:")
    print(ilu0_table.to_string(index=False))
    print("\nILUK Table:")
    print(iluk_table.to_string(index=False)) 