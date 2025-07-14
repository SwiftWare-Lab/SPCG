import pandas as pd
import os

# Input and output paths
input_csv = "../../../logs/cpu_raw.csv"
output_csv = "../../../logs/cpu_speedups.csv"

def compute_cpu_speedups():
    """
    Compute CPU speedups from raw results
    """
    # Read the raw CPU results
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        print("Please ensure T3 has been completed and cpu_raw.csv exists.")
        return
    
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} raw CPU entries")
    
    # Initialize results list
    results = []
    
    # Group by matrix name
    grouped = df.groupby('Matrix Name')
    
    for matrix_name, group in grouped:
        print(f"Processing matrix: {matrix_name}")
        
        # Find the non-sparsified row (sparsification ratio = 0)
        nonsp_rows = group[group['Sparsification Ratio'] == 0]
        
        if len(nonsp_rows) == 0:
            print(f"  Warning: No non-sparsified row found for {matrix_name}")
            continue
        
        # Use the first non-sparsified row if multiple exist
        nonsp_row = nonsp_rows.iloc[0]
        
        # Extract non-sparsified metrics
        nonsp_overall_time = nonsp_row['Overall Time (ms)']
        nonsp_precond_time = nonsp_row['Preconditioning Time (ms)']
        nonsp_iterations = nonsp_row['Iterations Spent']
        nonsp_converged = nonsp_row.get('Converged', True)  # Assume converged if not specified
        
        # Compute non-sparsified per-iteration time
        nonsp_per_iter_time = (nonsp_overall_time - nonsp_precond_time) / nonsp_iterations
        
        # Process sparsified rows (sparsification ratio > 0)
        sp_rows = group[group['Sparsification Ratio'] > 0]
        
        for _, sp_row in sp_rows.iterrows():
            sp_ratio = sp_row['Sparsification Ratio']
            sp_overall_time = sp_row['Overall Time (ms)']
            sp_precond_time = sp_row['Preconditioning Time (ms)']
            sp_iterations = sp_row['Iterations Spent']
            sp_converged = sp_row.get('Converged', True)  # Assume converged if not specified
            
            # Compute sparsified per-iteration time
            sp_per_iter_time = (sp_overall_time - sp_precond_time) / sp_iterations
            
            # Compute speedups
            per_iter_speedup = nonsp_per_iter_time / sp_per_iter_time
            end_to_end_speedup = nonsp_overall_time / sp_overall_time
            
            # Convergence analysis
            originally_converging = nonsp_converged and nonsp_iterations < 1000
            unaffectedly_converging = sp_converged and sp_iterations < 1000
            
            # Add to results
            results.append({
                'Matrix Name': matrix_name,
                'Sparsification Ratio': sp_ratio,
                'Per-iteration Speedup': per_iter_speedup,
                'End-to-end Speedup': end_to_end_speedup,
                'Originally Converging': originally_converging,
                'Unaffectedly Converging': unaffectedly_converging
            })
            
            print(f"  Ratio {sp_ratio}: per-iter={per_iter_speedup:.4f}, end-to-end={end_to_end_speedup:.4f}")
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Sort by matrix name and sparsification ratio
    results_df = results_df.sort_values(['Matrix Name', 'Sparsification Ratio'])
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    
    print(f"\nCPU speedups saved to: {output_csv}")
    print(f"Total speedup entries: {len(results_df)}")

if __name__ == '__main__':
    compute_cpu_speedups() 