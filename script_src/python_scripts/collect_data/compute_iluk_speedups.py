import pandas as pd
import os

# Input and output paths
input_csv = "../../../logs/iluk_raw.csv"
output_csv = "../../../logs/iluk_speedups_best_fill_factor.csv"

def compute_iluk_speedups():
    """
    Compute ILUK speedups from raw results, selecting best fill factor for each matrix/ratio combination
    """
    # Read the raw ILUK results
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        print("Please ensure T3 has been completed and iluk_raw.csv exists.")
        return
    
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} raw ILUK entries")
    
    # Initialize results list
    results = []
    
    # Group by matrix name and sparsification ratio to find best fill factor
    grouped = df.groupby(['Matrix Name', 'Sparsification Ratio'])
    
    for (matrix_name, sp_ratio), group in grouped:
        # Find the best fill factor (minimum per-iteration time for this matrix/ratio)
        if len(group) > 1:
            # Calculate per-iteration time for each fill factor
            group = group.copy()
            group['Per-iteration Time'] = (group['Overall Time (ms)'] - group['Preconditioning Time (ms)']) / group['Iterations Spent']
            
            # Select the fill factor with minimum per-iteration time
            best_row = group.loc[group['Per-iteration Time'].idxmin()]
        else:
            best_row = group.iloc[0]
        
        # Store the best result for this matrix/ratio combination
        result_entry = {
            'Matrix Name': matrix_name,
            'Sparsification Ratio': sp_ratio,
            'Fill Factor': best_row['Fill Factor'],
            'Overall Time (ms)': best_row['Overall Time (ms)'],
            'Preconditioning Time (ms)': best_row['Preconditioning Time (ms)'],
            'Iterations Spent': best_row['Iterations Spent']
        }
        results.append(result_entry)
    
    # Convert to DataFrame
    best_df = pd.DataFrame(results)
    
    # Now compute speedups by grouping by matrix name
    speedup_results = []
    matrix_groups = best_df.groupby('Matrix Name')
    
    for matrix_name, group in matrix_groups:
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
        
        # Compute non-sparsified per-iteration time
        nonsp_per_iter_time = (nonsp_overall_time - nonsp_precond_time) / nonsp_iterations
        
        # Process sparsified rows (sparsification ratio > 0)
        sp_rows = group[group['Sparsification Ratio'] > 0]
        
        for _, sp_row in sp_rows.iterrows():
            sp_ratio = sp_row['Sparsification Ratio']
            sp_overall_time = sp_row['Overall Time (ms)']
            sp_precond_time = sp_row['Preconditioning Time (ms)']
            sp_iterations = sp_row['Iterations Spent']
            
            # Compute sparsified per-iteration time
            sp_per_iter_time = (sp_overall_time - sp_precond_time) / sp_iterations
            
            # Compute speedups
            per_iter_speedup = nonsp_per_iter_time / sp_per_iter_time
            end_to_end_speedup = nonsp_overall_time / sp_overall_time
            
            # Add to results
            speedup_results.append({
                'Matrix Name': matrix_name,
                'Sparsification Ratio': sp_ratio,
                'Per-iteration Speedup': per_iter_speedup,
                'End-to-end Speedup': end_to_end_speedup
            })
            
            print(f"  Ratio {sp_ratio}: per-iter={per_iter_speedup:.4f}, end-to-end={end_to_end_speedup:.4f}")
    
    # Create DataFrame and save
    speedup_df = pd.DataFrame(speedup_results)
    
    # Sort by matrix name and sparsification ratio
    speedup_df = speedup_df.sort_values(['Matrix Name', 'Sparsification Ratio'])
    
    # Save to CSV
    speedup_df.to_csv(output_csv, index=False)
    
    print(f"\nILUK speedups saved to: {output_csv}")
    print(f"Total speedup entries: {len(speedup_df)}")
    
    # Also save the best fill factor raw data
    best_raw_csv = "../../../logs/iluk_raw_best_fill_factor.csv"
    best_df.to_csv(best_raw_csv, index=False)
    print(f"Best fill factor raw data saved to: {best_raw_csv}")

if __name__ == '__main__':
    compute_iluk_speedups() 