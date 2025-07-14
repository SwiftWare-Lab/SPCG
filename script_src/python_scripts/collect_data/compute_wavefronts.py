import os
import pandas as pd
import glob

# Input and output paths
factors_timing_dir = "../../../factors/timing"
output_csv = "../../../logs/wavefronts.csv"

def combine_wavefront_data():
    """
    Combine all timing CSV files from iluk_factorize.py to create wavefronts.csv
    """
    combined_df = pd.DataFrame()
    
    # Check if timing directory exists
    if not os.path.exists(factors_timing_dir):
        print(f"Error: Timing directory {factors_timing_dir} not found.")
        print("Please ensure iluk_factorize.py has been run to generate timing data.")
        return
    
    # Find all timing CSV files
    timing_files = glob.glob(os.path.join(factors_timing_dir, "*_timing.csv"))
    
    if not timing_files:
        print(f"No timing files found in {factors_timing_dir}")
        return
    
    print(f"Found {len(timing_files)} timing files to process...")
    
    for file_path in timing_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        try:
            if os.path.getsize(file_path) > 0:  # Check if the file is not empty
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                required_cols = ['Matrix Name', 'Fill Factor', 'Sparsified', 'Removal Percentage', 
                               'Threshold', '#Wavefront Set', '#Wavefront Set Sp', 'Time (s)']
                
                if not all(col in df.columns for col in required_cols):
                    print(f"  Warning: Missing required columns in {filename}, skipping...")
                    continue
                
                # Convert time to milliseconds
                df['Factorization Time (ms)'] = df['Time (s)'] * 1000
                
                # Select relevant columns for wavefront analysis
                wavefront_df = df[['Matrix Name', 'Fill Factor', 'Sparsified', 'Removal Percentage',
                                 'Threshold', '#Wavefront Set', '#Wavefront Set Sp', 
                                 'Factorization Time (ms)']].copy()
                
                # Rename columns to match expected format
                wavefront_df.rename(columns={
                    '#Wavefront Set': 'Wavefront Count',
                    '#Wavefront Set Sp': 'Wavefront Count After Sparsification'
                }, inplace=True)
                
                combined_df = pd.concat([combined_df, wavefront_df], ignore_index=True)
            else:
                print(f"  Skipping empty file: {filename}")
        
        except pd.errors.EmptyDataError:
            print(f"  EmptyDataError: {filename} is empty or incorrectly formatted, skipping.")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    if len(combined_df) == 0:
        print("No valid data found to combine.")
        return
    
    # Sort by matrix name, fill factor, and sparsified status
    combined_df = combined_df.sort_values(['Matrix Name', 'Fill Factor', 'Sparsified', 'Removal Percentage'])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save to CSV
    combined_df.to_csv(output_csv, index=False)
    
    print(f"\nWavefront data saved to: {output_csv}")
    print(f"Total entries: {len(combined_df)}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Matrices processed: {combined_df['Matrix Name'].nunique()}")
    print(f"  Fill factors: {sorted(combined_df['Fill Factor'].unique())}")
    print(f"  Sparsification ratios: {sorted(combined_df['Removal Percentage'].unique())}")

if __name__ == '__main__':
    combine_wavefront_data() 