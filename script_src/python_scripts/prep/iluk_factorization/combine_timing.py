import os
import pandas as pd

directory = os.getcwd()

combined_df = pd.DataFrame()

for filename in os.listdir(directory):
    if filename.endswith("_timing.csv"):
        file_path = os.path.join(directory, filename)
        
        try:
            if os.path.getsize(file_path) > 0:  # Check if the file is not empty
                df = pd.read_csv(file_path)
                
                df['Factorization Time (ms)'] = df['Time (s)'] * 1000
                
                df = df.drop(columns=['Time (s)'])
                
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                print(f"Skipping empty file: {filename}")
        
        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: {filename} is empty or incorrectly formatted, skipping.")

combined_df.to_csv('combined_timing_results.csv', index=False)

print("Combined CSV file 'combined_timing_results.csv' has been created with time converted to milliseconds.")
