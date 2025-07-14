import os
import csv
import ssgetpy

# Define the paths
matrix_dir = '../../../matrices'
output_file = '../../../logs/matrix_application.csv'

# Function to get matrix "Kind" (Application) using ssgetpy
def get_matrix_kind(matrix_name):
    try:
        # Search for the matrix by name in the SuiteSparse collection
        matrices = ssgetpy.search(name=matrix_name)
        if len(matrices) == 0:
            print(f"Matrix '{matrix_name}' not found in SuiteSparse collection.")
            return "Unknown"
        # Retrieve the "Kind" of the matrix
        matrix = matrices[0]
        return matrix.kind
    except Exception as e:
        print(f"Error retrieving 'Kind' for {matrix_name}: {e}")
        return "Error"

# Function to process matrix directories and ignore banded versions
def process_matrices(matrix_dir):
    results = []
    
    for subdir_name in os.listdir(matrix_dir):
        subdir_path = os.path.join(matrix_dir, subdir_name)
        if os.path.isdir(subdir_path):
            original_matrix_name = f"{subdir_name}.mtx"
            original_matrix_path = os.path.join(subdir_path, original_matrix_name)
            
            # Check if the original matrix exists
            if os.path.exists(original_matrix_path):
                print(f"Processing {original_matrix_name}")
                
                # Get the "Kind" (Application) of the matrix
                kind = get_matrix_kind(subdir_name)
                
                # Store the result
                results.append({'Matrix Name': subdir_name, 'Application': kind})
            else:
                print(f"Original matrix file {original_matrix_name} not found in {subdir_path}")
    
    return results

# Write results to CSV
def write_csv(results):
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Matrix Name', 'Application'])
        writer.writeheader()
        writer.writerows(results)

# Main function
def main():
    # Process matrices and retrieve their kinds
    results = process_matrices(matrix_dir)
    
    # Write the results to CSV
    write_csv(results)
    
    print("All matrices processed and exported to CSV.")

if __name__ == '__main__':
    main()
