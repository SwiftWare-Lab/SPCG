import os
import csv

def parse_dirname(dirname):
    if dirname.startswith("spilu_sp_"):
        prefix = "spilu_sp_"
    elif dirname.startswith("spilu_nonsp_"):
        prefix = "spilu_nonsp_"
    else:
        return None, None, None
    
    matrix_name = dirname.split(prefix)[1].split("__")[0]
    
    fill_factor = float(dirname.split("fill_factor_")[1].split("_")[0])
    
    removal_percentage = float(dirname.split("_")[-1])
    
    return matrix_name, fill_factor, removal_percentage

def get_nnz_from_mtx(mtx_file):
    with open(mtx_file, 'r') as file:
        for line in file:
            if not line.startswith("%"):
                parts = line.split()
                return int(parts[2])

def process_subdirectories(base_dir="."):
    output_file = "matrix_info.csv"
    with open(output_file, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Matrix Name", "Fill Factor", "Removal Percentage", "NNZ_L", "NNZ_U"])
        
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            
            if os.path.isdir(subdir_path) and (subdir.startswith("spilu_sp_") or subdir.startswith("spilu_nonsp_")):
                
                matrix_name, fill_factor, removal_percentage = parse_dirname(subdir)
                if matrix_name is None:
                    continue  # Skip if directory name format is incorrect
                
                nnz_L, nnz_U = None, None
                
                for file in os.listdir(subdir_path):
                    if "L_factor" in file and file.endswith(".mtx"):
                        nnz_L = get_nnz_from_mtx(os.path.join(subdir_path, file))
                    elif "U_factor" in file and file.endswith(".mtx"):
                        nnz_U = get_nnz_from_mtx(os.path.join(subdir_path, file))
                
                if nnz_L is not None and nnz_U is not None:
                    csv_writer.writerow([matrix_name, fill_factor, removal_percentage, nnz_L, nnz_U])

                    print(f"Logged: {matrix_name}, Fill Factor: {fill_factor}, Removal Percentage: {removal_percentage}, NNZ_L: {nnz_L}, NNZ_U: {nnz_U}")

    print(f"Data successfully logged to {output_file}")

if __name__ == "__main__":
    process_subdirectories()
