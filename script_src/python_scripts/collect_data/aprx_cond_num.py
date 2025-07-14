import pandas as pd

# Input paths
diag_csv = "../../../logs/diag_min.csv"
norm_csv = "../../../logs/inf_norm_os.csv"
output_csv = "../../../logs/approximated_condition_number_inf.csv"

# Read CSVs
diag_df = pd.read_csv(diag_csv)
norm_df = pd.read_csv(norm_csv)

# Ensure matching column types for merging
diag_df["Sparsification Ratio"] = diag_df["Sparsification Ratio"].astype(float)
norm_df["Sparsification Ratio"] = norm_df["Sparsification Ratio"].astype(float)

# Merge on Matrix Name and Sparsification Ratio
merged = pd.merge(diag_df, norm_df, on=["Matrix Name", "Sparsification Ratio"], how="inner")

# Compute Approximated Condition Number
merged["Approximated Condition Number"] = (
        merged["Infinity Norm"] / merged["Smallest Diagonal Entry (Abs)"]
)

# Select and export result
result = merged[["Matrix Name", "Sparsification Ratio", "Approximated Condition Number"]]
result.to_csv(output_csv, index=False)

print(f"Saved result to: {output_csv}")
