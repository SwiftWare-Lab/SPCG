import pandas as pd
from scipy.stats import gmean
import numpy as np
import itertools

def select_sparsification_ratio(matrix_name, threshold, threshold_wf, cond_df, norm2_A_df, norm2_S_df, wf_df):
    for ratio in [0.1, 0.05, 0.01]:
        try:
            k_A = cond_df.loc[
                (cond_df["Matrix Name"] == matrix_name) &
                (cond_df["Sparsification Ratio"] == ratio),
                "Approximated Condition Number"
            ].values[0]

            norm_A = norm2_A_df.loc[
                (norm2_A_df["Matrix Name"] == matrix_name) &
                (norm2_A_df["Sparsification Ratio"] == ratio),
                "2-norm of Aos"
            ].values[0]

            inverse_norm = k_A / norm_A

            num_wf_o = wf_df.loc[
                (wf_df["Matrix Name"] == matrix_name) &
                (wf_df["Sparsification Ratio"] == 0),
                "Wavefront Count"
            ].values[0]
        except Exception as e:
            print(f"[WARN] Skipping {matrix_name} @ ratio={ratio} due to missing base data: {e}")
            continue

        try:
            norm_s = norm2_S_df.loc[
                (norm2_S_df["Matrix Name"] == matrix_name) &
                (norm2_S_df["Sparsification Ratio"] == ratio),
                "2-norm of E"
            ].values[0]

            num_wf_os = wf_df.loc[
                (wf_df["Matrix Name"] == matrix_name) &
                (wf_df["Sparsification Ratio"] == ratio),
                "Wavefront Count"
            ].values[0]
        except Exception as e:
            print(f"[WARN] Skipping {matrix_name} @ ratio={ratio} due to missing sparsified data: {e}")
            continue

        product = inverse_norm * norm_s
        if product >= threshold:
            if ratio == 0.01:
                return 0.1
            continue

        wf_reduction = ((num_wf_o - num_wf_os) / num_wf_o) * 100
        if wf_reduction >= threshold_wf:
            return ratio

        if ratio == 0.01:
            return 0.1

    return 0.01

if __name__ == "__main__":
    print(" Loading input data...")

    full_df = pd.read_csv("../../../logs/cpu_speedups.csv")
    cond_df = pd.read_csv("../../../logs/approximated_condition_number_inf.csv")
    norm2_A_df = pd.read_csv("../../../logs/norm2_os.csv")
    norm2_S_df = pd.read_csv("../../../logs/norm2_s.csv")
    wf_df = pd.read_csv("../../../logs/wavefronts.csv")
    app_df = pd.read_csv("../../../logs/matrix_application.csv")

    print(f"Loaded {len(full_df)} rows from CPU speedups.")

    # Ensure required columns exist
    required_cols = [
        "Matrix Name", "Sparsification Ratio", "Per-iteration Speedup",
        "End-to-end Speedup", "Originally Converging", "Unaffectedly Converging"
    ]
    for col in required_cols:
        if col not in full_df.columns:
            raise ValueError(f"Missing expected column in CPU speedups CSV: {col}")

    valid_grouped = full_df.groupby("Matrix Name").filter(lambda x: x["Sparsification Ratio"].isin([0.01, 0.05, 0.1]).sum() == 3)
    all_matrices = sorted(valid_grouped["Matrix Name"].unique())
    print(f"Found {len(all_matrices)} matrices with 3 sparsification levels (0.01, 0.05, 0.1)")

    threshold_list = [1]
    threshold_wf_list = [10]

    results = []
    best_gmean = -np.inf
    best_combo = None
    best_df = None

    for threshold, threshold_wf in itertools.product(threshold_list, threshold_wf_list):
        print(f"\n Testing threshold={threshold}, wf_thresh={threshold_wf}")
        selected = []

        for matrix in all_matrices:
            ratio = select_sparsification_ratio(matrix, threshold, threshold_wf, cond_df, norm2_A_df, norm2_S_df, wf_df)
            if ratio is not None:
                selected.append((matrix, ratio))
            else:
                print(f"[INFO] Matrix '{matrix}' skipped during selection (no valid ratio found)")

        result_df = pd.DataFrame(selected, columns=["Matrix Name", "Selected Sparsification Ratio"])
        merged = pd.merge(result_df, full_df, left_on=["Matrix Name", "Selected Sparsification Ratio"],
                          right_on=["Matrix Name", "Sparsification Ratio"], how="inner")

        if merged.empty:
            print("[WARN] Merged result is empty. Skipping this threshold combo.")
            continue

        if merged["Per-iteration Speedup"].isnull().any():
            print("[WARN] Null values in speedup column found. Dropping those.")
            merged = merged.dropna(subset=["Per-iteration Speedup"])

        if merged.empty:
            continue

        pi_gmean = gmean(merged["Per-iteration Speedup"])
        total_checked = len(merged)
        converged_alg = merged["Unaffectedly Converging"].sum()

        orig_merged = pd.merge(result_df[["Matrix Name"]], full_df[full_df["Sparsification Ratio"] == 0],
                               on="Matrix Name", how="left")
        converged_orig = orig_merged["Originally Converging"].sum()

        percent_alg = 100 * converged_alg / total_checked if total_checked else 0.0
        percent_orig = 100 * converged_orig / total_checked if total_checked else 0.0

        results.append((threshold, threshold_wf, pi_gmean, percent_alg, percent_orig))

        if pi_gmean > best_gmean:
            best_gmean = pi_gmean
            best_combo = (threshold, threshold_wf)
            best_df = result_df.copy()

    top_results = sorted(results, key=lambda x: x[2], reverse=True)[:5]
    print("\n Top 5 Threshold Combinations by gmean (with convergence %):")
    for i, (t, wf, gm, pc_alg, pc_orig) in enumerate(top_results, 1):
        print(f"{i}. threshold={t:<6} wf_thresh={wf:<4} gmean={gm:.6f}  converged(alg): {pc_alg:.1f}% | orig: {pc_orig:.1f}%")

    if best_df is not None:
        # Merge with full_df again to attach speedup columns
        export_df = pd.merge(
            best_df,
            full_df[["Matrix Name", "Sparsification Ratio", "Per-iteration Speedup", "End-to-end Speedup"]],
            left_on=["Matrix Name", "Selected Sparsification Ratio"],
            right_on=["Matrix Name", "Sparsification Ratio"],
            how="left"
        )

        export_df = export_df[
            ["Matrix Name", "Selected Sparsification Ratio", "Per-iteration Speedup", "End-to-end Speedup"]]
        export_df.to_csv("best_oracle_selection_gpu_pi_from_prediction_cpu.csv", index=False)

        print("Saved best selection to 'best_oracle_selection_gpu_pi_from_prediction_cpu.csv' with speedups")

    if best_combo:
        print(f"\nBest Threshold Combination: threshold={best_combo[0]}, threshold_wf={best_combo[1]}")
        print(f"Best Geometric Mean Per-iteration Speedup: {best_gmean:.6f}")
    else:
        print("No valid threshold combination found. Check input data or parameter range.")
