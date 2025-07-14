import pandas as pd
from scipy.stats import gmean
import numpy as np
import itertools

def select_sparsification_ratio(matrix_name, threshold, threshold_wf, cond_df, norm2_A_df, norm2_S_df, wf_df):

    for ratio in [0.1, 0.05, 0.01]:
        try:
            k_A = cond_df.loc[(cond_df["Matrix Name"] == matrix_name) & (
                        cond_df["Sparsification Ratio"] == ratio), "Approximated Condition Number"].values[0]
            norm_A = norm2_A_df.loc[(norm2_A_df["Matrix Name"] == matrix_name) & (norm2_A_df["Sparsification Ratio"] == ratio), "2-norm of Aos"].values[0]
            inverse_norm = k_A / norm_A
            num_wf_o = wf_df.loc[
                (wf_df["Matrix Name"] == matrix_name) & (wf_df["Sparsification Ratio"] == 0), "Wavefront Count"].values[
                0]
        except:
            continue

        try:
            norm_s = norm2_S_df.loc[
                (norm2_S_df["Matrix Name"] == matrix_name) & (norm2_S_df["Sparsification Ratio"] == ratio),
                "2-norm of E"
            ].values[0]
            num_wf_os = wf_df.loc[
                (wf_df["Matrix Name"] == matrix_name) & (wf_df["Sparsification Ratio"] == ratio),
                "Wavefront Count"
            ].values[0]
        except:
            continue

        product = inverse_norm * norm_s
        if product >= threshold:
            if ratio == 0.01:
                return 0.1
            else:
                continue

        wf_reduction = ((num_wf_o - num_wf_os) / num_wf_o) * 100
        if wf_reduction >= threshold_wf:
            return ratio

        if ratio == 0.01:
            return 0.1

    return 0.01

if __name__ == "__main__":
    # Load all data from logs directory
    speedup_df = pd.read_csv("../../../logs/iluk_speedups_best_fill_factor.csv")
    cond_df = pd.read_csv("../../../logs/approximated_condition_number_inf.csv")
    norm2_A_df = pd.read_csv("../../../logs/norm2_os.csv")
    norm2_S_df = pd.read_csv("../../../logs/norm2_s.csv")
    wf_df = pd.read_csv("../../../logs/wavefronts.csv")
    raw_df = pd.read_csv("../../../logs/iluk_raw_best_fill_factor.csv")
    app_df = pd.read_csv("../../../logs/matrix_application.csv")

    valid_grouped = speedup_df.groupby("Matrix Name").filter(lambda x: len(x) == 3)
    all_matrices = sorted(valid_grouped["Matrix Name"].unique())

    threshold_list = [1]
    threshold_wf_list = [10, 15, 20, 25, 30, 40, 50]

    results = []
    best_gmean = -np.inf
    best_combo = None
    best_df = None

    for threshold, threshold_wf in itertools.product(threshold_list, threshold_wf_list):
        selected = []

        for matrix in all_matrices:
            ratio = select_sparsification_ratio(matrix, threshold, threshold_wf, cond_df, norm2_A_df, norm2_S_df, wf_df)
            if ratio is not None:
                selected.append((matrix, ratio))

        result_df = pd.DataFrame(selected, columns=["Matrix Name", "Selected Sparsification Ratio"])
        merged = pd.merge(result_df, speedup_df, left_on=["Matrix Name", "Selected Sparsification Ratio"],
                          right_on=["Matrix Name", "Sparsification Ratio"])

        if not merged.empty:
            pi_gmean = gmean(merged["Per-iteration Speedup"])

            converged_alg = 0
            converged_orig = 0
            total_checked = 0

            for _, row in result_df.iterrows():
                matrix = row["Matrix Name"]
                ratio = row["Selected Sparsification Ratio"]

                try:
                    iter_alg = raw_df.loc[
                        (raw_df["Matrix Name"] == matrix) & (raw_df["Sparsification Ratio"] == ratio),
                        "Iterations Spent"
                    ].values[0]

                    iter_orig = raw_df.loc[
                        (raw_df["Matrix Name"] == matrix) & (raw_df["Sparsification Ratio"] == 0),
                        "Iterations Spent"
                    ].values[0]

                    total_checked += 1
                    if iter_alg < 1000:
                        converged_alg += 1
                    if iter_orig < 1000:
                        converged_orig += 1

                except:
                    continue

            if total_checked > 0:
                percent_alg = 100 * converged_alg / total_checked
                percent_orig = 100 * converged_orig / total_checked
            else:
                percent_alg = percent_orig = 0.0

            results.append((threshold, threshold_wf, pi_gmean, percent_alg, percent_orig))

            if pi_gmean > best_gmean:
                best_gmean = pi_gmean
                best_combo = (threshold, threshold_wf)
                best_df = result_df.copy()

    top_results = sorted(results, key=lambda x: x[2], reverse=True)[:5]
    print("\n Top 5 Threshold Combinations by gmean (with convergence %):")
    for i, (t, wf, gm, pc_alg, pc_orig) in enumerate(top_results, 1):
        print(
            f"{i}. threshold={t:<6} wf_thresh={wf:<4} gmean={gm:.6f}  converged(alg): {pc_alg:.1f}% | orig: {pc_orig:.1f}%")

    if best_df is not None:
        best_df.to_csv("best_oracle_selection_gpu_pi_from_prediction.csv", index=False)

        # NEW: Compute percentage of matrices with speedup > 1
        merged_best = pd.merge(
            best_df, speedup_df,
            left_on=["Matrix Name", "Selected Sparsification Ratio"],
            right_on=["Matrix Name", "Sparsification Ratio"],
            how="inner"
        )
        if not merged_best.empty:
            count_speedup_gt_1 = (merged_best["Per-iteration Speedup"] > 1).sum()
            percent_speedup_gt_1 = 100 * count_speedup_gt_1 / len(merged_best)
            print(
                f"\nPercentage of algorithm-selected matrices with per-iteration speedup > 1: {percent_speedup_gt_1:.2f}%")

    if best_combo:
        print(f"\nBest Threshold Combination: threshold={best_combo[0]}, threshold_wf={best_combo[1]}")
        print(f"Best Geometric Mean Per-iteration Speedup: {best_gmean:.6f}")
    else:
        print("No valid threshold combination found. Check input data or parameter range.")

    print(f"\nBest Threshold Combination: threshold={best_combo[0]}, threshold_wf={best_combo[1]}")
    print(f"Best Geometric Mean Per-iteration Speedup: {best_gmean:.6f}")

    # ----- New Feature: Gmean Speedup per Application -----
    final_merged = pd.merge(best_df, speedup_df, left_on=["Matrix Name", "Selected Sparsification Ratio"],
                            right_on=["Matrix Name", "Sparsification Ratio"])
    final_merged = pd.merge(final_merged, app_df, on="Matrix Name")

    grouped = final_merged.groupby("Application")["Per-iteration Speedup"].agg(gmean).reset_index()
    grouped.columns = ["Application", "Gmean Speedup"]
    grouped.to_csv("gmean_speedup_by_application_iluk.csv", index=False)
    print("\nExported Gmean Speedup per Application to 'gmean_speedup_by_application.csv'")
