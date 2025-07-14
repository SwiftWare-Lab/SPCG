import scipy.sparse as sp
import scipy.sparse.linalg as ss
import scipy.io as si
import numpy as np
import sys
import argparse
import os
import time
import csv


def compute_topological_sort_csr(matrix_csr):
    EMPTY = 1
    n = matrix_csr.shape[0]
    in_degrees = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(matrix_csr.indptr[i], matrix_csr.indptr[i + 1]):
            in_degrees[matrix_csr.indices[j]] += 1
    stack = []
    for i in range(n):
        if in_degrees[i] == EMPTY:
            stack.append(i)
    wavefront_set = []
    node_count = 0
    while node_count < n:
        cur_stack = stack.copy()
        wavefront_set.append(np.array(stack))
        stack = []
        while cur_stack:
            node = cur_stack.pop()
            node_count += 1
            for j in range(matrix_csr.indptr[node], matrix_csr.indptr[node + 1]):
                neighbor = matrix_csr.indices[j]
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == EMPTY:
                    stack.append(neighbor)
    return wavefront_set


def save_matrix_mtx(matrix, filename, precision=40):
    matrix = matrix.tocoo()

    with open(filename, 'w') as f:
        f.write('%%MatrixMarket matrix coordinate real general\n')
        f.write(f'% precision: {precision} digits\n')
        f.write(f'{matrix.shape[0]} {matrix.shape[1]} {matrix.nnz}\n')

        format_str = f'{{}} {{}} {{:.{precision}e}}\n'
        for i, j, v in zip(matrix.row, matrix.col, matrix.data):
            f.write(format_str.format(i + 1, j + 1, v))  # Matrix Market format is 1-based index


def remove_below_threshold(matrix, threshold):
    matrix = matrix.tolil()
    rows, cols = matrix.nonzero()
    for row, col in zip(rows, cols):
        if row != col and np.abs(matrix[row, col]) < threshold:
            matrix[row, col] = 0
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    return matrix


def count_small_values(matrix, threshold):
    return np.sum(np.abs(matrix.data) < threshold)


def factorization(A, choice, fill_factor, drop_threshold):
    factors = None

    try:
        if choice == 0:
            factors = ss.splu(A.tocsc(), permc_spec='NATURAL', diag_pivot_thresh=1e-15)
        elif choice == 1:
            factors = ss.spilu(A.tocsc(), fill_factor=fill_factor, permc_spec='NATURAL', diag_pivot_thresh=1e-15)
        elif choice == 2:
            factors = ss.spilu(A.tocsc(), fill_factor=fill_factor, drop_tol=drop_threshold, permc_spec='NATURAL', diag_pivot_thresh=1e-15)
        elif choice == 3:
            factors = ss.spilu(A.tocsc(), drop_tol=0, fill_factor=1, permc_spec='NATURAL', diag_pivot_thresh=1e-15)
    except RuntimeError as e:
        print(f"Error during ILU factorization")
        return factors

    return factors


def analyze_and_determine_threshold(matrix, percentage=0.1):
    matrix_coo = matrix.tocoo()
    n = matrix.shape[0]

    dia_count = np.zeros(n, dtype=int)

    for i, j in zip(matrix_coo.row, matrix_coo.col):
        diagonal_index = abs(i - j)
        if diagonal_index < n:
            dia_count[diagonal_index] += 1

    abs_values = np.abs(matrix_coo.data)

    sorted_abs_values = np.sort(abs_values)

    threshold_index = int(len(sorted_abs_values) * percentage)
    removal_threshold = sorted_abs_values[threshold_index]

    print(f'Determined removal threshold: {removal_threshold}')
    print(f'NNZ per diagonal: {dia_count}')

    return removal_threshold


def sparsify_matrix(matrix, threshold):
    matrix_coo = matrix.tocoo()
    n = matrix.shape[0]
    nnz_count = matrix.nnz

    count_eliminated = 0
    max_val = -np.inf
    min_val = np.inf
    min_abs_val = np.inf

    row, col, data = matrix_coo.row, matrix_coo.col, matrix_coo.data
    for i in range(len(data)):
        if row[i] != col[i]:  # Skip diagonal elements
            if data[i] < min_val:
                min_val = data[i]
            if abs(data[i]) < min_abs_val:
                min_abs_val = abs(data[i])
            if data[i] > max_val:
                max_val = data[i]
            if abs(data[i]) <= threshold:
                data[i] = 0
                count_eliminated += 1

    print(f'Eliminated NNZ: {count_eliminated} ({(count_eliminated * 100.0) / nnz_count}%)')
    print(f'Max value: {max_val}, Min value: {min_val}, Min absolute value: {min_abs_val}')

    matrix_coo = sp.coo_matrix((data, (row, col)), shape=(n, n))
    matrix_csr = matrix_coo.tocsr()
    matrix_csr.eliminate_zeros()

    return matrix_csr


def process_fill_factor(fil_factor, A, sp, threshold, percentage, choice, matrix_name, drop_tol=1e-12):
    factors = factorization(A, choice=choice, fill_factor=fil_factor, drop_threshold=drop_tol)

    if factors is None:
        print(f"Factorization failed for matrix {matrix_name}")
        return None
        
    L_nnz_count = factors.L.nnz
    U_nnz_count = factors.U.nnz

    print(f"L_nnz_count: {L_nnz_count}, U_nnz_count: {U_nnz_count}")

    L = factors.L
    U = factors.U

    exp_tag = "default"

    if sp:
        if choice == 0:
            exp_tag = "splu_sp"
        elif choice == 1:
            exp_tag = "spilu_sp"
    else:
        if choice == 0:
            exp_tag = "splu_nonsp"
        elif choice == 1:
            exp_tag = "spilu_nonsp"

    # Create factors directory if it doesn't exist
    factors_dir = "../../../../factors"
    os.makedirs(factors_dir, exist_ok=True)
    
    si.mmwrite(f"{factors_dir}/{exp_tag}_{matrix_name}_l_{fil_factor}_{sp}_{percentage}.mtx", L)
    si.mmwrite(f"{factors_dir}/{exp_tag}_{matrix_name}_u_{fil_factor}_{sp}_{percentage}.mtx", U)

    print(f"Finished exporting for: fill factor of {fil_factor}, sparsification choice of {sp}, threshold of {threshold}, percentage of {percentage}.")
    return factors


def process_fill_factor_with_timing(fil_factor, A, sparsified, threshold, percentage_choice, choice, matrix_name,
                                    writer):
    start_time = time.time()
    factors = process_fill_factor(fil_factor, A, sparsified, threshold, percentage_choice, choice, matrix_name)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if factors is None:
        # Write row with N/A values for wavefront counts if factorization failed
        writer.writerow([matrix_name, fil_factor, sparsified, percentage_choice, threshold, 'N/A', 'N/A', elapsed_time])
        return

    L = factors.L
    wf_set = compute_topological_sort_csr(L.tocsr())
    wavefront_count = len(wf_set)

    # Only compute sparsified wavefronts if percentage_choice > 0
    if percentage_choice > 0:
        removal_threshold = analyze_and_determine_threshold(L, percentage=percentage_choice)
        L_sp = sparsify_matrix(L, removal_threshold)
        wf_set_after = compute_topological_sort_csr(L_sp.tocsr())
        wavefront_count_after = len(wf_set_after)
    else:
        # For non-sparsified matrices, use the same wavefront count
        wavefront_count_after = wavefront_count

    print(f"Wavefront count for {'sparsified' if sparsified else 'non-sparsified'} matrix: Before: {wavefront_count}; After: {wavefront_count_after}")
    writer.writerow([matrix_name, fil_factor, sparsified, percentage_choice, threshold, wavefront_count, wavefront_count_after, elapsed_time])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some matrices.')
    parser.add_argument('matrix_file', type=str, help='Path to the matrix file in .mtx format')
    parser.add_argument('--export', action='store_true', help='Export the ILU factors and exit')

    args = parser.parse_args()

    matrix_name = os.path.splitext(os.path.basename(args.matrix_file))[0]

    try:
        A = si.mmread(args.matrix_file)
    except MemoryError as e:
        print(f"Memory error when reading the matrix: {e}")
        sys.exit(1)

    n = A.shape[0]
    b = np.ones(A.shape[0])

    fill_factors = [10, 20, 30]
    removal_percentage = [0.01, 0.05, 0.1]

    # Create timing directory in logs if it doesn't exist
    timing_dir = "../../../../factors/timing"
    os.makedirs(timing_dir, exist_ok=True)

    with open(f'{timing_dir}/{matrix_name}_timing.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Matrix Name', 'Fill Factor', 'Sparsified', 'Removal Percentage', 'Threshold', 'Time (s)'])
        writer.writerow(['Matrix Name', 'Fill Factor', 'Sparsified', 'Removal Percentage', 'Threshold', '#Wavefront Set', '#Wavefront Set Sp', 'Time (s)'])
        for fil_factor in fill_factors:
            process_fill_factor_with_timing(fil_factor, A, False, 0, 0, choice=1, matrix_name=matrix_name, writer=writer)

        for percentage_choice in removal_percentage:
            removal_threshold = analyze_and_determine_threshold(A, percentage=percentage_choice)
            A_sparsified = sparsify_matrix(A, removal_threshold)

            for fil_factor in fill_factors:
                process_fill_factor_with_timing(fil_factor, A_sparsified, True, removal_threshold, percentage_choice, choice=1, matrix_name=matrix_name, writer=writer)
