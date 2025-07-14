#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

//--------------------------------------------------------------
// Utility: Read a Matrix Market file (coordinate format) into a sparse matrix.
bool read_mtx_eigen(const std::string &filename, Eigen::SparseMatrix<float> &mat) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "cannot open " << filename << "\n";
        return false;
    }
    bool symmetric = false;
    std::string line;
    while (true) {
        if (!std::getline(in, line)) {
            std::cerr << "bad file header\n";
            return false;
        }
        if (!line.empty() && line[0] == '%') {
            if (line.find("symmetric") != std::string::npos)
                symmetric = true;
            continue;
        }
        break;
    }
    int m, n, nnz;
    {
        std::stringstream ss(line);
        ss >> m >> n >> nnz;
    }
    std::vector<Eigen::Triplet<float>> trips;
    trips.reserve(symmetric ? nnz * 2 : nnz);
    for (int i = 0; i < nnz; i++) {
        int r, c;
        float v;
        in >> r >> c >> v;
        --r; // convert to 0-based
        --c;
        trips.emplace_back(r, c, v);
        if (symmetric && r != c)
            trips.emplace_back(c, r, v);
    }
    in.close();
    mat.resize(m, n);
    mat.setFromTriplets(trips.begin(), trips.end());
    return true;
}

//--------------------------------------------------------------
// MyWavefrontILU0Preconditioner implements ILU0 factorization with a
// custom wavefront-level parallel triangular solve.
template<typename Scalar>
class MyWavefrontILU0Preconditioner {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;

    MyWavefrontILU0Preconditioner() : computed(false), num_levels_L(0), num_levels_U(0) {}

    template<typename MatType>
    MyWavefrontILU0Preconditioner& compute(const MatType& A_in) {
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A = A_in;
        const int n = A.rows();
        std::vector<std::vector<std::pair<int, Scalar>>> L_rows(n), U_rows(n);

        std::vector<Scalar> x(n, 0);
        std::vector<int> pattern;
        pattern.reserve(n);

        // --- ILU(0) factorization ---
        for (int i = 0; i < n; i++) {
            pattern.clear();
            for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
                int j = it.col();
                x[j] = it.value();
                pattern.push_back(j);
            }
            std::sort(pattern.begin(), pattern.end());

            for (int idx = 0; idx < (int)pattern.size(); idx++) {
                int k = pattern[idx];
                if (k >= i) break;
                Scalar U_kk = 0;
                for (auto &entry : U_rows[k]) {
                    if (entry.first == k) {
                        U_kk = entry.second;
                        break;
                    }
                }
                if (U_kk == 0) continue;
                Scalar multiplier = x[k] / U_kk;
                L_rows[i].push_back({k, multiplier});
                for (auto &entry : U_rows[k]) {
                    int col = entry.first;
                    if (col > k) {
                        if (std::binary_search(pattern.begin(), pattern.end(), col))
                            x[col] -= multiplier * entry.second;
                    }
                }
            }
            for (int j : pattern) {
                if (j >= i) {
                    U_rows[i].push_back({j, x[j]});
                }
                x[j] = 0;
            }
            L_rows[i].push_back({i, Scalar(1)});
        }

        // --- Convert L_rows, U_rows to sparse L and U ---
        std::vector<Eigen::Triplet<Scalar>> L_triplets, U_triplets;
        L_triplets.reserve(A.nonZeros());
        U_triplets.reserve(A.nonZeros());
        for (int i = 0; i < n; i++) {
            for (auto &p : L_rows[i]) {
                L_triplets.push_back(Eigen::Triplet<Scalar>(i, p.first, p.second));
            }
            for (auto &p : U_rows[i]) {
                U_triplets.push_back(Eigen::Triplet<Scalar>(i, p.first, p.second));
            }
        }
        L.resize(n, n);
        U.resize(n, n);
        L.setFromTriplets(L_triplets.begin(), L_triplets.end());
        U.setFromTriplets(U_triplets.begin(), U_triplets.end());

        // --- Compute level scheduling for forward substitution on L ---
        level_L.resize(n, 0);
        for (int i = 0; i < n; i++) {
            int lvl = 0;
            for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(L, i); it; ++it) {
                int j = it.col();
                if (j < i) {
                    lvl = std::max(lvl, level_L[j] + 1);
                }
            }
            level_L[i] = lvl;
        }
        num_levels_L = 0;
        for (int i = 0; i < n; i++)
            num_levels_L = std::max(num_levels_L, level_L[i]);
        num_levels_L++;

        // --- Compute level scheduling for backward substitution on U ---
        level_U.resize(n, 0);
        for (int i = n - 1; i >= 0; i--) {
            int lvl = 0;
            for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(U, i); it; ++it) {
                int j = it.col();
                if (j > i) {
                    lvl = std::max(lvl, level_U[j] + 1);
                }
            }
            level_U[i] = lvl;
        }
        num_levels_U = 0;
        for (int i = 0; i < n; i++)
            num_levels_U = std::max(num_levels_U, level_U[i]);
        num_levels_U++;

        computed = true;
        return *this;
    }

    // Solve using wavefront parallelism for triangular solves
    template<typename Derived>
    VectorType solve(const Eigen::MatrixBase<Derived>& b_in) const {
        assert(computed);
        const int n = b_in.size();
        VectorType y(n);
        y = b_in;

        // --- Forward substitution: L * y = b ---
        for (int lev = 0; lev < num_levels_L; lev++) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) schedule(auto)
#endif
            for (int i = 0; i < n; i++) {
                if (level_L[i] == lev) {
                    Scalar sum = 0;
                    for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(L, i); it; ++it) {
                        int j = it.col();
                        if (j < i) {
                            sum += it.value() * y(j);
                        }
                    }
                    y(i) = y(i) - sum;
                }
            }
        }

        // --- Backward substitution: U * x = y ---
        VectorType x(n);
        x = y;
        for (int lev = 0; lev < num_levels_U; lev++) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) schedule(auto)
#endif
            for (int i = 0; i < n; i++) {
                if (level_U[i] == lev) {
                    Scalar sum = 0;
                    for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(U, i); it; ++it) {
                        int j = it.col();
                        if (j > i) {
                            sum += it.value() * x(j);
                        }
                    }
                    x(i) = (x(i) - sum) / getDiagonal(U, i);
                }
            }
        }
        return x;
    }

    Eigen::ComputationInfo info() const {
        return computed ? Eigen::Success : Eigen::NumericalIssue;
    }

private:
    // Helper: retrieve the diagonal element of U in row i.
    Scalar getDiagonal(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor>& M, int i) const {
        for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(M, i); it; ++it) {
            if (it.col() == i)
                return it.value();
        }
        return Scalar(1);
    }

    // Data members
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> L, U;
    std::vector<int> level_L, level_U;
    int num_levels_L, num_levels_U;
    bool computed;
};

//--------------------------------------------------------------
// Main: read A (for Ax=b) and B (for ILU0(B)), run PCG, log A's info only.
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " path/to/A.mtx path/to/B.mtx [serial|parallel]\n";
        return 1;
    }

    std::string fullpathA(argv[1]);
    size_t slash_pos = fullpathA.find_last_of("/\\");
    std::string fnameA = (slash_pos == std::string::npos) ? fullpathA : fullpathA.substr(slash_pos + 1);
    if (fnameA.size() >= 4 && fnameA.compare(fnameA.size() - 4, 4, ".mtx") == 0) {
        fnameA.erase(fnameA.size() - 4);
    }

    std::string matrixNameA = fnameA;

    std::string fullpathB(argv[2]);
    slash_pos = fullpathB.find_last_of("/\\");
    std::string fnameB = (slash_pos == std::string::npos) ? fullpathB : fullpathB.substr(slash_pos + 1);
    if (fnameB.size() >= 4 && fnameB.compare(fnameB.size() - 4, 4, ".mtx") == 0) {
        fnameB.erase(fnameB.size() - 4);
    }

    std::string sparRatio = "N/A";
    if (fnameB.size() > 4) {
        std::string suffix = fnameB.substr(fnameB.size() - 5); // e.g. "_0.05"
        if (suffix == "_0.01" || suffix == "_0.05" || suffix == "_0.10") {
            sparRatio = suffix.substr(1); // remove underscore => "0.05"
        }
    }

    std::string csv_filename = "cpu_results_" + matrixNameA + ".csv";

    std::string parallelism;
    omp_set_num_threads(8);

#ifdef _OPENMP
    if (argc >= 4) {
        std::string mode = argv[3];
        if (mode == "serial") {
            omp_set_num_threads(1);
            parallelism = "Serial";
            std::cout << "Using Serial mode (forced via command-line)." << std::endl;
        } else if (mode == "parallel") {
            parallelism = "OpenMP";
            std::cout << "Using Parallel mode." << std::endl;
        } else {
            parallelism = "OpenMP";
            std::cout << "Unknown mode specified; defaulting to Parallel mode." << std::endl;
        }
    } else {
        parallelism = "OpenMP";
        std::cout << "No mode specified; defaulting to Parallel mode." << std::endl;
    }
#else
    parallelism = "Serial";
    std::cout << "OpenMP not available; running in Serial mode." << std::endl;
#endif

    Eigen::SparseMatrix<float> A;
    if (!read_mtx_eigen(argv[1], A))
        return 1;
    std::cout << "[A] loaded: " << A.rows() << " x " << A.cols()
              << ", nnz=" << A.nonZeros() << "\n";

    Eigen::SparseMatrix<float> B;
    if (!read_mtx_eigen(argv[2], B))
        return 1;
    std::cout << "[B] loaded: " << B.rows() << " x " << B.cols()
              << ", nnz=" << B.nonZeros() << "\n";

    Eigen::VectorXf bvec(A.rows());
    bvec.setOnes();
    Eigen::VectorXf x(A.rows());
    x.setZero();

    typedef MyWavefrontILU0Preconditioner<float> MyILU;
    MyILU iluB;

    auto factor_start = std::chrono::high_resolution_clock::now();
    iluB.compute(B);
    auto factor_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> factor_time = factor_end - factor_start;

    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper, MyILU> solver;
    solver.setMaxIterations(1000);
    float abs_tol = 1e-12f;
    solver.setTolerance(abs_tol);

    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "solver.compute(A) failed.\n";
        return 1;
    }

    solver.preconditioner() = iluB;

    // Solve
    auto solve_start = std::chrono::high_resolution_clock::now();
    x = solver.solve(bvec);
    auto solve_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> solve_time = solve_end - solve_start;

    if (solver.info() != Eigen::Success) {
        std::cerr << "solve failed\n";
    }

    int iters = solver.iterations();
    double finalError = solver.error();

    std::cout << "\n--------------------------------------\n"
              << "Solving A*x=b with ILU(B) preconditioner.\n"
              << "Matrix A: " << matrixNameA
              << "  (Rows=" << A.rows() << ", Cols=" << A.cols() << ", nnz=" << A.nonZeros() << ")\n"
              << "Matrix B: " << fnameB << " (for ILU)\n"
              << "Sparsification ratio from B => " << sparRatio << "\n"
              << "Final residual: " << finalError << "\n"
              << "Iterations: " << iters << "\n"
              << "CG Solve Time (ms): " << solve_time.count() << "\n"
              << "ILU(B) Factor Time (ms): " << factor_time.count() << "\n"
              << "Parallelism: " << parallelism << "\n";

    // CSV logging:
    //    "Matrix Name,Sparsification ratio,Rows,Cols,Nonzeros,Final Residual,Iterations Spent,CG Time (ms),Factorization Time (ms),Parallelism"
    {
        std::ofstream csv_file(csv_filename, std::ios::app);
        if (!csv_file) {
            std::cerr << "cannot open " << csv_filename << " for appending\n";
            return 0;
        }
        csv_file.seekp(0, std::ios::end);
        if (csv_file.tellp() == 0) {
            csv_file << "Matrix Name,Sparsification ratio,Rows,Cols,Nonzeros,"
                     << "Final Residual,Iterations Spent,CG Time (ms),Factorization Time (ms),Parallelism\n";
        }
        csv_file << matrixNameA << "," << sparRatio << ","
                 << A.rows() << "," << A.cols() << "," << A.nonZeros() << ","
                 << finalError << "," << iters << ","
                 << solve_time.count() << "," << factor_time.count() << ","
                 << parallelism << "\n";
    }

    return 0;
}
