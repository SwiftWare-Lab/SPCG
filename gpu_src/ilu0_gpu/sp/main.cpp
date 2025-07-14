/*
module load cuda/12.2
module load gcc/8.5
module load cmake

M/N --> nB
nz --> nnzB
I --> colB
J --> rowB
val --> valB
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <iterator>
#include "util.h"

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // shared functions common to CUDA Samples
#include <helper_cuda.h>       // CUDA error checking

using namespace std;

/*
 * Solve Ax=b using the conjugate gradient method
 * a) without any preconditioning,
 * b) using an Incomplete Cholesky preconditioner, and
 * c) using an ILU0 preconditioner.
 */
int main(int argc, char **argv) {
    // Check if the path to the .mtx file is provided as an argument
    if (argc < 3) {
        fprintf(stderr, "Error: Please provide the path to the .mtx file and sparsification ratio as arguments.\n");
        return 1;
    }

    const float sparsification_ratio = atof(argv[2]);

    const int max_iter = 1000; //default 1000
    int k, *I = NULL, *J = NULL;
    size_t M = 0, N = 0, nz = 0; // int to
    int *d_col, *d_row;
    int *d_colB, *d_rowB;
    int qatest = 0;
    const float tol = 1e-12f;
    float *x, *rhs;
    float *xB, *rhsB;
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_valB, *d_xB;
    float *d_zm1, *d_zm2, *d_rm2;
    float *d_zm1B, *d_zm2B, *d_rm2B;
    float *d_r, *d_p, *d_omega, *d_y;
    float *d_rB, *d_pB, *d_omegaB, *d_yB;
    float *val = NULL;
    float *d_valsILU0;
    float *d_valsILU0B;
    float rsum, diff, err = 0.0;
    float qaerr1, qaerr2 = 0.0;
    float qaerr1B, qaerr2B = 0.0;
    float dot, numerator, denominator, nalpha;
    const float floatone = 1.0;
    const float floatzero = 0.0;
    int nErrors = 0;

    printf("conjugateGradientPrecond starting...\n");

    /* This will pick the best possible CUDA capable device */
    cudaDeviceProp deviceProp;
    int devID = 0;
    findCudaDevice(argc, (const char **)argv);
    printf("GPU selected Device ID = %d \n", devID);

    if (devID < 0) {
        printf("Invalid GPU device %d selected,  exiting...\n", devID);
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    /* Statistics about the GPU device */
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    /* Generate a Laplace matrix in CSR (Compressed Sparse Row) format */
    std::string inputMatrix(argv[1]);

    if (!readMatrix(inputMatrix, M, nz, I, J, val)) {
        std::cout << "Read FAILED\n";
        return -1;
    }
    N = M;
    make_full(M, nz, I, J, val);

    std::cout << "Finished reading matrix:  " << inputMatrix << " dimension: " << M << "x" << N << " nnz= " << nz << "\n";

    x = (float *)malloc(sizeof(float) * N);
    rhs = (float *)malloc(sizeof(float) * N);

//     // COO
//     int *colC;
//     int *rowC;
//     float *valC;
//     size_t nC, nnzC;
//     if (!readMatrixCOO(inputMatrix, nC, nnzC, colC, rowC, valC)) return -1;
//     std::cout << "COO dimension: " << nC << " nnz: " << nnzC << "\n";

//     int sym_dia = nC;
//     int dia[sym_dia] = {0};

//     // Find NNZ per diagonal
//     for (size_t i = 0; i < nnzC; ++i) {
//         int tmp = std::abs(rowC[i] - colC[i]);
//         if (tmp > sym_dia)
//             std::cout << " Larger than half number of diagonals " << tmp << " from " << sym_dia << "\n";
//         dia[tmp] = dia[tmp] + 1;
//     }

//     // The 10% poorest values
//     double arr[nnzC];
//     for (int n = 0; n < nnzC; n++) {
//         arr[n] = abs(valC[n]);
//     }
//     int nn = sizeof(arr) / sizeof(arr[0]);
//     sort(arr, arr + nn);
//     int range = nnzC * sparsification_ratio; // Use input ratio instead of fixed value
//     double Rval = arr[range];
//     std::cout << "Abs values less than abs Rval <= " << Rval << " will perish\n";

//     int count_ele = 0;
//     double mox = std::numeric_limits<double>::min();
//     double mon = std::numeric_limits<double>::max();
//     double mon_abs = std::numeric_limits<double>::max();
//     std::cout << "Total NNZ " << nnzC << "\n";

//     for (size_t i = 0; i < nnzC; ++i) {
//         if (rowC[i] == colC[i]) continue; // skip diagonal elements
//         if (valC[i] < mon) mon = valC[i];
//         if (abs(valC[i]) < mon_abs) mon_abs = abs(valC[i]);
//         if (valC[i] > mox) mox = valC[i];
//         if (abs(valC[i]) <= Rval) {
//             count_ele++;
//             valC[i] = 0;
//         } else if (valC[i] == 0) count_ele++;
//     }
//     std::cout << "Eliminated NNZ= " << count_ele << " % removed= " << (count_ele * 100.0) / nnzC << "\n";
//     std::cout << "Max= " << mox << " Min= " << mon << " abs[Min]= " << mon_abs << "\n";

//     ofstream out(inputMatrix + "banded");
//     out << "%%MatrixMarket matrix coordinate real symmetric" << endl;
//     out << nC << " " << nC << " " << nnzC - count_ele << endl;
//     for (int i = 0; i < nnzC; ++i) {
//         if (valC[i] == 0) continue;
//         out << rowC[i] + 1 << " " << colC[i] + 1 << " " << std::scientific << valC[i] << endl;
//     }
//     out.close();
//     std::cout << "Done writing bounded matrix to file\n";

//     int *colB;
//     int *rowB;
//     float *valB;
//     size_t nB, nnzB;
//     if (!readMatrix(inputMatrix + "_" + argv[2], nB, nnzB, colB, rowB, valB)) return -1;

//     make_full(nB, nnzB, colB, rowB, valB);

    // Get the absolute path to the matrix file
    std::string matrix_path = argv[1];

    // Find the last '/' in the path
    size_t last_slash = matrix_path.find_last_of("/\\");

    // Extract the matrix name (without extension)
    std::string matrix_name = matrix_path.substr(last_slash + 1, matrix_path.find_last_of('.') - last_slash - 1);

    // Extract the directory path (including the trailing slash)
    std::string matrix_dir = matrix_path.substr(0, last_slash + 1);

    // Construct the output CSV file name
    std::string csv_filename = "residuals_" + matrix_name + ".csv";

    // Open CSV file for writing
    std::ofstream outFile(csv_filename);
    // Write the header
    outFile << "Iteration,Residual,Time\n";

    std::ostringstream ratio_stream;
    ratio_stream << std::fixed << std::setprecision(2) << sparsification_ratio;
    std::string matrix_sp = matrix_dir + matrix_name + "_" + ratio_stream.str() + ".mtx";
    std::cout << "Reading matrix: " << matrix_sp << "\n";

    int *colB;
    int *rowB;
    float *valB;
    size_t nB, nnzB;
    if (!readMatrix(matrix_sp, nB, nnzB, colB, rowB, valB)) return -1;

    make_full(nB, nnzB, colB, rowB, valB);

    xB = (float *)malloc(sizeof(float) * nB);
    rhsB = (float *)malloc(sizeof(float) * nB);

    for (int i = 0; i < N; i++) // N = rows or columns size
    {
        rhs[i] = 1.0;  // Initialize RHS
        rhsB[i] = 1.0;
        x[i] = 0.0;    // Initial solution approximation
        xB[i] = 0.0;
    }

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    checkCudaErrors(cublasCreate(&cublasHandle));
    cublasHandle_t cublasHandleB = NULL;
    checkCudaErrors(cublasCreate(&cublasHandleB));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    checkCudaErrors(cusparseCreate(&cusparseHandle));
    cusparseHandle_t cusparseHandleB = NULL;
    checkCudaErrors(cusparseCreate(&cusparseHandleB));

    /* Description of the A matrix */
    cusparseMatDescr_t descr = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    checkCudaErrors(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    cusparseMatDescr_t descrB = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descrB));
    checkCudaErrors(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate required memory */
    checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_y, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_omega, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_zm1, (N) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_zm2, (N) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_rm2, (N) * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_colB, nnzB * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_rowB, (nB + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_valB, nnzB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_xB, nB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_yB, nB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_rB, nB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_pB, nB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_omegaB, nB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_valsILU0B, nnzB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_zm1B, (nB) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_zm2B, (nB) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_rm2B, (nB) * sizeof(float)));

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseDnVecDescr_t vecp = NULL, vecX = NULL, vecY = NULL, vecR = NULL, vecZM1 = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecR, N, d_r, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecZM1, N, d_zm1, CUDA_R_32F));
    cusparseDnVecDescr_t vecomega = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecomega, N, d_omega, CUDA_R_32F));

    cusparseDnVecDescr_t vecpB = NULL, vecXB = NULL, vecYB = NULL, vecRB = NULL, vecZM1B = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecpB, nB, d_pB, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecXB, nB, d_xB, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecYB, nB, d_yB, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecRB, nB, d_rB, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecZM1B, nB, d_zm1B, CUDA_R_32F));
    cusparseDnVecDescr_t vecomegaB = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecomegaB, nB, d_omegaB, CUDA_R_32F));

    /* Initialize problem data */
    checkCudaErrors(cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_colB, rowB, nnzB * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rowB, colB, (nB + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_valB, valB, nnzB * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_valB, valB, nnzB * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_xB, xB, nB * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rB, rhsB, nB * sizeof(float), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matA = NULL;
    cusparseSpMatDescr_t matM_lower, matM_upper;
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    checkCudaErrors(cusparseCreateCsr(
        &matA, N, N, nz, d_row, d_col, d_val, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpMatDescr_t matAB = NULL;
    cusparseSpMatDescr_t matM_lowerB, matM_upperB;
    cusparseFillMode_t fill_lowerB = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_unitB = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t fill_upperB = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unitB = CUSPARSE_DIAG_TYPE_NON_UNIT;

    checkCudaErrors(cusparseCreateCsr(
        &matAB, nB, nB, nnzB, d_rowB, d_colB, d_valB, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    /* Copy A data to ILU(0) vals as input*/
    checkCudaErrors(cudaMemcpy(
        d_valsILU0, d_val, nz * sizeof(float), cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(
        d_valsILU0B, d_valB, nnzB * sizeof(float), cudaMemcpyDeviceToDevice));

    // Lower Part
    checkCudaErrors(cusparseCreateCsr(&matM_lower, N, N, nz, d_row, d_col, d_valsILU0,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    checkCudaErrors(cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)));
    checkCudaErrors(cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_unit, sizeof(diag_unit)));

    checkCudaErrors(cusparseCreateCsr(&matM_lowerB, nB, nB, nnzB, d_rowB, d_colB, d_valsILU0B,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    checkCudaErrors(cusparseSpMatSetAttribute(matM_lowerB,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_lowerB, sizeof(fill_lowerB)));
    checkCudaErrors(cusparseSpMatSetAttribute(matM_lowerB,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_unitB, sizeof(diag_unitB)));

    // M_upper
    checkCudaErrors(cusparseCreateCsr(&matM_upper, N, N, nz, d_row, d_col, d_valsILU0,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    checkCudaErrors(cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_upper, sizeof(fill_upper)));
    checkCudaErrors(cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)));

    checkCudaErrors(cusparseCreateCsr(&matM_upperB, nB, nB, nnzB, d_rowB, d_colB, d_valsILU0B,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    checkCudaErrors(cusparseSpMatSetAttribute(matM_upperB,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_upperB, sizeof(fill_upperB)));
    checkCudaErrors(cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unitB,
                                              sizeof(diag_non_unitB)));

    /* Create ILU(0) info object */
    int bufferSizeLU = 0;
    size_t bufferSizeMV, bufferSizeL, bufferSizeU;
    void *d_bufferLU, *d_bufferMV, *d_bufferL, *d_bufferU;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    cusparseMatDescr_t matLU;
    csrilu02Info_t infoILU = NULL;

    int bufferSizeLUB = 0;
    size_t bufferSizeMVB, bufferSizeLB, bufferSizeUB;
    void *d_bufferLUB, *d_bufferMVB, *d_bufferLB, *d_bufferUB;
    cusparseSpSVDescr_t spsvDescrLB, spsvDescrUB;
    cusparseMatDescr_t matLUB;
    csrilu02Info_t infoILUB = NULL;

    checkCudaErrors(cusparseCreateCsrilu02Info(&infoILU));
    checkCudaErrors(cusparseCreateMatDescr(&matLU));
    checkCudaErrors(cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cusparseCreateCsrilu02Info(&infoILUB));
    checkCudaErrors(cusparseCreateMatDescr(&matLUB));
    checkCudaErrors(cusparseSetMatType(matLUB, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matLUB, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate workspace for cuSPARSE */
    checkCudaErrors(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
        vecp, &floatzero, vecomega, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSizeMV));
    checkCudaErrors(cudaMalloc(&d_bufferMV, bufferSizeMV));

    checkCudaErrors(cusparseSpMV_bufferSize(
        cusparseHandleB, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matAB,
        vecpB, &floatzero, vecomegaB, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSizeMVB));
    checkCudaErrors(cudaMalloc(&d_bufferMVB, bufferSizeMVB));

    checkCudaErrors(cusparseScsrilu02_bufferSize(
        cusparseHandle, N, nz, matLU, d_val, d_row, d_col, infoILU, &bufferSizeLU));
    checkCudaErrors(cudaMalloc(&d_bufferLU, bufferSizeLU));

    checkCudaErrors(cusparseScsrilu02_bufferSize(
        cusparseHandleB, nB, nnzB, matLUB, d_valB, d_rowB, d_colB, infoILUB, &bufferSizeLUB));
    checkCudaErrors(cudaMalloc(&d_bufferLUB, bufferSizeLUB));

    checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrL));
    checkCudaErrors(cusparseSpSV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_lower, vecR, vecX, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL));
    checkCudaErrors(cudaMalloc(&d_bufferL, bufferSizeL));

    checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrLB));
    checkCudaErrors(cusparseSpSV_bufferSize(
        cusparseHandleB, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_lowerB, vecRB, vecXB, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLB, &bufferSizeLB));
    checkCudaErrors(cudaMalloc(&d_bufferLB, bufferSizeLB));

    checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrU));
    checkCudaErrors(cusparseSpSV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper, vecR, vecX, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU));
    checkCudaErrors(cudaMalloc(&d_bufferU, bufferSizeU));

    checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrUB));
    checkCudaErrors(cusparseSpSV_bufferSize(
        cusparseHandleB, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upperB, vecRB, vecXB, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrUB, &bufferSizeUB));
    checkCudaErrors(cudaMalloc(&d_bufferUB, bufferSizeUB));

    // timing setup
    // Declare CUDA event variables
    cudaEvent_t start, stop, iterStart, iterStop, precondStart, precondStop;;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    cudaEventCreate(&precondStart);
    cudaEventCreate(&precondStop);

    // Banded Preconditioned Conjugate Gradient using ILU.
    // Follows the description by Golub & Van Loan,
    // "Matrix Computations 3rd ed.", Algorithm 10.3.1
    // Section 11.5.2 A Few Practical Details in 4th edition book
    printf("\n Banded Convergence of CG using ILU(0) preconditioning: \n");

    // Record the start event
    cudaEventRecord(start);

    // lets time the preconditioner work
    cudaEventRecord(precondStart);

    // Perform analysis for ILU(0)
    checkCudaErrors(cusparseScsrilu02_analysis(
        cusparseHandleB, nB, nnzB, descrB, d_valsILU0B, d_rowB, d_colB, infoILUB,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLUB));

    // generate the ILU(0) factors
    checkCudaErrors(cusparseScsrilu02(
        cusparseHandleB, nB, nnzB, matLUB, d_valsILU0B, d_rowB, d_colB, infoILUB,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLUB));

    // perform triangular solve analysis
    checkCudaErrors(cusparseSpSV_analysis(
        cusparseHandleB, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
        matM_lowerB, vecRB, vecXB, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLB, d_bufferLB));

    checkCudaErrors(cusparseSpSV_analysis(
        cusparseHandleB, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
        matM_upperB, vecRB, vecXB, CUDA_R_32F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrUB, d_bufferUB));

    // Record the stop event
    cudaEventRecord(precondStop);
    cudaEventSynchronize(precondStop);

    // Calculate the elapsed time
    float precond_time = 0;
    cudaEventElapsedTime(&precond_time, precondStart, precondStop);
    std::cout << "Preconditioning Time: " << precond_time << " ms" << std::endl;

    // reset the initial guess of the solution to zero
    for (int i = 0; i < nB; i++) {
        xB[i] = 0.0;
    }
    checkCudaErrors(cudaMemcpy(
        d_rB, rhsB, nB * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_xB, xB, nB * sizeof(float), cudaMemcpyHostToDevice));

    k = 0;
    checkCudaErrors(cublasSdot(cublasHandleB, nB, d_rB, 1, d_rB, 1, &r1));

    while (r1 > tol * tol && k <= max_iter) {
        // Record the start event for this iteration
        cudaEventRecord(iterStart);

        // preconditioner application: d_zm1 = U^-1 L^-1 d_r
        checkCudaErrors(cusparseSpSV_solve(cusparseHandleB,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
                                           matM_lowerB, vecRB, vecYB, CUDA_R_32F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrLB));

        checkCudaErrors(cusparseSpSV_solve(cusparseHandleB,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upperB,
                                           vecYB, vecZM1B, CUDA_R_32F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrUB));
        k++;

        if (k == 1) {
            checkCudaErrors(cublasScopy(cublasHandleB, nB, d_zm1B, 1, d_pB, 1));
        } else {
            checkCudaErrors(cublasSdot(
                cublasHandleB, nB, d_rB, 1, d_zm1B, 1, &numerator));
            checkCudaErrors(cublasSdot(
                cublasHandleB, nB, d_rm2B, 1, d_zm2B, 1, &denominator));
            beta = numerator / denominator;
            checkCudaErrors(cublasSscal(cublasHandleB, nB, &beta, d_pB, 1));
            checkCudaErrors(cublasSaxpy(
                cublasHandleB, nB, &floatone, d_zm1B, 1, d_pB, 1));
        }

        checkCudaErrors(cusparseSpMV(
            cusparseHandleB, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
            vecpB, &floatzero, vecomegaB, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_bufferMVB));
        checkCudaErrors(cublasSdot(
            cublasHandleB, nB, d_rB, 1, d_zm1B, 1, &numerator));
        checkCudaErrors(cublasSdot(
            cublasHandleB, nB, d_pB, 1, d_omegaB, 1, &denominator));
        alpha = numerator / denominator;
        checkCudaErrors(cublasSaxpy(cublasHandleB, nB, &alpha, d_pB, 1, d_xB, 1));
        checkCudaErrors(cublasScopy(cublasHandleB, nB, d_rB, 1, d_rm2B, 1));
        checkCudaErrors(cublasScopy(cublasHandleB, nB, d_zm1B, 1, d_zm2B, 1));
        nalpha = -alpha;
        checkCudaErrors(cublasSaxpy(
            cublasHandleB, nB, &nalpha, d_omegaB, 1, d_rB, 1));
        checkCudaErrors(cublasSdot(cublasHandleB, nB, d_rB, 1, d_rB, 1, &r1));

        // Record the stop event for this iteration
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);

        // Calculate the elapsed time for this iteration
        float iterMilliseconds = 0;
        cudaEventElapsedTime(&iterMilliseconds, iterStart, iterStop);

        // Print the residual and time for this iteration
        std::cout << "Current iteration: " << k << ", Residual: " << sqrt(r1) << ", Time: " << iterMilliseconds << " ms" << std::endl;
        std::cout << std::endl;

        // Write the iteration, residual, and time to the CSV file
        outFile << k << "," << sqrt(r1) << "," << iterMilliseconds << "\n";
    }

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));
    printf("  Elapsed seconds %e \n", milliseconds);

    checkCudaErrors(cudaMemcpy(
        xB, d_xB, nB * sizeof(float), cudaMemcpyDeviceToHost));

    // check result
    err = 0.0;

    for (int i = 0; i < nB; i++) {
        rsum = 0.0;

        for (int j = colB[i]; j < colB[i + 1]; j++) {
            rsum += valB[j] * xB[J[j]];
        }

        diff = fabs(rsum - rhsB[i]);

        if (diff > err) {
            err = diff;
        }
    }

    // Prepare the CSV file name and open it for appending
    std::ofstream outFile_sum("results_summary.csv", std::ios::app);

    // Check if the file is newly created (empty), write headers if so
    if (outFile_sum.tellp() == 0) {
        outFile_sum << "Matrix Name,Sparsification Ratio,Rows,Cols,Nonzeros,Final Residual,Iterations Spent,Overall Time (ms),Preconditioning Time (ms)" << std::endl;
    }

    // Append the results for this matrix to the CSV file
    outFile_sum << matrix_name << "," << sparsification_ratio << "," << M << "," << N << "," << nz << ","
            << sqrt(r1) << "," << k << "," << milliseconds << "," << precond_time << std::endl;

    // Close the file
    outFile.close();

    std::cout << "Results for matrix " << matrix_name << " have been appended to results_summary.csv" << std::endl;

    printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr2B = err;

    //* Destroy descriptors
    checkCudaErrors(cusparseDestroyCsrilu02Info(infoILU));
    checkCudaErrors(cusparseDestroyMatDescr(matLU));
    checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrL));
    checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrU));
    checkCudaErrors(cusparseDestroySpMat(matM_lower));
    checkCudaErrors(cusparseDestroySpMat(matM_upper));
    checkCudaErrors(cusparseDestroySpMat(matA));
    checkCudaErrors(cusparseDestroyDnVec(vecp));
    checkCudaErrors(cusparseDestroyDnVec(vecomega));
    checkCudaErrors(cusparseDestroyDnVec(vecR));
    checkCudaErrors(cusparseDestroyDnVec(vecX));
    checkCudaErrors(cusparseDestroyDnVec(vecY));
    checkCudaErrors(cusparseDestroyDnVec(vecZM1));

    checkCudaErrors(cusparseDestroyCsrilu02Info(infoILUB));
    checkCudaErrors(cusparseDestroyMatDescr(matLUB));
    checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrLB));
    checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrUB));
    checkCudaErrors(cusparseDestroySpMat(matM_lowerB));
    checkCudaErrors(cusparseDestroySpMat(matM_upperB));
    checkCudaErrors(cusparseDestroySpMat(matAB));
    checkCudaErrors(cusparseDestroyDnVec(vecpB));
    checkCudaErrors(cusparseDestroyDnVec(vecomegaB));
    checkCudaErrors(cusparseDestroyDnVec(vecRB));
    checkCudaErrors(cusparseDestroyDnVec(vecXB));
    checkCudaErrors(cusparseDestroyDnVec(vecYB));
    checkCudaErrors(cusparseDestroyDnVec(vecZM1B));

    //* Destroy contexts
    checkCudaErrors(cusparseDestroy(cusparseHandle));
    checkCudaErrors(cublasDestroy(cublasHandle));
    checkCudaErrors(cusparseDestroy(cusparseHandleB));
    checkCudaErrors(cublasDestroy(cublasHandleB));

    // Free device memory
    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    checkCudaErrors(cudaFree(d_bufferMV));
    checkCudaErrors(cudaFree(d_bufferLU));
    checkCudaErrors(cudaFree(d_bufferL));
    checkCudaErrors(cudaFree(d_bufferU));
    checkCudaErrors(cudaFree(d_col));
    checkCudaErrors(cudaFree(d_row));
    checkCudaErrors(cudaFree(d_val));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_r));
    checkCudaErrors(cudaFree(d_p));
    checkCudaErrors(cudaFree(d_omega));
    checkCudaErrors(cudaFree(d_valsILU0));
    checkCudaErrors(cudaFree(d_zm1));
    checkCudaErrors(cudaFree(d_zm2));
    checkCudaErrors(cudaFree(d_rm2));
    free(colB);
    free(rowB);
    free(valB);
    free(xB);
    free(rhsB);
    checkCudaErrors(cudaFree(d_bufferMVB));
    checkCudaErrors(cudaFree(d_bufferLUB));
    checkCudaErrors(cudaFree(d_bufferLB));
    checkCudaErrors(cudaFree(d_bufferUB));
    checkCudaErrors(cudaFree(d_colB));
    checkCudaErrors(cudaFree(d_rowB));
    checkCudaErrors(cudaFree(d_valB));
    checkCudaErrors(cudaFree(d_xB));
    checkCudaErrors(cudaFree(d_yB));
    checkCudaErrors(cudaFree(d_rB));
    checkCudaErrors(cudaFree(d_pB));
    checkCudaErrors(cudaFree(d_omegaB));
    checkCudaErrors(cudaFree(d_valsILU0B));
    checkCudaErrors(cudaFree(d_zm1B));
    checkCudaErrors(cudaFree(d_zm2B));
    checkCudaErrors(cudaFree(d_rm2B));

    // Destroy CUDA events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(iterStart));
    checkCudaErrors(cudaEventDestroy(iterStop));
    checkCudaErrors(cudaEventDestroy(precondStart));
    checkCudaErrors(cudaEventDestroy(precondStop));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    printf("\n");
    printf("Test Summary:\n");
    printf("   Counted total of %d errors\n", nErrors);
    printf("   qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
    printf("   qaerr1B = %f qaerr2B = %f\n\n", fabs(qaerr1B), fabs(qaerr2B));
    exit((nErrors == 0 && fabs(qaerr1) < 1e-5 && fabs(qaerr2) < 1e-5
          ? EXIT_SUCCESS
          : EXIT_FAILURE));
    exit((nErrors == 0 && fabs(qaerr1B) < 1e-5 && fabs(qaerr2B) < 1e-5
          ? EXIT_SUCCESS
          : EXIT_FAILURE));
}
