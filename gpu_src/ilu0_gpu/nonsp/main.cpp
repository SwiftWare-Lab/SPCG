// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // shared functions common to CUDA Samples
#include <helper_cuda.h>       // CUDA error checking

// Da's added included
#include <iostream>
#include <fstream>

#include "helper_cusolver.h"
#include "mmio_wrapper.h"

int main(int argc, char** argv) {
    // Check if the path to the .mtx file is provided as an argument
    if (argc < 2) {
        fprintf(stderr, "Error: Please provide the path to the .mtx file as an argument.\n");
        return 1;
    }

    int max_iter = 1000;
    int M = 0, N = 0, nz = 0;
    const float tol = 1e-12f;
    float qaerr1 = 0.0, qaerr2 = 0.0, err = 0.0;
    float rsum, diff;
    int nErrors = 0;
    const float floatone = 1.0;
    const float floatzero = 0.0;

    printf("conjugateGradientPrecond starting (ilu0)...\n");

    /* This will pick the best possible CUDA capable device */
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char**)argv);
    printf("GPU selected Device ID = %d \n", devID);

    if (devID < 0) {
        printf("Invalid GPU device %d selected,  exiting...\n", devID);
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    /* Statistics about the GPU device */
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    /* Load matrix from .mtx files */
    int *row_indices, *col_indices;
    float *values;

    size_t bufferSizeMV, bufferSizeL, bufferSizeU;
    void* d_bufferMV, * d_bufferL, * d_bufferU;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;

    struct testOpts opts_matA{};

    // For sparse matrix A
    const char *filename_matA = argv[1];
    opts_matA.sparse_mat_filename = (char *)malloc(strlen(filename_matA) + 1);
    strcpy(opts_matA.sparse_mat_filename, filename_matA);
    printf("Using sparse matrix [%s]\n", opts_matA.sparse_mat_filename);

    loadMMSparseMatrix<float>(opts_matA.sparse_mat_filename, 'd', true, &M,
                               &N, &nz, &values, &row_indices,
                               &col_indices, true);

    int baseA = 0;
    baseA = row_indices[0];  // baseA = {0,1}
    if (M != N) {
        fprintf(stderr, "Error: only support square matrix\n");
        return 1;
    }

    printf("WARNING: only works for base-0 \n");
    if (baseA) {
        for (int i = 0; i <= M; i++) {
            row_indices[i]--;
        }
        for (int i = 0; i < nz; i++) {
            col_indices[i]--;
        }
        baseA = 0;
    }

    std::cout << "Sparse matrix A is " << M << " x " << N << " with " << nz << " nonzeros" << std::endl;

    // Get the absolute path to the matrix file
    std::string matrix_path = argv[1];

    // Find the last '/' in the path
    size_t last_slash = matrix_path.find_last_of("/\\");

    // Extract the matrix name (without extension)
    std::string matrix_name = matrix_path.substr(last_slash + 1, matrix_path.find_last_of('.') - last_slash - 1);

    // Construct the output CSV file name
    std::string csv_filename = "float_residuals_" + matrix_name + ".csv";

    // Open CSV file for writing
    std::ofstream outFile(csv_filename);
    // Write the header
    outFile << "Iteration,Residual,Time\n";

    /* Initialize rhs and x vectors */
    float *x, *rhs;
    x = (float *)malloc(sizeof(float) * N);
    rhs = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;  // Initialize RHS
        x[i] = 0.0;    // Initial solution approximation
    }

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    checkCudaErrors(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    /* Description of the A matrix */
    cusparseMatDescr_t descr = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    checkCudaErrors(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate required memory */
    int* d_col, * d_row;
    float* d_val, * d_x, * d_y, * d_r, * d_p, * d_omega, * d_zm1, * d_zm2, * d_rm2, *d_valsILU0;
    checkCudaErrors(cudaMalloc((void**)&d_col, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_row, (N + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_val, nz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_x, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_y, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_r, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_p, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_omega, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_zm1, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_zm2, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_rm2, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nz * sizeof(float)));

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseDnVecDescr_t vecp = NULL, vecX = NULL, vecY = NULL, vecR = NULL, vecZM1 = NULL, vecomega = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecR, N, d_r, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecZM1, N, d_zm1, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnVec(&vecomega, N, d_omega, CUDA_R_32F));

    /* Initialize problem data */
    checkCudaErrors(cudaMemcpy(d_col, col_indices, nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row, row_indices, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, values, nz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matA = NULL, matM_lower, matM_upper;
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    checkCudaErrors(cusparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    /* Copy A data to ILU(0) vals as input*/
    checkCudaErrors(cudaMemcpy(
            d_valsILU0, d_val, nz*sizeof(float), cudaMemcpyDeviceToDevice));

    //Lower Part
    checkCudaErrors( cusparseCreateCsr(&matM_lower, N, N, nz, d_row, d_col, d_valsILU0,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    checkCudaErrors( cusparseSpMatSetAttribute(matM_lower,
                                               CUSPARSE_SPMAT_FILL_MODE,
                                               &fill_lower, sizeof(fill_lower)) );
    checkCudaErrors( cusparseSpMatSetAttribute(matM_lower,
                                               CUSPARSE_SPMAT_DIAG_TYPE,
                                               &diag_unit, sizeof(diag_unit)) );
    // M_upper
    checkCudaErrors( cusparseCreateCsr(&matM_upper, N, N, nz, d_row, d_col, d_valsILU0,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    checkCudaErrors( cusparseSpMatSetAttribute(matM_upper,
                                               CUSPARSE_SPMAT_FILL_MODE,
                                               &fill_upper, sizeof(fill_upper)) );
    checkCudaErrors( cusparseSpMatSetAttribute(matM_upper,
                                               CUSPARSE_SPMAT_DIAG_TYPE,
                                               &diag_non_unit,
                                               sizeof(diag_non_unit)) );

    /* Create ILU(0) info object */
    int bufferSizeLU = 0;
    void *d_bufferLU;
    cusparseMatDescr_t matLU;
    csrilu02Info_t infoILU = NULL;

    checkCudaErrors(cusparseCreateCsrilu02Info(&infoILU));
    checkCudaErrors(cusparseCreateMatDescr(&matLU));
    checkCudaErrors(cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate workspace for cuSPARSE */
    checkCudaErrors(cusparseSpMV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
            vecp, &floatzero, vecomega, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            &bufferSizeMV));
    checkCudaErrors(cudaMalloc(&d_bufferMV, bufferSizeMV));

    checkCudaErrors(cusparseScsrilu02_bufferSize(
            cusparseHandle, N, nz, matLU, d_val, d_row, d_col, infoILU, &bufferSizeLU));
    checkCudaErrors(cudaMalloc(&d_bufferLU, bufferSizeLU));

    checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrL));
    checkCudaErrors(cusparseSpSV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_lower, vecR, vecX, CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL));
    checkCudaErrors(cudaMalloc(&d_bufferL, bufferSizeL));

    checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrU));
    checkCudaErrors(cusparseSpSV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper, vecR, vecX, CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU));
    checkCudaErrors(cudaMalloc(&d_bufferU, bufferSizeU));

    /* Main CG computation starts here */
    // Declare CUDA event variables
    cudaEvent_t start, stop, iterStart, iterStop, precondStart, precondStop;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    cudaEventCreate(&precondStart);
    cudaEventCreate(&precondStop);

    int k = 0;
    float r1, alpha, beta, numerator, denominator, nalpha;

    checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

    // Record the start event
    cudaEventRecord(start);
    // lets time the preconditioner work
    cudaEventRecord(precondStart);

    /* Perform analysis for ILU(0) */
    checkCudaErrors(cusparseScsrilu02_analysis(
            cusparseHandle, N, nz, descr, d_valsILU0, d_row, d_col, infoILU,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

    /* generate the ILU(0) factors */
    checkCudaErrors(cusparseScsrilu02(
            cusparseHandle, N, nz, matLU, d_valsILU0, d_row, d_col, infoILU,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

    checkCudaErrors(cusparseSpSV_analysis(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
            matM_lower, vecR, vecY, CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL));
    checkCudaErrors(cusparseSpSV_analysis(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
            matM_upper, vecY, vecZM1, CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_bufferU));

    // Record the stop event
    cudaEventRecord(precondStop);
    cudaEventSynchronize(precondStop);

    // Calculate the elapsed time
    float precond_time = 0;
    cudaEventElapsedTime(&precond_time, precondStart, precondStop);
    std::cout << "Preconditioning Time: " << precond_time << " ms" << std::endl;

    std::cout << "Entering CG loop with initial residual (r1): " << sqrt(r1) << std::endl;
    std::cout << std::endl;

    while (r1 > tol * tol && k < max_iter)
    {
        // Record the start event for this iteration
        cudaEventRecord(iterStart);

        // preconditioner application: d_zm1 = U^-1 L^-1 d_r
        checkCudaErrors(cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
                                           matM_lower, vecR, vecY, CUDA_R_32F,
                                           CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL));

        checkCudaErrors(cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
                                           matM_upper, vecY, vecZM1, CUDA_R_32F,
                                           CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU));

        k++;

        if (k == 1)
        {
            /* First spin:
             * p = np.empty_like(r)
             * p[:] = z[:]
             */
            checkCudaErrors(cublasScopy(cublasHandle, N, d_zm1, 1, d_p, 1));
        }
        else
        {
            // rho_cur = dotprod(r, z)
            checkCudaErrors(cublasSdot(
                    cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
            // rho_prev
            checkCudaErrors(cublasSdot(
                    cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator));

            // beta = rho_cur / rho_prev
            beta = numerator / denominator;

            // p *= beta
            checkCudaErrors(cublasSscal(cublasHandle, N, &beta, d_p, 1));
            // p += z
            checkCudaErrors(cublasSaxpy(
                    cublasHandle, N, &floatone, d_zm1, 1, d_p, 1));
        }

        // q = matvec(p), where matvec = A.matvec; So omega is equivalent to q
        checkCudaErrors(cusparseSpMV(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
                vecp, &floatzero, vecomega, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                d_bufferMV));
        checkCudaErrors(cudaDeviceSynchronize()); // Check for errors after kernel launch

        // dotprod(r, z) as computed above for the numerator of beta
        checkCudaErrors(cublasSdot(
                cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
        // dotprod(p, q), where p is equivalent p, and q is equivalent omega
        checkCudaErrors(cublasSdot(
                cublasHandle, N, d_p, 1, d_omega, 1, &denominator));

        alpha = numerator / denominator;

        // Update d_x: x += alpha*p
        checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

        // Record current d_r for usage in the next iteration
        checkCudaErrors(cublasScopy(cublasHandle, N, d_r, 1, d_rm2, 1));
        // Record current d_zm1 for usage in the next iteration
        checkCudaErrors(cublasScopy(cublasHandle, N, d_zm1, 1, d_zm2, 1));

        // Update d_r: r -= alpha*q
        nalpha = -alpha;
        checkCudaErrors(cublasSaxpy(
                cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
        checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

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

    std::cout << std::endl;
    printf("CG computation summary: \n");
    printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));
    printf("  Total time for CG operation: %f ms\n", milliseconds);

    // Prepare the CSV file name and open it for appending
    std::ofstream outFile_sum("results_summary_float.csv", std::ios::app);

    // Check if the file is newly created (empty), write headers if so
    if (outFile_sum.tellp() == 0) {
        outFile_sum << "Matrix Name,Rows,Cols,Nonzeros,Final Residual,Iterations Spent,Overall Time (ms),Preconditioning Time (ms),PCG Time (ms)" << std::endl;
    }

    // Append the results for this matrix to the CSV file
    outFile_sum << matrix_name << "," << M << "," << N << "," << nz << ","
                << sqrt(r1) << "," << k << "," << milliseconds << "," << precond_time << "," << (milliseconds - precond_time) << std::endl;

    // Close the file
    outFile.close();

    std::cout << "Results for matrix " << matrix_name << " have been appended to results_summary_float.csv" << std::endl;

    checkCudaErrors(cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Check result */
    err = 0.0;

    for (int i = 0; i < N; i++) {
        rsum = 0.0;

        for (int j = row_indices[i]; j < row_indices[i + 1]; j++) {
            rsum += values[j] * x[col_indices[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err) {
            err = diff;
        }
    }

    // TODO: Change the measurement of convergence
    printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr2 = err;

    // Close the CSV file
    outFile.close();

    // Destroy CUDA events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    /* Destroy descriptors */
    checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrL));
    checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrU));
    checkCudaErrors(cusparseDestroySpMat(matM_lower));
    checkCudaErrors(cusparseDestroySpMat(matM_upper));
    checkCudaErrors(cusparseDestroySpMat(matA));
    checkCudaErrors(cusparseDestroyDnVec(vecp));
    checkCudaErrors(cusparseDestroyDnVec(vecomega));
    checkCudaErrors(cusparseDestroyDnVec(vecX));
    checkCudaErrors(cusparseDestroyDnVec(vecY));
    checkCudaErrors(cusparseDestroyDnVec(vecZM1));

    /* Destroy contexts */
    checkCudaErrors(cusparseDestroy(cusparseHandle));
    checkCudaErrors(cublasDestroy(cublasHandle));

    /* Free device memory */
    checkCudaErrors(cudaFree(d_bufferMV));
    checkCudaErrors(cudaFree(d_bufferL));
    checkCudaErrors(cudaFree(d_bufferU));
    checkCudaErrors(cudaFree(d_col));
    checkCudaErrors(cudaFree(d_row));
    checkCudaErrors(cudaFree(d_val));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_p));
    checkCudaErrors(cudaFree(d_omega));
    checkCudaErrors(cudaFree(d_zm1));
    checkCudaErrors(cudaFree(d_zm2));
    checkCudaErrors(cudaFree(d_rm2));

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
    exit((nErrors == 0 && fabs(qaerr1) < 1e-5 && fabs(qaerr2) < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE));
}
