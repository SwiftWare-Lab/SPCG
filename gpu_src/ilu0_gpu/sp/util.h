//
// Created by kazem on 4/5/18.
//

#include <iostream>
#include <sstream>
#include <fstream>

void make_full(int nB, size_t &nnzB, int* &colB, int* &rowB, float* &valB) {
    // Step 1: Calculate the number of non-zeros in the full matrix
    int* ind = new int[nB]();  // Temporary array to count new non-zeros

    for (size_t i = 0; i < nB; i++) {
        for (size_t p = colB[i]; p < colB[i + 1]; p++) {
            int row = rowB[p];
            ind[i]++;
            if (row != i) {
                ind[row]++;
            }
        }
    }

    // Calculate new nnz for the full matrix
    size_t new_nnzB = nnzB;
    for (size_t i = 0; i < nB; i++) {
        new_nnzB += ind[i];
    }

    // Allocate new arrays for the full matrix
    int* new_colB = new int[nB + 1];
    int* new_rowB = new int[new_nnzB];
    float* new_valB = new float[new_nnzB];

    // Step 2: Fill in the new_colB with the correct pointers
    new_colB[0] = 0;
    for (size_t i = 0; i < nB; i++) {
        new_colB[i + 1] = new_colB[i] + ind[i];
    }

    // Reset ind array to use as index tracker
    for (size_t i = 0; i < nB; i++) {
        ind[i] = 0;
    }

    // Step 3: Populate the new_rowB and new_valB arrays with values
    for (size_t i = 0; i < nB; i++) {
        for (size_t p = colB[i]; p < colB[i + 1]; p++) {
            int row = rowB[p];
            int index = new_colB[i] + ind[i];
            new_rowB[index] = row;
            new_valB[index] = valB[p];
            ind[i]++;
            if (row != i) {
                index = new_colB[row] + ind[row];
                new_rowB[index] = i;
                new_valB[index] = valB[p];
                ind[row]++;
            }
        }
    }

    // Step 4: Update the original pointers to the new arrays
    delete[] colB;
    delete[] rowB;
    delete[] valB;

    colB = new_colB;
    rowB = new_rowB;
    valB = new_valB;
    nnzB = new_nnzB;

    delete[] ind;
}


/*
 * Read a full matrix especially when the input is symmetric
 */
bool readFullMatrix(std::string fName, size_t &n, size_t &NNZ, int* &col, int* &row, float* &val) {
    std::ifstream inFile(fName);
    std::string line, banner, mtx, crd, arith, sym;

    std::getline(inFile, line);
    for (unsigned i = 0; i < line.length(); line[i] = tolower(line[i]), i++);
    std::istringstream iss(line);
    if (!(iss >> banner >> mtx >> crd >> arith >> sym)) {
        std::cout << "Invalid header (first line does not contain 5 tokens)\n";
        return false;
    }

    if (banner.compare("%%matrixmarket") || mtx.compare("matrix") || crd.compare("coordinate")) {
        std::cout << "Invalid matrix format; this driver cannot handle that.\n";
        return false;
    }
    bool isSymmetric = (sym == "symmetric");

    if (arith.compare("real")) {
        if (!arith.compare("complex")) {
            std::cout << "Complex matrix; use zreadMM instead!\n";
        } else if (!arith.compare("pattern")) {
            std::cout << "Pattern matrix; values are needed!\n";
        } else {
            std::cout << "Unknown arithmetic\n";
        }
        return false;
    }

    while (!line.compare(0, 1, "%")) {
        std::getline(inFile, line);
    }

    std::istringstream issDim(line);
    if (!(issDim >> n >> n >> NNZ)) {
        std::cout << "The matrix dimension is missing\n";
        return false;
    }

    if (n <= 0 || NNZ <= 0)
        return false;

    int estimatedNNZ = isSymmetric ? 2 * NNZ - n : NNZ;
    col = (int *)malloc(sizeof(int) * (n + 1));
    row = (int *)malloc(sizeof(int) * estimatedNNZ);
    val = (float *)malloc(sizeof(float) * estimatedNNZ);

    if (!val || !col || !row)
        return false;

    int y, x, colCnt = 0, nnzCnt = 0;
    float value;

    col[0] = 0;
    for (int i = 0; nnzCnt < NNZ; ) {
        inFile >> x;
        x--;
        inFile >> y;
        y--; // zero indexing
        inFile >> value;
        if (y > n)
            return false;
        if (y == i) {
            val[nnzCnt] = value;
            row[nnzCnt] = x;
            colCnt++;
            nnzCnt++;
            if (isSymmetric && x != y) {
                val[nnzCnt] = value;
                row[nnzCnt] = y;
                colCnt++;
                nnzCnt++;
            }
        } else { // New column
            col[i + 1] = col[i] + colCnt;
            i++;
            colCnt = 1;
            val[nnzCnt] = value;
            row[nnzCnt] = x;
            nnzCnt++;
            if (isSymmetric && x != y) {
                val[nnzCnt] = value;
                row[nnzCnt] = y;
                colCnt++;
                nnzCnt++;
            }
        }
    }
    col[n] = col[n - 1] + colCnt; // last column
    return true;
}


/*
 * Reading a CSC matrix from a coordinate file, stored column-ordered.
 */
bool readMatrix(std::string fName, size_t &n, size_t &NNZ, int* &col, int* &row, float* &val) {
    std::ifstream inFile(fName);
    std::string line, banner, mtx, crd, arith, sym;

    std::getline(inFile, line);
    for (unsigned i = 0; i < line.length(); line[i] = tolower(line[i]), i++);
    std::istringstream iss(line);
    if (!(iss >> banner >> mtx >> crd >> arith >> sym)) {
        std::cout << "Invalid header (first line does not contain 5 tokens)\n";
        return false;
    }

    if (banner.compare("%%matrixmarket") || mtx.compare("matrix") || crd.compare("coordinate")) {
        std::cout << "Invalid matrix format; this driver cannot handle that.\n";
        return false;
    }
    if (arith.compare("real")) {
        if (!arith.compare("complex")) {
            std::cout << "Complex matrix; use zreadMM instead!\n";
        } else if (!arith.compare("pattern")) {
            std::cout << "Pattern matrix; values are needed!\n";
        } else {
            std::cout << "Unknown arithmetic\n";
        }
        return false;
    }

    while (!line.compare(0, 1, "%")) {
        std::getline(inFile, line);
    }

    std::istringstream issDim(line);
    if (!(issDim >> n >> n >> NNZ)) {
        std::cout << "The matrix dimension is missing\n";
        return false;
    }

    if (n <= 0 || NNZ <= 0)
        return false;

    col = (int *)malloc(sizeof(int) * (n + 1));
    row = (int *)malloc(sizeof(int) * NNZ);
    val = (float *)malloc(sizeof(float) * NNZ);

    if (!val || !col || !row)
        return false;

    int y, x, colCnt = 0, nnzCnt = 0;
    float value;

    col[0] = 0;
    for (int i = 0; nnzCnt < NNZ; ) {
        inFile >> x;
        x--;
        inFile >> y;
        y--; // zero indexing
        inFile >> value;
        if (y > n)
            return false;
        if (y == i) {
            val[nnzCnt] = value;
            row[nnzCnt] = x;
            colCnt++;
            nnzCnt++;
        } else { // New column
            col[i + 1] = col[i] + colCnt;
            i++;
            colCnt = 1;
            val[nnzCnt] = value;
            row[nnzCnt] = x;
            nnzCnt++;
        }
    }
    col[n] = col[n - 1] + colCnt; // last column
    return true;
}

/*
 * Reading a COO matrix from a coordinate file, stored column-ordered.
 */
bool readMatrixCOO(std::string fName, size_t &n, size_t &NNZ, int* &col, int* &row, float* &val) {
    std::ifstream inFile(fName);
    std::string line, banner, mtx, crd, arith, sym;

    std::getline(inFile, line);
    for (unsigned i = 0; i < line.length(); line[i] = tolower(line[i]), i++);
    std::istringstream iss(line);
    if (!(iss >> banner >> mtx >> crd >> arith >> sym)) {
        std::cout << "Invalid header (first line does not contain 5 tokens)\n";
        return false;
    }

    if (banner.compare("%%matrixmarket") || mtx.compare("matrix") || crd.compare("coordinate")) {
        std::cout << "Invalid matrix format; this driver cannot handle that.\n";
        return false;
    }
    if (arith.compare("real")) {
        if (!arith.compare("complex")) {
            std::cout << "Complex matrix; use zreadMM instead!\n";
        } else if (!arith.compare("pattern")) {
            std::cout << "Pattern matrix; values are needed!\n";
        } else {
            std::cout << "Unknown arithmetic\n";
        }
        return false;
    }

    while (!line.compare(0, 1, "%")) {
        std::getline(inFile, line);
    }

    std::istringstream issDim(line);
    if (!(issDim >> n >> n >> NNZ)) {
        std::cout << "The matrix dimension is missing\n";
        return false;
    }

    if (n <= 0 || NNZ <= 0)
        return false;

    col = new int[NNZ];
    row = new int[NNZ];
    val = new float[NNZ];
    if (!val || !col || !row)
        return false;

    int y, x;
    float value;

    for (int nnzCnt = 0; nnzCnt < NNZ; nnzCnt++) {
        inFile >> x;
        x--;
        inFile >> y;
        y--; // zero indexing
        inFile >> value;
        if (y > n)
            return false;

        val[nnzCnt] = value;
        row[nnzCnt] = x;
        col[nnzCnt] = y;
    }

    return true;
}
