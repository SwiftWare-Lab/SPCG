# SPCG: Sparsified Preconditioned Conjugate Gradient Solver on GPUs

This repository contains GPU and CPU implementations of sparse matrix iterative solvers with ILU0 and ILUK preconditioning.

## Build Targets

This project provides three main executable targets:

1. **ilu0_gpu** - GPU-accelerated ILU0 preconditioned conjugate gradient solver
2. **iluk_gpu** - GPU-accelerated ILUK preconditioned conjugate gradient solver  
3. **ilu0_cpu** - CPU-based ILU0 preconditioned conjugate gradient solver

## Prerequisites

### System Dependencies

**For GPU targets:**
- CUDA Toolkit (version 11.0 or later)
- CMake 3.20 or later
- C++17 compatible compiler
- Git

**For CPU target:**
- OpenMP
- Eigen3 library
- CMake 3.0 or later
- C++17 compatible compiler

**For matrix preparation:**
- Python 3.x
- Required Python packages:
  ```bash
  pip install numpy scipy pandas matplotlib ssgetpy
  ```

### Installing Dependencies

**On Ubuntu/Debian:**
```bash
# Install basic development tools
sudo apt update
sudo apt install cmake build-essential git

# Install Eigen3 and OpenMP
sudo apt install libeigen3-dev libomp-dev

# Install CUDA Toolkit (follow NVIDIA's official guide)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu20.04/x86_64/cuda-ubuntu20.04.pin
sudo mv cuda-ubuntu20.04.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu20.04-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu20.04-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu20.04-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda
```

**On CentOS/RHEL:**
```bash
# Install development tools
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 eigen3-devel

# Install CUDA (follow NVIDIA's official guide)
# Download and install CUDA toolkit from NVIDIA website
```

## Data Preparation

Before building and running the targets, you need to prepare the matrix data:

### 1. Download Test Matrices

Create a directory structure and download matrices from the SuiteSparse Matrix Collection:

```bash
# Create matrices directory
mkdir -p matrices

# Download matrices using the provided script
cd script_src/python_scripts/prep
python3 matrix_download.py
cd ../../..
```

This will download approximately 100+ test matrices from the SuiteSparse collection into the `matrices/` directory.

### 2. Generate Sparsified Matrices

Generate sparsified versions of the matrices for testing different sparsification ratios:

```bash
# Generate sparsified matrices (may take several hours)
cd script_src/matlab_scripts
matlab -batch "matrix_sparsification"
cd ../..
```

### 3. Generate Matrix Properties

Compute various matrix properties needed for the algorithms:

```bash
# Compute matrix properties
cd script_src/matlab_scripts
matlab -batch "matrix_sparsification_analysis"
cd ../..
```

### 4. Generate ILUK Factorization Data (Required for ILUK target)

For the ILUK GPU target, you need to pre-generate factorization data:

```bash
# Create factors directory
mkdir -p factors/timing

# Generate ILUK factorization data
cd script_src/python_scripts/prep/iluk_factorization
python3 iluk_factorize.py ../../../matrices/1138_bus/1138_bus.mtx --export
cd ../../../..
```

Repeat this process for all matrices you want to test with ILUK.

## Building the Targets

### 1. Build ILU0 GPU Target

The ILU0 GPU implementation has two variants: non-sparsified (nonsp) and sparsified (sp).

**Build non-sparsified version:**
```bash
cd gpu_src/ilu0_gpu/nonsp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ../../../..
```

**Build sparsified version:**
```bash
cd gpu_src/ilu0_gpu/sp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ../../../..
```

### 2. Build ILUK GPU Target

```bash
cd gpu_src/iluk_gpu
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ../../..
```

### 3. Build ILU0 CPU Target

```bash
cd cpu_src
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ../..
```

## Running the Targets

### ILU0 GPU (Non-sparsified)

```bash
cd gpu_src/ilu0_gpu/nonsp/build
./conjugateGradientPrecond ../../../../matrices/1138_bus/1138_bus.mtx
```

### ILU0 GPU (Sparsified)

```bash
cd gpu_src/ilu0_gpu/sp/build
./conjugateGradientPrecond ../../../../matrices/1138_bus/1138_bus.mtx ../../../../matrices/1138_bus/1138_bus.mtx <sparsification ratio>
```

The <sparsification ratio> can be 0.01, 0.05, 0.1 etc., as long as it aligns with what the matrix_sparsification.m script produces.

### ILUK GPU

```bash
cd gpu_src/iluk_gpu/build
./conjugateGradientPrecond ../../../../matrices/1138_bus/1138_bus.mtx <path to lower factor> <path to upper factor>
```

### ILU0 CPU

```bash
cd cpu_src/build
./conjugateGradientPrecond ../../matrices/1138_bus/1138_bus.mtx ../../matrices/1138_bus/1138_bus_sp_0.05.mtx
```

## Performance Optimization

### CPU Optimization

- Set appropriate OpenMP thread count:
  ```bash
  export OMP_NUM_THREADS=8
  ```
- For optimal performance, pin threads to specific cores:
  ```bash
  export OMP_PLACES=cores
  export OMP_PROC_BIND=close
  ```

### Common Runtime Issues

1. **Matrix file not found:**
   - Verify matrix paths are correct relative to executable location
   - Ensure matrices were downloaded properly

2. **GPU memory errors:**
   - Check available GPU memory: `nvidia-smi`
   - Try smaller matrices first
   - Reduce batch size

3. **Performance issues:**
   - Verify OpenMP is properly linked for CPU version
   - Check CUDA installation for GPU versions
   - Monitor system resources during execution
