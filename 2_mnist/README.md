# MNIST Neural Network Examples

Progressive implementations of a two-layer MLP for MNIST digit classification,
from **Chapter 04** of the book. The same network (input 784 -> hidden 256 ->
output 10) is rebuilt in Python, C, and CUDA so you can compare the host and
device code directly.

## Files

| File | Description |
|------|-------------|
| `v0.py` | Data preprocessing: downloads MNIST and writes `data/*.bin` |
| `v1.py` | NumPy reference implementation |
| `v2.py` | PyTorch implementation |
| `v3.c` | Single-threaded C implementation |
| `v4.cu` | Custom CUDA kernels |
| `v5.cu` | CUDA with cuBLAS for the matrix multiplications |

## Prerequisites

- NVIDIA GPU with CUDA support (for `v4`/`v5`)
- CUDA Toolkit and a C/C++ compiler
- Python with `torch` and `torchvision` (for the data prep and Python versions)

## Step 1: Prepare the data

The C and CUDA versions read the dataset from `data/*.bin`. Generate those files
once with the preprocessing script:

```bash
python v0.py
```

This downloads MNIST via torchvision and writes `X_train.bin`, `y_train.bin`,
`X_test.bin`, and `y_test.bin` into `data/`.

## Step 2: Build and run

```bash
# C version
gcc -O2 v3.c -o mnist_c -lm
./mnist_c

# CUDA version (custom kernels)
nvcc -O3 v4.cu -o mnist_cuda
./mnist_cuda

# CUDA version with cuBLAS
nvcc -O3 v5.cu -o mnist_cublas -lcublas
./mnist_cublas
```

The Python versions run directly:

```bash
python v1.py   # NumPy
python v2.py   # PyTorch
```

## Hyperparameters

Defined at the top of `v3.c`/`v4.cu`/`v5.cu` and matching the book:

```c
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
```
