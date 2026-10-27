# ASSIGNMENT 02 - Optimised Matrix Multiplication Approaches and Sparse Matrices

## Overview

This assignment studies how different implementation approaches affect matrix multiplication performance.

**Part A** compares three dense matrix multiplication implementations:
- `baseline` – the reference implementation.
- `reordered` – a version with reordered loops.
- `blocked` – a version using blocked multiplication.

**Part B** compares dense matrix multiplication with a sparse implementation using the **Compressed Sparse Row (CSR)** format. The sparse experiments use target zero proportions of **90%, 95% and 99%**.

## Project Structure

- `matrix_benchmark.py` – dense matrix multiplication implementation and benchmark functions.
- `run_benchmarks.py` – runs the dense benchmarks and saves the measurements.
- `sparse_matrix.py` – sparse matrix generation, CSR conversion, multiplication and validation.
- `run_sparse_matrix.py` – runs the sparse benchmarks and saves the measurements.
- `check_sparse_results.py` – compares dense and CSR timings, including the measured conversion cost.
- `data/` – experimental CSV files and environment information.

The exact files and subfolders may vary depending on the current repository structure.

## Requirements

- Python 3

No additional third-party libraries are required by the benchmark scripts if they use only the Python standard library and the project's own modules.

## How to Run

Run the scripts from the `Assignment 02` folder.

**Dense benchmark**
```bash
python matrix_benchmark.py --test
python run_benchmarks.py
```

**Sparse benchmark**
```bash
python run_sparse_matrix.py --test
python run_sparse_matrix.py
```

**Check sparse results, including conversion time**
```bash
python -B check_sparse_results.py
```

The `-B` option prevents Python from writing `.pyc` files during that execution. If your checker file has a different name, use its actual filename.

## Experimental Setup

The benchmarks test matrix sizes of **10, 30, 60, 100 and 150**, with one warm-up run and five measured repetitions.

For Part A, the baseline, reordered and blocked dense implementations are compared using median execution times.

For Part B, dense multiplication is compared with CSR multiplication at target zero proportions of **90%, 95% and 99%**. The CSV files record execution times, input and output nonzero counts, estimated storage payload and correctness results.

The CSR multiplication timing excludes the conversion from dense to CSR. A separate conversion time is recorded once per size/sparsity configuration. The analysis script adds that conversion time to the median CSR multiplication time to estimate the cost when inputs start in dense format.

Memory figures are estimates of matrix storage payload, not measurements of peak process RAM.

## Results

The results are saved in CSV files and environment text files under `data/`.

- **Part A:** no single dense implementation is fastest for every tested matrix size.
- **Part B:** CSR has a lower measured multiplication time than dense multiplication in all tested configurations. CSR also remains faster in the conversion-aware comparison using the recorded conversion times.

These findings apply to the tested implementations, matrix sizes and experimental setup. The largest tested matrix size is **150 × 150**; this is not a claim that it is the maximum size the computer can handle.

## Objective

The objective is to compare dense matrix multiplication optimisations, evaluate the effect of sparsity and CSR storage, and understand how measured execution time and storage estimates relate to practical implementation costs.