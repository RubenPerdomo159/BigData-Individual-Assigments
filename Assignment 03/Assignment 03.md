# Assignment 03 - Parallel Matrix Multiplication

## Overview

This assignment investigates the impact of parallel execution on matrix multiplication using shared-memory parallelism.

The main objective is to compare a sequential implementation with a parallel implementation using different numbers of threads, evaluating execution time, speedup, parallel efficiency and memory requirements.

The experiments are implemented in Python using NumPy and Numba.

## Project Structure

```text
Assignment 03/
└── python/
    ├── matrix_benchmark.py
    ├── run_benchmarks.py
    ├── analyze_results.py
    └── data/
        ├── raw_measurements.csv
        ├── environment.txt
        ├── summary.csv
        ├── runtime_plot.png
        ├── speedup_plot.png
        └── efficiency_plot.png
```

## Implementations

Two matrix multiplication implementations are evaluated:

- **Sequential:** Executes the standard triple-loop matrix multiplication algorithm.
- **Parallel:** Uses Numba's `prange` to distribute the outer loop iterations across multiple threads.

Both implementations use the same algorithm, input matrices, numerical type (`float64`) and memory layout. This allows the comparison to focus on the effect of parallel execution.

The parallel implementation assigns different output matrix rows to parallel iterations, avoiding overlapping writes to the output matrix.

## Experimental Methodology

The experiments follow a consistent benchmarking procedure:

- Matrix sizes: 100, 200, 300, 400 and 500.
- Thread configurations: 1, 2, 4 and 8, depending on the available Numba thread configuration.
- Warm-up executions: 1.
- Measured repetitions: 5.
- Timing function: `time.perf_counter_ns()`.

The sequential implementation provides the baseline. The parallel implementation is evaluated using each selected thread configuration, including one thread to identify the overhead introduced by parallel execution.

Compilation and warm-up executions are excluded from the measured times.

## Performance Metrics

The analysis evaluates the following metrics:

**Execution Time**

The time required to perform matrix multiplication, measured in nanoseconds and converted to seconds during analysis.

**Median and Variability**

For each configuration, the median, first quartile (Q1), third quartile (Q3) and interquartile range (IQR) are calculated.

**Speedup**

Measures the performance improvement of parallel execution compared with the sequential baseline:

\[
S(p)=\frac{T_{seq}(n)}{T_{parallel}(n,p)}
\]

**Parallel Efficiency**

Measures how effectively the available threads are used:

\[
E(p)=\frac{S(p)}{p}
\]

Where:
- \(T_{seq}(n)\) is the sequential execution time.
- \(T_{parallel}(n,p)\) is the parallel execution time using \(p\) threads.
- \(p\) is the number of threads.

## Memory Measurement

Memory usage is estimated using the combined array payload of the two input matrices and the output matrix:

\[
M=3\cdot n^2\cdot 8
\]

Each matrix element uses 8 bytes (`float64`).

This represents the estimated matrix storage, not the peak memory consumption of the Python process.

## Correctness Validation

Before benchmarking, correctness tests are executed to validate both implementations.

The tests include:
- Multiplication using known matrices.
- Identity matrix multiplication.
- Zero matrix multiplication.
- Comparison between sequential and parallel results.

All measured executions are also associated with a correctness flag in the raw CSV file.

## How to Run

Run the commands from the `python/` directory.

### 1. Run benchmarks

```bash
python run_benchmarks.py
```

Optional parameters can be specified:

```bash
python run_benchmarks.py --sizes 100,200,300,400,500 --threads 1,2,4,8 --warmups 1 --repetitions 5
```

This executes the correctness tests, runs the experiments and saves the raw measurements and environment information in the `data/` directory.

### 2. Analyze results

```bash
python analyze_results.py
```

This generates the summary CSV file and the following graphs:

- `runtime_plot.png`: comparison of sequential and parallel execution times.
- `speedup_plot.png`: parallel speedup for each thread configuration.
- `efficiency_plot.png`: parallel efficiency for each thread configuration.

## Dependencies

The project requires:

- Python 3
- NumPy
- Numba
- Matplotlib (optional, for generating graphs)

Install the required packages using:

```bash
python -m pip install numpy numba matplotlib
```

## Reproducibility

The raw measurements and environment information are stored in the `data/` directory.

The environment file records the operating system, processor information, Python and library versions, thread configuration and benchmark parameters.

The matrix generation uses deterministic inputs to ensure that the sequential and parallel implementations work with equivalent data.

## Conclusion

The experiments aim to determine how the number of threads affects matrix multiplication performance, identify the matrix sizes for which parallel execution becomes beneficial, and evaluate the impact of parallelization overhead and thread utilization.