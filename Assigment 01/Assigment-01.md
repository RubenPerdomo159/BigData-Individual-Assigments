# ASSIGMENT 01 - Basic Matrix Multiplication in Different Languages

## Overview

This assignment compares the performance of basic matrix multiplication in **Python, Java and C**.

The three implementations use the same algorithm, based on three nested loops, with a time complexity of **O(n³)**.

## Project Structure

- `python/` – Python implementation and benchmark.
- `java/` – Java implementation and benchmark.
- `c/` – C implementation and benchmark.
- `analysis/` – Analysis scripts and performance plots.
- `data/` – Experimental results and CSV files.
- `report/` – Final report.

## Requirements

- Python 3
- Java JDK
- GCC compiler (for C)

## How to Run

Run each implementation from its corresponding folder.

**Python**
```bash
python matrix_benchmark.py --test
python matrix_benchmark.py --benchmark
```

**Java**
Compile the Java files and run the corresponding test, benchmark and analysis programs.

**C**
Compile the programs with GCC and run the test, benchmark and analysis programs.

See each language's folder for its source files and scripts.

## Results

The benchmarks test matrix sizes of **10, 30, 60, 100 and 150**, using one warm-up run and five measured repetitions.

The results are saved in CSV files and used to compare execution times. The report discusses the results, algorithm complexity and memory estimates.

## Objective

The objective is to understand how the same algorithm performs in different programming languages and how experimental results relate to theoretical complexity.