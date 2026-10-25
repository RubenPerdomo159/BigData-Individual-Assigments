import argparse
import csv
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from matrix_benchmark import multiply
from sparse_matrix import (
    SEED,
    assert_matrix_equal,
    count_nonzeros,
    csr_to_dense,
    dense_to_csr,
    make_sparse_matrices,
    multiply_csr,
    run_tests as run_sparse_tests,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sparse"

FIELDS = [
    "language",
    "method",
    "n",
    "sparsity",
    "density",
    "repetition",
    "time_ns",
    "conversion_time_ns",
    "input_nonzeros_a",
    "input_nonzeros_b",
    "output_nonzeros",
    "matrix_payload_bytes",
    "checksum",
    "correct",
]


def checksum(matrix):
    """Calculate the sum of all matrix elements."""
    return sum(sum(row) for row in matrix)


def dense_multiply(a, b):
    """Multiply dense matrices and return the result."""
    n = len(a)
    c = [[0.0] * n for _ in range(n)]
    multiply(a, b, c)
    return c


def estimated_csr_bytes(csr):
    """Estimate CSR storage payload, excluding Python object overhead."""
    values, column_indices, row_pointers, _ = csr

    return (
        len(values) * 8
        + len(column_indices) * 8
        + len(row_pointers) * 8
    )


def benchmark_case(n, sparsity, warmups, repetitions, seed):
    """Benchmark dense and sparse multiplication on matched inputs."""
    a, b = make_sparse_matrices(n, sparsity, seed)

    input_nonzeros_a = count_nonzeros(a)
    input_nonzeros_b = count_nonzeros(b)

    # Measure the conversion from dense matrices to CSR separately.
    start = time.perf_counter_ns()
    csr_a = dense_to_csr(a)
    csr_b = dense_to_csr(b)
    conversion_time_ns = time.perf_counter_ns() - start

    # Calculate the dense reference result outside the timed measurements.
    expected = dense_multiply(a, b)

    sparse_check = csr_to_dense(multiply_csr(csr_a, csr_b))
    assert_matrix_equal(sparse_check, expected)

    methods = [
        ("dense", lambda: dense_multiply(a, b)),
        ("sparse", lambda: multiply_csr(csr_a, csr_b)),
    ]

    samples = []

    for method_name, method in methods:
        for _ in range(warmups):
            method()

        for repetition in range(1, repetitions + 1):
            start = time.perf_counter_ns()
            result = method()
            elapsed = time.perf_counter_ns() - start

            if method_name == "sparse":
                result_dense = csr_to_dense(result)
                output_nonzeros = len(result[0])

                memory_bytes = (
                    estimated_csr_bytes(csr_a)
                    + estimated_csr_bytes(csr_b)
                    + estimated_csr_bytes(result)
                )
            else:
                result_dense = result
                output_nonzeros = count_nonzeros(result)

                # Estimate storage for A, B and C using 8 bytes per element.
                memory_bytes = 3 * n * n * 8

            # Validate each measured result against the dense reference.
            assert_matrix_equal(result_dense, expected)
            correct = "true"

            samples.append({
                "language": "Python",
                "method": method_name,
                "n": n,
                "sparsity": sparsity,
                "density": 1.0 - sparsity,
                "repetition": repetition,
                "time_ns": elapsed,
                "conversion_time_ns": (
                    conversion_time_ns if method_name == "sparse" else 0
                ),
                "input_nonzeros_a": input_nonzeros_a,
                "input_nonzeros_b": input_nonzeros_b,
                "output_nonzeros": output_nonzeros,
                "matrix_payload_bytes": memory_bytes,
                "checksum": format(checksum(result_dense), ".17g"),
                "correct": correct,
            })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dense and sparse matrix multiplication."
    )
    parser.add_argument(
        "--sizes",
        default="10,30,60,100,150",
        help="Comma-separated matrix sizes",
    )
    parser.add_argument(
        "--sparsities",
        default="0.90,0.95,0.99",
        help="Comma-separated proportions of zero entries",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.test:
        run_sparse_tests()
        return

    try:
        sizes = [
            int(value.strip())
            for value in args.sizes.split(",")
            if value.strip()
        ]
        sparsities = [
            float(value.strip())
            for value in args.sparsities.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error("Sizes and sparsities must contain valid numbers.")

    if not sizes or any(n < 1 for n in sizes):
        parser.error("All matrix sizes must be positive.")

    if not sparsities or any(
        sparsity < 0.0 or sparsity > 1.0
        for sparsity in sparsities
    ):
        parser.error("Sparsities must be between 0 and 1.")

    if args.repetitions < 1 or args.warmups < 0:
        parser.error(
            "Repetitions must be positive and warmups cannot be negative."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print("Running sparse correctness tests...")
    run_sparse_tests()

    for n in sizes:
        for sparsity in sparsities:
            print(f"\nBenchmarking n={n}, sparsity={sparsity:.0%}...")

            rows = benchmark_case(
                n,
                sparsity,
                args.warmups,
                args.repetitions,
                args.seed,
            )
            all_rows.extend(rows)

    output_path = DATA_DIR / "sparse_measurements.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    environment_path = DATA_DIR / "sparse_environment.txt"

    environment = [
        f"Run timestamp: {datetime.now().astimezone().isoformat()}",
        f"Operating system: {platform.platform()}",
        f"Machine architecture: {platform.machine()}",
        f"Processor information: {platform.processor()}",
        f"Logical CPU count: {os.cpu_count()}",
        f"Python version: {sys.version}",
        f"Matrix sizes: {sizes}",
        f"Zero proportions: {sparsities}",
        f"Random seed: {args.seed}",
        f"Warm-up runs per method and configuration: {args.warmups}",
        f"Measured repetitions per method and configuration: {args.repetitions}",
        "Storage: CSR (values, column indices, row pointers).",
        "Timing: multiplication only; dense-to-CSR conversion measured separately.",
        "Memory: estimated storage payload, excluding Python object overhead; not peak process RAM.",
        "Sparse timing excludes converting the CSR output back to dense for validation.",
        "Conversion time is measured once per size and sparsity configuration, not repeated.",
    ]

    environment_path.write_text(
        "\n".join(environment) + "\n",
        encoding="utf-8",
    )

    print(f"\nMeasurements saved to: {output_path}")
    print(f"Environment details saved to: {environment_path}")


if __name__ == "__main__":
    main()