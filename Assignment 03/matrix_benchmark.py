import argparse
import csv
import sys
import time
import numpy as np
import numba
from numba import njit, prange


SEED = 7


def make_matrices(n):
    """Create two deterministic matrices using float64 values."""

    a = np.empty((n, n), dtype=np.float64)
    b = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            a[i, j] = (((i * 31 + j * 17 + SEED) % 101) - 50) / 50.0
            b[i, j] = (((i * 13 + j * 29 + SEED * 3) % 101) - 50) / 50.0

    return a, b


@njit
def multiply_sequential(a, b, c):
    """Multiply matrices sequentially using i-j-k loop order."""

    n = a.shape[0]

    for i in range(n):
        for j in range(n):
            total = 0.0

            for k in range(n):
                total += a[i, k] * b[k, j]

            c[i, j] = total


@njit(parallel=True)
def multiply_parallel(a, b, c):
    """Multiply matrices in parallel by distributing rows."""

    n = a.shape[0]

    for i in prange(n):
        for j in range(n):
            total = 0.0

            for k in range(n):
                total += a[i, k] * b[k, j]

            c[i, j] = total


@njit
def checksum(c):
    """Calculate the sum of all matrix elements."""

    total = 0.0

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            total += c[i, j]

    return total


def assert_matrix_equal(actual, expected, tolerance=1e-9):
    """Check whether two matrices are equal within a tolerance."""

    if actual.shape != expected.shape:
        raise AssertionError("Matrices have different dimensions.")

    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            difference = abs(actual[i, j] - expected[i, j])

            if difference > tolerance:
                raise AssertionError(
                    f"Mismatch at ({i}, {j}): "
                    f"{actual[i, j]} != {expected[i, j]}"
                )


def run_tests():
    """Run correctness tests for both implementations."""

    # Test 1: multiplication with a known result.
    a = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float64
    )

    b = np.array(
        [[5.0, 6.0], [7.0, 8.0]],
        dtype=np.float64
    )

    expected = np.array(
        [[19.0, 22.0], [43.0, 50.0]],
        dtype=np.float64
    )

    c = np.zeros((2, 2), dtype=np.float64)
    multiply_sequential(a, b, c)
    assert_matrix_equal(c, expected)

    c = np.zeros((2, 2), dtype=np.float64)
    multiply_parallel(a, b, c)
    assert_matrix_equal(c, expected)

    # Test 2: multiplication by the identity matrix.
    identity = np.eye(2, dtype=np.float64)

    c = np.zeros((2, 2), dtype=np.float64)
    multiply_sequential(a, identity, c)
    assert_matrix_equal(c, a)

    c = np.zeros((2, 2), dtype=np.float64)
    multiply_parallel(a, identity, c)
    assert_matrix_equal(c, a)

    # Test 3: multiplication by the zero matrix.
    zero = np.zeros((2, 2), dtype=np.float64)

    c = np.zeros((2, 2), dtype=np.float64)
    multiply_sequential(a, zero, c)
    assert_matrix_equal(c, zero)

    c = np.zeros((2, 2), dtype=np.float64)
    multiply_parallel(a, zero, c)
    assert_matrix_equal(c, zero)

    # Test 4: compare both implementations with the same input.
    a, b = make_matrices(4)

    sequential_result = np.zeros((4, 4), dtype=np.float64)
    parallel_result = np.zeros((4, 4), dtype=np.float64)

    multiply_sequential(a, b, sequential_result)
    multiply_parallel(a, b, parallel_result)

    assert_matrix_equal(parallel_result, sequential_result)

    print("PASS: all correctness tests.", file=sys.stderr)


def benchmark(n, warmups, repetitions, thread_counts):
    """Measure sequential and parallel matrix multiplication."""

    a, b = make_matrices(n)
    samples = []

    # Sequential reference implementation.
    c = np.zeros((n, n), dtype=np.float64)

    for _ in range(warmups):
        multiply_sequential(a, b, c)

    for repetition in range(1, repetitions + 1):
        c = np.zeros((n, n), dtype=np.float64)

        start = time.perf_counter_ns()
        multiply_sequential(a, b, c)
        elapsed = time.perf_counter_ns() - start

        result_checksum = checksum(c)

        samples.append(
            ("sequential", 1, repetition, elapsed, result_checksum)
        )

    # Parallel implementation with different thread counts.
    for thread_count in thread_counts:
        numba.set_num_threads(thread_count)

        c = np.zeros((n, n), dtype=np.float64)

        # Warm-up ensures compilation is excluded from timing.
        for _ in range(warmups):
            multiply_parallel(a, b, c)

        for repetition in range(1, repetitions + 1):
            c = np.zeros((n, n), dtype=np.float64)

            start = time.perf_counter_ns()
            multiply_parallel(a, b, c)
            elapsed = time.perf_counter_ns() - start

            result_checksum = checksum(c)

            samples.append(
                ("parallel", thread_count, repetition,
                 elapsed, result_checksum)
            )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sequential and parallel matrix multiplication."
    )

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)

    parser.add_argument(
        "--threads",
        default="1,2,4,8",
        help="Comma-separated parallel thread counts"
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if not args.benchmark:
        parser.error("Use --test or --benchmark.")

    if args.n < 1 or args.warmups < 0 or args.repetitions < 1:
        parser.error(
            "n and repetitions must be positive; "
            "warmups cannot be negative."
        )

    try:
        thread_counts = [
            int(value.strip())
            for value in args.threads.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error("Thread counts must be integers separated by commas.")

    if not thread_counts or any(t < 1 for t in thread_counts):
        parser.error("All thread counts must be positive.")

    if len(thread_counts) != len(set(thread_counts)):
        parser.error("Thread counts must not contain duplicates.")

    if 1 not in thread_counts:
        parser.error("Include 1 thread to measure parallel overhead.")

    max_threads = numba.config.NUMBA_NUM_THREADS

    if any(t > max_threads for t in thread_counts):
        parser.error(
            f"Maximum configured Numba thread count is {max_threads}. "
            "Use lower values or configure NUMBA_NUM_THREADS before "
            "starting Python."
        )

    samples = benchmark(
        args.n,
        args.warmups,
        args.repetitions,
        thread_counts
    )

    # Write CSV data to standard output.
    writer = csv.writer(sys.stdout, lineterminator="\n")

    writer.writerow([
        "language",
        "method",
        "n",
        "threads",
        "repetition",
        "time_ns",
        "matrix_payload_bytes",
        "checksum",
        "correct"
    ])

    payload_bytes = 3 * args.n * args.n * 8

    for method, threads, repetition, elapsed, result_checksum in samples:
        writer.writerow([
            "Python",
            method,
            args.n,
            threads,
            repetition,
            elapsed,
            payload_bytes,
            format(result_checksum, ".17g"),
            "true"
        ])


if __name__ == "__main__":
    main()
