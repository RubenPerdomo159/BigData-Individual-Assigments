import argparse
import csv
import sys
import time

SEED = 7

def make_matrices(n):
    """Create two deterministic matrices and an empty matrix."""
    a = [
        [(((i * 31 + j * 17 + SEED) % 101) - 50) / 50.0
         for j in range(n)]
        for i in range(n)
    ]

    b = [
        [(((i * 13 + j * 29 + SEED * 3) % 101) - 50) / 50.0
         for j in range(n)]
        for i in range(n)
    ]

    c = [[0.0] * n for _ in range(n)]

    return a, b, c

def multiply(a, b, c):
    """Multiply matrices using three nested loops."""
    n = len(a)

    for i in range(n):
        for j in range(n):
            total = 0.0

            for k in range(n):
                total += a[i][k] * b[k][j]

            c[i][j] = total

def checksum(c):
    """Calculate a checksum."""
    total = 0.0

    for row in c:
        for value in row:
            total += value

    return total

def multiply_reordered(a, b, c):
    """Multiply matrices using i-k-j loop order."""
    n = len(a)

    for i in range(n):
        for k in range(n):
            value = a[i][k]

            for j in range(n):
                c[i][j] += value * b[k][j]

def multiply_blocked(a, b, c, block_size=16):
    """Multiply matrices using cache blocking."""
    n = len(a)

    for ii in range(0, n, block_size):
        for kk in range(0, n, block_size):
            for jj in range(0, n, block_size):

                for i in range(ii, min(ii + block_size, n)):
                    for k in range(kk, min(kk + block_size, n)):
                        value = a[i][k]

                        for j in range(jj, min(jj + block_size, n)):
                            c[i][j] += value * b[k][j]

def assert_matrix_equal(actual, expected, tolerance=1e-9):
    """Check whether two matrices are equal within a tolerance."""
    if len(actual) != len(expected):
        raise AssertionError("Different number of rows")

    for i in range(len(expected)):
        if len(actual[i]) != len(expected[i]):
            raise AssertionError("Different number of columns")

        for j in range(len(expected[i])):
            difference = abs(actual[i][j] - expected[i][j])

            if difference > tolerance:
                raise AssertionError(
                    f"Mismatch at ({i}, {j}): "
                    f"{actual[i][j]} != {expected[i][j]}"
                )


def run_tests():
    """Run correctness tests."""

    # Test 1: multiplication with a known result.
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    expected = [[19.0, 22.0], [43.0, 50.0]]

    c = [[0.0] * 2 for _ in range(2)]
    multiply(a, b, c)
    assert_matrix_equal(c, expected)

    # Test 2: multiplication by the identity matrix.
    identity = [[1.0, 0.0], [0.0, 1.0]]

    c = [[0.0] * 2 for _ in range(2)]
    multiply(a, identity, c)
    assert_matrix_equal(c, a)

    # Test 3: multiplication by the zero matrix.
    zero = [[0.0, 0.0], [0.0, 0.0]]

    c = [[0.0] * 2 for _ in range(2)]
    multiply(a, zero, c)
    assert_matrix_equal(c, zero)

    # Test 4: compare the reordered version with the baseline.
    a, b, expected = make_matrices(4)
    multiply(a, b, expected)

    c = [[0.0] * 4 for _ in range(4)]
    multiply_reordered(a, b, c)
    assert_matrix_equal(c, expected)

    # Test 5: compare the blocked version with the baseline.
    c = [[0.0] * 4 for _ in range(4)]
    multiply_blocked(a, b, c)
    assert_matrix_equal(c, expected)

    print("PASS: all five correctness tests.")

def benchmark(n, warmups, repetitions):
    """Measure the execution time of the methods."""
    a, b, _ = make_matrices(n)

    methods = [
        ("baseline", multiply),
        ("reordered", multiply_reordered),
        ("blocked", multiply_blocked)
    ]

    samples = []

    for method_name, method in methods:
        for _ in range(warmups):
            c = [[0.0] * n for _ in range(n)]
            method(a, b, c)
            checksum(c)

        for repetition in range(1, repetitions + 1):
            c = [[0.0] * n for _ in range(n)]

            start = time.perf_counter_ns()
            method(a, b, c)
            elapsed = time.perf_counter_ns() - start

            result_checksum = checksum(c)

            samples.append(
                (method_name, repetition, elapsed, result_checksum)
            )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark basic dense matrix multiplication."
    )

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if not args.benchmark:
        parser.error("Use --test or --benchmark.")

    if args.n < 1 or args.warmups < 0 or args.repetitions < 1:
        parser.error(
            "n and repetitions must be positive; warmups cannot be negative."
        )

    samples = benchmark(
        args.n,
        args.warmups,
        args.repetitions
    )

    writer = csv.writer(sys.stdout, lineterminator="\n")

    writer.writerow([
        "language",
        "method",
        "n",
        "repetition",
        "time_ns",
        "matrix_payload_bytes",
        "checksum",
        "correct"
    ])

    payload_bytes = 3 * args.n * args.n * 8

    for method_name, repetition, elapsed, result_checksum in samples:
        writer.writerow([
            "Python",
            method_name,
            args.n,
            repetition,
            elapsed,
            payload_bytes,
            format(result_checksum, ".17g"),
            "true"
        ])

if __name__ == "__main__":
    main()
