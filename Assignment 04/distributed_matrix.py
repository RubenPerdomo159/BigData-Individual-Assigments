import argparse
import sys
import time
import numpy as np

from distributed import Client, LocalCluster

from matrix_benchmark import (
    make_matrices,
    multiply_sequential,
    multiply_block,
    assert_matrix_equal
)


class DistributedMatrixExecutor:
    """Execute matrix multiplication using a local Dask cluster."""

    def __init__(self, num_workers):
        """Create a Dask cluster with the requested number of workers."""

        if num_workers < 1:
            raise ValueError("The number of workers must be positive.")

        start = time.perf_counter()

        self.cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None
        )

        self.client = Client(self.cluster)

        self.num_workers = num_workers
        self.setup_time = time.perf_counter() - start

    
    def multiply(self, a, b):
        """Multiply two matrices by distributing row blocks."""
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both inputs must be two-dimensional.")
        if a.shape[1] != b.shape[0]:
            raise ValueError("Matrix dimensions are not compatible.")
        if a.shape[0] == 0 or b.shape[1] == 0:
            raise ValueError("Matrices cannot have empty dimensions.")

        start_total = time.perf_counter()
        rows = a.shape[0]

        # Divide A into row blocks.
        start_distribution = time.perf_counter()
        blocks = []

        for i in range(self.num_workers):
            start_row = i * rows // self.num_workers
            end_row = (i + 1) * rows // self.num_workers

            if start_row < end_row:
                blocks.append(a[start_row:end_row, :])

        distribution_time = time.perf_counter() - start_distribution

        # Submit tasks directly with their input data.
        start_submission = time.perf_counter()
        futures = []

        for block in blocks:
            future = self.client.submit(
                multiply_block,
                block,
                b,
                pure=False
            )
            futures.append(future)

        submission_time = time.perf_counter() - start_submission

        # Collect the computed blocks.
        start_gather = time.perf_counter()
        result_blocks = self.client.gather(futures)
        gather_time = time.perf_counter() - start_gather

        c = np.vstack(result_blocks)
        total_time = time.perf_counter() - start_total

        timings = {
            "distribution_time": distribution_time,
            "submission_time": submission_time,
            "gather_time": gather_time,
            "operation_time": total_time
        }

        return c, timings


    def close(self):
        """Close the Dask client and cluster."""

        self.client.close()
        self.cluster.close()

    def __enter__(self):
        """Allow the executor to be used with a with statement."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the cluster when leaving the with statement."""

        self.close()


def run_tests():
    """Test distributed multiplication against the sequential version."""

    a, b = make_matrices(5)

    expected = np.zeros((5, 5), dtype=np.float64)
    multiply_sequential(a, b, expected)

    # Test with two workers and an uneven number of rows.
    with DistributedMatrixExecutor(2) as executor:
        result, timings = executor.multiply(a, b)

    assert_matrix_equal(result, expected)

    print("PASS: distributed multiplication is correct.", file=sys.stderr)


def run_example(n, workers):
    """Run one example of distributed matrix multiplication."""

    a, b = make_matrices(n)

    with DistributedMatrixExecutor(workers) as executor:
        result, timings = executor.multiply(a, b)

        print(f"Matrix size: {n} x {n}")
        print(f"Workers: {workers}")
        print(f"Setup time: {executor.setup_time:.6f} seconds")

        for name, elapsed in timings.items():
            print(f"{name}: {elapsed:.6f} seconds")

        print(f"Result shape: {result.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed matrix multiplication using Dask."
    )

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--example", action="store_true")

    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if not args.example:
        parser.error("Use --test or --example.")

    if args.n < 1 or args.workers < 1:
        parser.error("Matrix size and workers must be positive.")

    run_example(args.n, args.workers)


if __name__ == "__main__":
    main()
