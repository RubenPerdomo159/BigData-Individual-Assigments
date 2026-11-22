import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).parent / "data"

INPUT_FILE = DATA_DIR / "raw_measurements.csv"
SUMMARY_FILE = DATA_DIR / "summary.csv"

PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_results():
    """Load raw benchmark measurements from the CSV file."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("The raw measurements file is empty.")

    required_columns = {
        "language", "method", "n", "threads", "workers",
        "repetition", "time_ns", "checksum", "correct"
    }

    missing = required_columns - set(rows[0].keys())

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    return rows


def validate_results(rows):
    """Check correctness flags and compare checksums for each size."""

    all_correct = True
    checksums = defaultdict(list)

    for row in rows:
        if row["correct"].strip().lower() != "true":
            print(
                f"ERROR: Incorrect result for "
                f"{row['method']}, n={row['n']}"
            )
            all_correct = False

        checksums[int(row["n"])].append(float(row["checksum"]))

    for n, values in checksums.items():
        if not np.allclose(values, values[0], rtol=1e-9, atol=1e-9):
            print(f"WARNING: Different checksums found for n={n}")
            all_correct = False

    if all_correct:
        print("Correctness validation: PASS")
    else:
        print("Correctness validation: CHECK WARNINGS ABOVE")

    return all_correct


def calculate_statistics(rows):
    """Calculate median, quartiles and IQR for each configuration."""

    groups = defaultdict(list)

    for row in rows:
        threads = int(row["threads"]) if row["threads"].strip() else 1
        workers = int(row["workers"]) if row["workers"].strip() else 1

        key = (
            row["language"],
            row["method"],
            int(row["n"]),
            threads,
            workers
        )

        groups[key].append(int(row["time_ns"]))

    results = []

    for key, times in groups.items():
        q1 = float(np.percentile(times, 25))
        median = float(np.median(times))
        q3 = float(np.percentile(times, 75))

        results.append({
            "language": key[0],
            "method": key[1],
            "n": key[2],
            "threads": key[3],
            "workers": key[4],
            "repetitions": len(times),
            "median_time_ns": median,
            "q1_time_ns": q1,
            "q3_time_ns": q3,
            "iqr_time_ns": q3 - q1
        })

    return results


def calculate_performance_metrics(results):
    """Calculate parallel and distributed speedup and efficiency."""

    lookup = {}

    for row in results:
        key = (
            row["method"],
            row["n"],
            row["threads"],
            row["workers"]
        )
        lookup[key] = row["median_time_ns"]

    for row in results:
        method = row["method"]
        n = row["n"]

        row["speedup"] = ""
        row["efficiency"] = ""

        if method == "parallel":
            sequential_time = lookup.get(("sequential", n, 1, 1))

            if sequential_time is not None and row["median_time_ns"] > 0:
                speedup = sequential_time / row["median_time_ns"]
                row["speedup"] = speedup
                row["efficiency"] = speedup / row["threads"]

        elif method == "distributed":
            baseline_time = lookup.get(("distributed", n, 1, 1))

            if baseline_time is not None and row["median_time_ns"] > 0:
                speedup = baseline_time / row["median_time_ns"]
                row["speedup"] = speedup
                row["efficiency"] = speedup / row["workers"]

    return results


def save_summary(results):
    """Save calculated statistics and performance metrics."""

    fieldnames = [
        "language", "method", "n", "threads", "workers",
        "repetitions", "median_time_ns", "q1_time_ns",
        "q3_time_ns", "iqr_time_ns", "speedup", "efficiency"
    ]

    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Summary saved to: {SUMMARY_FILE}")


def plot_parallel_times(results):
    """Plot sequential and parallel execution times."""

    plt.figure(figsize=(9, 5))

    sequential = [
        row for row in results
        if row["method"] == "sequential"
    ]
    sequential.sort(key=lambda row: row["n"])

    plt.plot(
        [row["n"] for row in sequential],
        [row["median_time_ns"] / 1e6 for row in sequential],
        marker="o",
        label="Sequential"
    )

    thread_counts = sorted(set(
        row["threads"] for row in results
        if row["method"] == "parallel"
    ))

    for threads in thread_counts:
        selected = [
            row for row in results
            if row["method"] == "parallel"
            and row["threads"] == threads
        ]
        selected.sort(key=lambda row: row["n"])

        plt.plot(
            [row["n"] for row in selected],
            [row["median_time_ns"] / 1e6 for row in selected],
            marker="o",
            label=f"Parallel ({threads} threads)"
        )

    plt.xlabel("Matrix size (n)")
    plt.ylabel("Median execution time (ms)")
    plt.title("Sequential vs Parallel Matrix Multiplication")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "parallel_execution_times.png")
    plt.close()



def plot_distributed_times(results):
    """Plot distributed execution times for different worker counts."""

    plt.figure(figsize=(9, 5))

    worker_counts = sorted(set(
        row["workers"] for row in results
        if row["method"] == "distributed"
    ))

    for workers in worker_counts:
        selected = [
            row for row in results
            if row["method"] == "distributed"
            and row["workers"] == workers
        ]
        selected.sort(key=lambda row: row["n"])

        plt.plot(
            [row["n"] for row in selected],
            [row["median_time_ns"] / 1e6 for row in selected],
            marker="o",
            label=f"{workers} worker(s)"
        )

    plt.xlabel("Matrix size (n)")
    plt.ylabel("Median operation time (ms)")
    plt.title("Distributed Matrix Multiplication")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "distributed_execution_times.png")
    plt.close()


def plot_parallel_speedup(results):
    """Plot parallel speedup for different thread counts."""

    plt.figure(figsize=(9, 5))

    for threads in sorted(set(
        row["threads"] for row in results
        if row["method"] == "parallel"
    )):
        if threads == 1:
            continue

        selected = [
            row for row in results
            if row["method"] == "parallel"
            and row["threads"] == threads
            and row["speedup"] != ""
        ]
        selected.sort(key=lambda row: row["n"])

        plt.plot(
            [row["n"] for row in selected],
            [row["speedup"] for row in selected],
            marker="o",
            label=f"{threads} threads"
        )

    plt.xlabel("Matrix size (n)")
    plt.ylabel("Speedup")
    plt.title("Parallel Speedup")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "parallel_speedup.png")
    plt.close()


def plot_distributed_speedup(results):
    """Plot distributed speedup relative to one worker."""

    plt.figure(figsize=(9, 5))

    for workers in sorted(set(
        row["workers"] for row in results
        if row["method"] == "distributed"
    )):
        if workers == 1:
            continue

        selected = [
            row for row in results
            if row["method"] == "distributed"
            and row["workers"] == workers
            and row["speedup"] != ""
        ]
        selected.sort(key=lambda row: row["n"])

        plt.plot(
            [row["n"] for row in selected],
            [row["speedup"] for row in selected],
            marker="o",
            label=f"{workers} workers"
        )

    plt.xlabel("Matrix size (n)")
    plt.ylabel("Speedup")
    plt.title("Distributed Speedup (relative to 1 worker)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "distributed_speedup.png")
    plt.close()


def main():
    rows = load_results()

    print(f"Loaded {len(rows)} measurements.")

    validate_results(rows)

    results = calculate_statistics(rows)
    results = calculate_performance_metrics(results)

    save_summary(results)

    plot_parallel_times(results)
    plot_distributed_times(results)
    plot_parallel_speedup(results)
    plot_distributed_speedup(results)

    print(f"Plots saved in: {PLOTS_DIR}")
    print("Analysis completed.")


if __name__ == "__main__":
    main()
