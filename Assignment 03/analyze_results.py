import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "raw_measurements.csv"
SUMMARY_FILE = DATA_DIR / "summary.csv"
RESULTS = DATA_DIR


def quantile(values, fraction):
    """Calculate a quantile using linear interpolation."""
    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def main():
    if not RAW_FILE.exists():
        raise SystemExit(
            "Raw measurements not found. "
            "Run run_benchmarks.py first."
        )

    groups = {}

    with RAW_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            language = row["language"]
            method = row["method"]
            n = int(row["n"])
            threads = int(row["threads"])

            if row["correct"] != "true":
                raise SystemExit(
                    f"Incorrect result detected: {row}"
                )

            seconds = int(row["time_ns"]) / 1_000_000_000

            key = (language, n, method, threads)
            groups.setdefault(key, []).append(seconds)

    summary_rows = []

    # Calculate execution-time statistics for each configuration.
    for (language, n, method, threads), times in sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            item[0][2],
            item[0][3]
        )
    ):
        q1 = quantile(times, 0.25)
        q3 = quantile(times, 0.75)

        summary_rows.append({
            "language": language,
            "n": n,
            "method": method,
            "threads": threads,
            "repetitions": len(times),
            "median_seconds": statistics.median(times),
            "q1_seconds": q1,
            "q3_seconds": q3,
            "iqr_seconds": q3 - q1,
            "matrix_payload_bytes": 3 * n * n * 8,
            "speedup": "",
            "efficiency": ""
        })

    # Calculate speedup and efficiency for parallel executions.
    for row in summary_rows:
        if row["method"] != "parallel":
            continue

        sequential = next(
            (
                item for item in summary_rows
                if item["language"] == row["language"]
                and item["n"] == row["n"]
                and item["method"] == "sequential"
            ),
            None
        )

        if sequential is None:
            raise RuntimeError(
                f"Sequential measurements missing for n={row['n']}."
            )

        speedup = (
            sequential["median_seconds"]
            / row["median_seconds"]
        )

        row["speedup"] = speedup
        row["efficiency"] = speedup / row["threads"]

    fields = [
        "language",
        "n",
        "method",
        "threads",
        "repetitions",
        "median_seconds",
        "q1_seconds",
        "q3_seconds",
        "iqr_seconds",
        "matrix_payload_bytes",
        "speedup",
        "efficiency"
    ]

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Summary of measured execution times:\n")

    for row in summary_rows:
        print(
            f"{row['method']}, n={row['n']}, "
            f"threads={row['threads']}: "
            f"median={row['median_seconds']:.6f} s, "
            f"IQR={row['iqr_seconds']:.6f} s"
        )

        if row["method"] == "parallel":
            print(
                f"  Speedup={row['speedup']:.3f}, "
                f"Efficiency={row['efficiency']:.3f}"
            )

    print(f"\nSummary saved to: {SUMMARY_FILE}")

    # Plotting is optional; the analysis works without matplotlib.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "No graph generated. Install matplotlib if you want one: "
            "python -m pip install matplotlib"
        )
        return

    configurations = sorted({
        (row["method"], row["threads"])
        for row in summary_rows
    })

    # Runtime plot.
    for method, threads in configurations:
        selected = [
            row for row in summary_rows
            if row["method"] == method
            and row["threads"] == threads
        ]

        selected.sort(key=lambda row: row["n"])

        label = (
            "Sequential"
            if method == "sequential"
            else f"Parallel ({threads} threads)"
        )

        plt.plot(
            [row["n"] for row in selected],
            [row["median_seconds"] for row in selected],
            marker="o",
            label=label
        )

    plt.xlabel("Matrix size n (n × n)")
    plt.ylabel("Median multiplication time (seconds)")
    plt.title("Sequential vs Parallel Matrix Multiplication")
    plt.legend()
    plt.grid(True)

    plot_path = RESULTS / "runtime_plot.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Runtime graph saved to: {plot_path}")

    # Speedup plot.
    for threads in sorted({
        row["threads"] for row in summary_rows
        if row["method"] == "parallel"
    }):
        selected = [
            row for row in summary_rows
            if row["method"] == "parallel"
            and row["threads"] == threads
        ]

        selected.sort(key=lambda row: row["n"])

        plt.plot(
            [row["n"] for row in selected],
            [row["speedup"] for row in selected],
            marker="o",
            label=f"{threads} threads"
        )

    plt.axhline(y=1, linestyle="--")
    plt.xlabel("Matrix size n (n × n)")
    plt.ylabel("Speedup")
    plt.title("Parallel Speedup")
    plt.legend()
    plt.grid(True)

    speedup_path = RESULTS / "speedup_plot.png"
    plt.savefig(speedup_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Speedup graph saved to: {speedup_path}")

    # Efficiency plot.
    for threads in sorted({
        row["threads"] for row in summary_rows
        if row["method"] == "parallel"
    }):
        selected = [
            row for row in summary_rows
            if row["method"] == "parallel"
            and row["threads"] == threads
        ]

        selected.sort(key=lambda row: row["n"])

        plt.plot(
            [row["n"] for row in selected],
            [row["efficiency"] for row in selected],
            marker="o",
            label=f"{threads} threads"
        )

    plt.xlabel("Matrix size n (n × n)")
    plt.ylabel("Parallel efficiency")
    plt.title("Parallel Efficiency")
    plt.legend()
    plt.grid(True)

    efficiency_path = RESULTS / "efficiency_plot.png"
    plt.savefig(efficiency_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Efficiency graph saved to: {efficiency_path}")


if __name__ == "__main__":
    main()
