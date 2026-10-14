
"""Summarize raw matrix benchmark measurements."""

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
            "Run Assignment01/run_benchmarks.py first."
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
            n = int(row["n"])

            # Convert nanoseconds to seconds.
            seconds = int(row["time_ns"]) / 1_000_000_000

            key = (language, n)
            groups.setdefault(key, []).append(seconds)

    summary_rows = []

    for (language, n), times in sorted(
        groups.items(),
        key=lambda item: (item[0][1], item[0][0])
    ):
        q1 = quantile(times, 0.25)
        q3 = quantile(times, 0.75)

        summary_rows.append({
            "language": language,
            "n": n,
            "repetitions": len(times),
            "median_seconds": statistics.median(times),
            "q1_seconds": q1,
            "q3_seconds": q3,
            "iqr_seconds": q3 - q1,
            "matrix_payload_bytes": 3 * n * n * 8
        })

    fields = [
        "language",
        "n",
        "repetitions",
        "median_seconds",
        "q1_seconds",
        "q3_seconds",
        "iqr_seconds",
        "matrix_payload_bytes"
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
            f"{row['language']}, n={row['n']}: "
            f"median={row['median_seconds']:.6f} s, "
            f"IQR={row['iqr_seconds']:.6f} s, "
            f"repetitions={row['repetitions']}"
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

    languages = sorted({
        row["language"] for row in summary_rows
    })

    for language in languages:
        selected = [
            row for row in summary_rows
            if row["language"] == language
        ]

        selected.sort(key=lambda row: row["n"])

        plt.plot(
            [row["n"] for row in selected],
            [row["median_seconds"] for row in selected],
            marker="o",
            label=language
        )

    plt.xlabel("Matrix size n (n × n)")
    plt.ylabel("Median multiplication time (seconds)")
    plt.title("Basic matrix multiplication: Python")
    plt.legend()
    plt.grid(True)

    plot_path = RESULTS / "runtime_plot.png"

    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Graph saved to: {plot_path}")

if __name__ == "__main__":
    main()
