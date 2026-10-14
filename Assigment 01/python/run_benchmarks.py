
"""Run Python matrix benchmarks and save raw measurements."""

import argparse
import csv
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROGRAM = ROOT / "matrix_benchmark.py"
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "raw_measurements.csv"

FIELDS = [
    "language",
    "n",
    "repetition",
    "time_ns",
    "matrix_payload_bytes",
    "checksum",
    "correct"
]

def run_command(command):
    """Run a command and raise an error if it fails."""
    print("+", " ".join(str(part) for part in command))

    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sizes",
        default="10,30,60,100,150",
        help="Comma-separated matrix sizes"
    )

    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)

    args = parser.parse_args()

    try:
        sizes = [
            int(value.strip())
            for value in args.sizes.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error("Sizes must be integers separated by commas.")

    if not sizes or any(n < 1 for n in sizes):
        parser.error("All matrix sizes must be positive.")

    if args.repetitions < 1 or args.warmups < 0:
        parser.error(
            "Repetitions must be positive and warmups cannot be negative."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Correctness tests must pass before timing starts.
    print("Running correctness tests...")

    run_command([
        sys.executable,
        str(PROGRAM),
        "--test"
    ])

    all_rows = []

    for n in sizes:
        print(f"\nBenchmarking Python with n={n}...")

        result = run_command([
            sys.executable,
            str(PROGRAM),
            "--benchmark",
            "--n", str(n),
            "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions)
        ])

        reader = csv.DictReader(result.stdout.splitlines())
        rows = list(reader)

        if len(rows) != args.repetitions:
            raise RuntimeError(
                f"Expected {args.repetitions} rows for n={n}, "
                f"but received {len(rows)}."
            )

        if any(row["correct"] != "true" for row in rows):
            raise RuntimeError(f"Correctness flag failed for n={n}.")

        all_rows.extend(rows)

    output_path = CSV_PATH

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    environment_path = DATA_DIR / "environment.txt"

    environment = [
        f"Run timestamp: {datetime.now().astimezone().isoformat()}",
        f"Operating system: {platform.platform()}",
        f"Machine architecture: {platform.machine()}",
        f"Processor information: {platform.processor()}",
        f"Logical CPU count: {__import__('os').cpu_count()}",
        f"Python version: {sys.version}",
        f"Matrix sizes: {sizes}",
        f"Warm-up runs per size: {args.warmups}",
        f"Measured repetitions per size: {args.repetitions}",
        "Timing: multiplication kernel only.",
        "Memory: estimated matrix-element storage, not peak process RAM."
    ]

    environment_path.write_text(
        "\n".join(environment) + "\n",
        encoding="utf-8"
    )

    print(f"\nRaw measurements saved to: {output_path}")
    print(f"Environment details saved to: {environment_path}")
    print("Add your CPU model, physical core count and installed RAM to environment.txt.")

if __name__ == "__main__":
    main()
