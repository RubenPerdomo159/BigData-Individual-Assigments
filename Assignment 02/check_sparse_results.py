import csv
from pathlib import Path
from statistics import median

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data" / "sparse" / "sparse_measurements.csv"

measurements = {}

with open(CSV_FILE, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        n = int(row["n"])
        sparsity = float(row["sparsity"])
        method = row["method"].lower()
        conversion_ns = float(row["conversion_time_ns"])

        key = (n, sparsity)

        if key not in measurements:
            measurements[key] = {
                "dense": [],
                "sparse": [],
                "conversion_ns": 0
            }

        if method in ("sparse", "csr"):
            measurements[key]["conversion_ns"] = conversion_ns

        if method == "dense":
            measurements[key]["dense"].append(
                int(row["time_ns"])
            )
        elif method in ("sparse", "csr"):
            measurements[key]["sparse"].append(
                int(row["time_ns"])
            )

print(
    f"{'n':>4} {'Ceros':>7} {'Densa (ms)':>12} "
    f"{'CSR (ms)':>12} {'Conv. (ms)':>12} "
    f"{'Total CSR':>12} {'Ventaja':>10}"
)

for (n, sparsity), values in sorted(measurements.items()):
    dense_ms = median(values["dense"]) / 1_000_000
    csr_ms = median(values["sparse"]) / 1_000_000
    conversion_ms = values["conversion_ns"] / 1_000_000

    total_csr_ms = csr_ms + conversion_ms
    speedup = dense_ms / total_csr_ms

    print(
        f"{n:4d} {sparsity * 100:6.0f}% "
        f"{dense_ms:12.3f} {csr_ms:12.3f} "
        f"{conversion_ms:12.4f} {total_csr_ms:12.4f} "
        f"{speedup:9.2f}x"
    )

    if total_csr_ms < dense_ms:
        print("     CSR sigue siendo más rápido incluyendo la conversión.")
    else:
        print("     La multiplicación densa es más rápida en este caso.")
