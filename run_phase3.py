from __future__ import annotations

from phase3_benchmark import run_phase3, write_phase3_report


def main() -> None:
    results = run_phase3()
    report = write_phase3_report(results)
    print(f"report: {report}")
    for exp in results["experiments"]:
        print(f"\n[{exp['name']}]")
        if "models" in exp:
            for row in exp["models"]:
                name = row["name"]
                clean = row.get("clean_accuracy", row.get("mean_accuracy"))
                if clean is None:
                    clean = row.get("old_accuracy_after")
                print(f"{name}: {clean:.6f}")
        else:
            print("ok")


if __name__ == "__main__":
    main()
