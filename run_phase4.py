from __future__ import annotations

from phase4_benchmark import run_phase4, write_phase4_report


def main() -> None:
    results = run_phase4()
    report = write_phase4_report(results)
    print(f"report: {report}")
    for exp in results["experiments"]:
        print(f"\n[{exp['name']}]")
        if exp["name"] == "phase4_large_adversarial_improvement":
            for name, row in exp["models"].items():
                print(
                    f"{name}: clean={row['clean_accuracy']:.6f} "
                    f"adv={row['adversarial_accuracy']:.6f}"
                )
        elif "phase4" in exp:
            row = exp["phase4"]
            if "new_after" in row:
                print(
                    f"phase4: old_after={row['old_after']:.6f} "
                    f"new_after={row['new_after']:.6f}"
                )
            else:
                print(
                    f"phase4: mean={row['mean_accuracy']:.6f} "
                    f"last={row['last_chunk_accuracy']:.6f}"
                )


if __name__ == "__main__":
    main()
