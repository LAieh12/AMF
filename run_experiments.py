from __future__ import annotations

from morphogenic_lab import run_all_experiments, write_results_report


def main() -> None:
    results = run_all_experiments()
    report = write_results_report(results)
    for item in results["experiments"]:
        print(f"\n[{item['name']}]")
        for key, value in item.items():
            if key in {"name", "challenge"} or isinstance(value, dict):
                continue
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
