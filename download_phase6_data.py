from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


DATA_DIR = Path("data") / "uci_phase6"

FILES = {
    "spambase.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
    "ionosphere.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
    "sonar.all-data": "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
    "pendigits.tra": "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra",
    "pendigits.tes": "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes",
    "sat.trn": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn",
    "sat.tst": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst",
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        path = DATA_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            print(f"exists: {path}")
            continue
        print(f"download: {url}")
        urlretrieve(url, path)
        print(f"saved: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
