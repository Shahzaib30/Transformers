"""Move notebook files into the notebooks folder.

Run from the repository root.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    notebooks_dir = ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    for item in ROOT.iterdir():
        if item.suffix == ".ipynb":
            dest = notebooks_dir / item.name
            print(f"Moving notebook: {item.name} -> notebooks/")
            shutil.move(str(item), str(dest))

    print("Reorganization complete.")


if __name__ == "__main__":
    main()
