import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Set repo root and add to sys.path for local imports
ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"

# Available NPZs in artifacts/ (adjust as new ones arrive)
NPZ_PATHS = [
    ARTIFACTS / "en-len16-n8192-rep1.npz",
    ARTIFACTS / "en-len16-n8192-rep2.npz",
    ARTIFACTS / "zh-llm-len16-n8192-rep1.npz",
    ARTIFACTS / "prng-len16-n8192-rep1.npz",
    ARTIFACTS / "prng-len16-n8192-rep2.npz",
]

# Optional: integrity helpers
try:
    from integrity_check import summarize_dataset, print_report
except ImportError:
    summarize_dataset = None
    print_report = None


def list_npz(artifacts: Path = ARTIFACTS) -> list[Path]:
    return sorted(artifacts.glob("*.npz"))


def plot_position_means(path: Path, label: str):
    arr = np.load(path)["sequences"]
    means = arr.mean(axis=0)
    plt.plot(range(len(means)), means, marker="o", label=label)

