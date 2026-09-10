"""Render the frozen report with matplotlib; no inference or metric pooling."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
ORDER = [
    ("screening_baseline", "Baseline"),
    ("idle_warmup", "Idle warmups"),
    ("final_aware_partials", "Defer partials"),
    ("silence_04", "Silence 0.4 s"),
    ("silence_035", "Silence 0.35 s"),
    ("conservative_marian", "Conservative routing"),
    ("terminology", "Terminology examples"),
    ("onnx_vad", "ONNX VAD"),
]


def main():
    report = json.loads((ROOT / "comparison.json").read_text())
    rows = report["replays"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.7), sharey=True)
    style = {"e4b": ("#2459b8", "o", -0.13), "e2b": ("#a94d0a", "D", 0.13)}
    for ax, endpoint, title in zip(
        axes, ("silence", "smart_cut", "hard_cut"), ("Silence endings", "Smart cuts", "Hard cuts"), strict=True
    ):
        maximum = 0.0
        for y, (experiment, _) in enumerate(ORDER):
            for size, (color, marker, offset) in style.items():
                matches = [
                    row
                    for row in rows
                    if row["experiment"] == experiment and row["size"] == size and row["endpoint"] == endpoint
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one frozen cohort for {experiment}/{size}/{endpoint}")
                stats = matches[0]["metrics"]["speech_end_to_final_ms"]
                median, tail = stats["p50"] / 1000, stats["p95"] / 1000
                maximum = max(maximum, tail)
                ax.plot([median, tail], [y + offset] * 2, color=color, linewidth=1.4, alpha=0.8)
                ax.plot(tail, y + offset, "|", color=color, markersize=7)
                ax.plot(median, y + offset, marker, color=color, markersize=5, label=size.upper() if y == 0 else None)
        ax.set_title(title, loc="left", weight="bold", pad=14)
        ax.set_xlabel("Seconds to final payload readiness", labelpad=10)
        ax.set_xlim(0, maximum * 1.08)
        ax.set_yticks(range(len(ORDER)), [label for _, label in ORDER])
        ax.grid(axis="x", color="#dfe3e8", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvline(1, color="#555d68", linestyle=":", linewidth=1)
        ax.tick_params(axis="both", length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].invert_yaxis()
    axes[2].legend(loc="upper right", bbox_to_anchor=(1, 1.18), ncol=2, frameon=False)
    fig.suptitle("Mac latency experiments: final captions", x=0.015, y=0.99, ha="left", weight="bold", fontsize=18)
    fig.text(0.015, 0.935, "Dots show medians; lines extend to p95. These are distributions, not confidence intervals.")
    fig.text(
        0.015,
        0.035,
        "Same 45-second English input; three runs per model/configuration; partial cadence 0.6 s. "
        "The dotted line marks 1 second.\n"
        "Lower silence thresholds change segmentation and sample counts. No browser delivery timing is included; "
        "see the matched-caption analysis.",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.835, bottom=0.17, wspace=0.25)
    fig.savefig(ROOT / "latency.png", dpi=170, facecolor="white")
    fig.savefig(ROOT / "latency.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
