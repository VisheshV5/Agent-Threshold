"""Renders a calibration sweep to a markdown report + matplotlib chart."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from escalation.eval.calibration import CalibrationPoint


def render_calibration_report(points: list[CalibrationPoint], output_dir: Path, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "calibration_curve.png"
    report_path = output_dir / "report.md"

    thresholds = [p.threshold for p in points]
    false_alarm_rates = [p.metrics.false_alarm_rate for p in points]
    miss_rates = [p.metrics.miss_rate for p in points]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, false_alarm_rates, label="false alarm rate", marker="o", markersize=3)
    ax.plot(thresholds, miss_rates, label="miss rate", marker="s", markersize=3)
    ax.set_xlabel("threshold")
    ax.set_ylabel("rate")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    lines = [
        f"# {title}",
        "",
        f"Scenarios: {points[0].metrics.total} | Threshold points: {len(points)}",
        "",
        "![calibration curve](calibration_curve.png)",
        "",
        "| threshold | false_alarm_rate | miss_rate | accuracy | TP | TN | FA | miss |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in points:
        m = p.metrics
        lines.append(
            f"| {p.threshold:.2f} | {m.false_alarm_rate:.0%} | {m.miss_rate:.0%} | {m.accuracy:.0%} "
            f"| {m.true_positives} | {m.true_negatives} | {m.false_alarms} | {m.misses} |"
        )
    report_path.write_text("\n".join(lines) + "\n")
