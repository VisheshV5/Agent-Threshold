import asyncio

from threshold.eval.ablation import run_ablation
from threshold.eval.calibration import sweep_thresholds
from threshold.eval.data.scenarios_m1 import SCENARIOS
from threshold.eval.eval_policy import build_eval_policy
from threshold.eval.report import render_calibration_report


def run(coro):
    return asyncio.run(coro)


def test_render_calibration_report_writes_chart_and_markdown(tmp_path):
    points = run(sweep_thresholds(SCENARIOS, "balanced"))
    render_calibration_report(points, tmp_path, title="Milestone 2 calibration curve")

    chart = tmp_path / "calibration_curve.png"
    report = tmp_path / "report.md"
    assert chart.exists()
    assert chart.stat().st_size > 0
    assert report.exists()
    content = report.read_text()
    assert "Milestone 2 calibration curve" in content
    assert "false_alarm_rate" in content
    assert "Per-signal ablation" not in content  # omitted when not provided


def test_render_calibration_report_includes_ablation_section_when_provided(tmp_path):
    points = run(sweep_thresholds(SCENARIOS, "balanced"))
    ablation_results = run(run_ablation(SCENARIOS, build_eval_policy("balanced")))
    render_calibration_report(points, tmp_path, title="Milestone 3 report", ablation_results=ablation_results)

    content = (tmp_path / "report.md").read_text()
    assert "Per-signal ablation" in content
    assert "reversibility" in content
    assert "novelty" in content
