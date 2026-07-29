import asyncio

from escalation.eval.calibration import sweep_thresholds
from escalation.eval.data.scenarios_m1 import SCENARIOS
from escalation.eval.report import render_calibration_report


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
