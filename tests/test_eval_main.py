import sys

from threshold.eval.__main__ import main


def test_main_writes_report_with_ablation_section(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["threshold.eval", "--out", str(tmp_path)])
    main()

    assert (tmp_path / "calibration_curve.png").exists()
    content = (tmp_path / "report.md").read_text()
    assert "balanced profile, 40 scenarios" in content
    assert "Per-signal ablation" in content
