"""Regenerates the calibration report: `uv run python -m threshold.eval`.

Runs the full scenario suite through the threshold sweep and the
per-signal ablation, then writes a markdown report and chart. Uses the
heuristic signal implementations only -- no API key, no network calls.
"""

import argparse
import asyncio
from pathlib import Path

from threshold.eval.ablation import run_ablation
from threshold.eval.calibration import sweep_thresholds
from threshold.eval.data.scenarios_m1 import SCENARIOS
from threshold.eval.eval_policy import build_eval_policy
from threshold.eval.report import render_calibration_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="balanced", choices=["conservative", "balanced", "autonomous"])
    parser.add_argument("--out", type=Path, default=Path("reports/latest"))
    args = parser.parse_args()

    points = asyncio.run(sweep_thresholds(SCENARIOS, args.profile))
    ablation = asyncio.run(run_ablation(SCENARIOS, build_eval_policy(args.profile)))
    render_calibration_report(
        points,
        args.out,
        title=f"Calibration curve ({args.profile} profile, {len(SCENARIOS)} scenarios)",
        ablation_results=ablation,
    )
    print(f"wrote {args.out / 'report.md'} and {args.out / 'calibration_curve.png'}")


if __name__ == "__main__":
    main()
