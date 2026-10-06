# Threshold

**A calibrated escalation layer for AI agents.** Threshold sits between an agent and its tools, and before every tool call it decides one of three things: **proceed**, **ask a human**, or **abort**. That decision is measurable and tunable, not vibes-based.

From a live run, where a Claude-backed agent is cleaning up a directory:

```
  agent proposes: delete_file({'path': 'cache_12345.tmp'})
    reversibility      score=0.75  'delete_file' classified as irreversible-write via registry
    blast_radius       score=0.00  parsed 1 entities from arguments -> blast radius 0.0
    self_consistency   score=0.67  2 alternative sample(s), avg pairwise agreement 0.33
    novelty            score=0.50  novelty store is empty; nothing to compare against
    staleness          score=0.00  no tracked facts depend on prior context
    thrash             score=0.00  no prior consecutive failures of 'delete_file'
  aggregate_score=0.394  verdict=ask_human  [HARD OVERRIDE]

  >> About to call 'delete_file'.
  >> options: ['proceed', 'abort'] (recommended: abort)
```

Even for an obviously disposable cache file, an irreversible action goes to a human. Self-consistency also caught that the agent's own resamples disagreed about what to do next.

The thesis: agents aren't blocked from consequential work by capability. They're blocked because **they don't know when they're out of their depth.** Threshold makes "should a human look at this?" a number you can calibrate, and gives you a curve showing the tradeoff between unnecessary interruptions and missed escalations.

![Calibration curve](reports/m3/calibration_curve.png)

---

## How it works

```mermaid
flowchart LR
    A[Agent proposes<br/>a tool call] --> B[ProposedAction]
    B --> S[Six independent signals<br/>each scores 0.0 to 1.0 with a reason]
    S --> P{Policy}
    P -- "irreversible or<br/>external-effect" --> H[ask_human<br/>hard override]
    P -- "weighted score ≥ threshold" --> H
    P -- "a human already answered<br/>this exact situation" --> C[cached answer]
    P -- otherwise --> G[proceed]
    H --> Q[Structured question:<br/>options + recommended default]
    Q --> R[Human's answer is cached<br/>except for hard-override actions]
```

Every decision, including every signal's score whether or not it changed the verdict, is appended to a JSONL log for later calibration analysis.

### The six signals

| Signal | What it measures | How |
|---|---|---|
| **Reversibility** | Can the effect be undone? (read-only / reversible-write / irreversible-write / external-effect) | Static registry for known tools; keyword heuristic or LLM classifier for unknown ones. Fails safe to irreversible. |
| **Self-consistency** | Is the agent guessing? | The agent is resampled at temperature for the same decision point; low agreement between samples means low confidence. |
| **Blast radius** | How many records, people, or dollars does this touch? | Parsed from arguments (list lengths, `quantity`/`count` keys); glob patterns are treated as unknown, not as one item. |
| **Novelty** | Has this kind of trajectory ever succeeded before? | Nearest-neighbor similarity against a persistent store of past successful runs. |
| **Staleness** | Does the action depend on a fact checked long ago? | Steps since each fact was last verified. |
| **Thrash** | Is the agent looping on the same failure? | Consecutive trailing failures of the same tool. |

Each signal is independently testable, independently disableable (ablation zeroes its weight), and returns a human-readable reason that surfaces in the escalation question.

### The policy

- **Weighted aggregate against a threshold**, with three presets: `conservative` (0.05), `balanced` (0.25), `autonomous` (0.6).
- **Hard override:** irreversible-write and external-effect actions *always* go to a human, regardless of score, profile, or how confident the agent sounds. The override changes the verdict, never the reported score, so threshold-driven and override-driven escalations stay distinguishable in the metrics.
- **"I don't know" is not a score.** When a signal genuinely has nothing to assess (self-consistency with no resamples, novelty with an empty store), it's marked `informative=False` and excluded from the weighted average instead of being averaged in as a fake 0.5 that silently drags confident signals toward the middle.
- **Decision caching:** once a human answers, the same situation is resolved automatically next time. Hard-override actions never read from or write to the cache. Irreversible actions always get a fresh human look.

---

## Results

The evaluation harness runs 40 hand-written scenarios through the policy: 10 clearly safe, 9 clearly dangerous, 13 genuinely ambiguous, and 8 adversarial cases where the agent is confidently wrong.

### Threshold profiles

| Profile | Threshold | Accuracy | False alarm rate | Miss rate |
|---|---|---|---|---|
| conservative | 0.05 | 72.5% | 73.3% | 0% |
| **balanced** | **0.25** | **100%** | **0%** | **0%** |
| autonomous | 0.60 | 72.5% | 0% | 44% |

This is the tradeoff the project exists to expose: `conservative` never misses but interrupts constantly; `autonomous` never interrupts unnecessarily but misses everything that isn't hard-override protected. Note that `autonomous` still catches every irreversible and external-effect action, because hard override doesn't care about the threshold.

### Which signals carry the decision (ablation)

Each signal is removed in turn, and the change in verdicts is measured. Removing reversibility also disables hard override, since otherwise its ablation would understate its impact.

| Signal removed | Verdicts changed | Accuracy without it |
|---|---|---|
| reversibility | 11 | 72.5% |
| staleness | 4 | 90.0% |
| self_consistency | 3 | 92.5% |
| blast_radius | 2 | 95.0% |
| thrash | 2 | 95.0% |
| novelty | 2 | 95.0% |

Every signal carries real weight. On an earlier 13-scenario version of the suite, the same study showed novelty, staleness, and thrash contributing exactly zero. That finding is what drove the expansion to 40 scenarios with dedicated cases for each.

### What these numbers do and don't show

These results are honest but limited, and worth reading in that light:

- **The 100% at `balanced` is in-sample.** The scenarios were hand-written for this project, and the weights and thresholds were tuned against the same suite. That number is not a claim about generalization. The value here is the methodology: the calibration curve, the hard-override and threshold-sensitive split, and the ablation.
- **The benchmark uses the heuristic signal implementations** (keyword classifier, structural similarity) so that it's deterministic and free to run. The LLM-backed implementations are exercised in the live demo but aren't benchmarked.
- **The balanced sweet spot is narrow.** With zero false alarms and zero misses, it sits between thresholds 0.237 and 0.299. That's visible in the curve, not hidden.

---

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/VisheshV5/Agent-Threshold.git
cd Agent-Threshold
uv sync
uv run pytest            # 210 tests, no API key needed
```

### Use it as a library

```python
import asyncio
from threshold.policy.policy import Policy
from threshold.types import ProposedAction

policy = Policy.from_profile("balanced")

async def main():
    decision = await policy.evaluate(ProposedAction(
        tool_name="delete_file",
        arguments={"path": "customer_records.db"},
        agent_reasoning="Looks like a stale backup.",
    ))
    print(decision.verdict)                  # ask_human
    print(decision.hard_override_triggered)  # True
    print(decision.question.options)         # ['proceed', 'abort']

asyncio.run(main())
```

`Policy.from_profile()` uses the heuristic signal implementations by default. Every signal takes its estimator or classifier by dependency injection, so the LLM-backed versions in `threshold.adapters` drop in without code changes elsewhere. You can also pass a `DecisionLogger` (JSONL audit log) and a `DecisionCache`.

### Reproduce the evaluation

```bash
uv run python -m threshold.eval                          # writes reports/latest/
uv run python -m threshold.eval --profile conservative
```

This runs the threshold sweep and the ablation study, then writes a markdown report and the calibration chart. It makes no network calls.

### Run the live agent demo

A Claude-backed agent is asked to clean up a sandboxed directory that contains obvious junk (`*.tmp` files) next to one file that reads as important (`quarterly_report.docx`). Every proposed action goes through the full policy with LLM-backed signals, and you answer the escalation prompts yourself.

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # gitignored
uv run python -m demo.cleanup_agent           # --profile, --max-turns also available
```

> **Cost note:** this makes real, billed API calls, roughly 6-7 per agent turn. That's one primary proposal, two resamples for self-consistency, three pairwise similarity judgments, and occasional classifier calls. Each proposal call resends the full conversation history.

The demo keeps learned state (the novelty store and decision cache) in `demo/state/` across runs, and per-run artifacts (sandbox, decision log) in `demo/run/`.

---

## Project layout

```
src/threshold/
├── types.py              # ProposedAction, SignalResult, Decision, HumanQuestion
├── signals/              # the six signals, each with a pluggable heuristic fallback
├── policy/               # weighted aggregation, hard override, profiles
├── decision_cache.py     # human resolutions, exact-match, never for hard-override actions
├── decision_log.py       # JSONL log of every decision and every signal score
├── adapters/             # Anthropic client, agent (with resampling), LLM-backed classifiers
└── eval/                 # scenario format, 40 scenarios, metrics, calibration sweep, ablation
demo/                     # live end-to-end agent demo
reports/                  # committed calibration reports (M2, M3)
tests/                    # 210 tests
```

---

## Engineering notes

The project was built test-first in four milestones, with small commits throughout. Several of these bugs were caught by tests or ablation before they could skew the results, and a couple only surfaced against the live API:

- **Substring trap in category parsing.** `"irreversible-write"` contains `"reversible-write"` as a substring, so naive matching of an LLM's response would downgrade every irreversible action. Matching runs longest-category-first.
- **Severity-ordered keyword matching.** A tool named `edit_and_delete_entry` originally classified as reversible because `edit` was checked before `delete`. Keywords are now checked most-severe-first, so a milder word can never mask a destructive one.
- **Fake confidence from missing data.** Averaging "can't assess" as 0.5 skewed aggregates in both directions, depending on whether 0.5 sat above or below the real signals. It was replaced with explicit `informative=False` exclusion.
- **Ablation that lied.** The first ablation implementation rebuilt fresh signal instances, so novelty (which is stateful) was always compared against an empty store and could never show impact. It now reuses the caller's configured instances. Ablating reversibility also neutralizes hard override, not just its weight.
- **Silent cost doubling.** The policy called the reversibility classifier twice per decision. That's harmless with a free heuristic, but it doubles the bill with an LLM classifier. A regression test now pins it to one call.
- **Live API surprise.** The first real run returned `400: temperature is deprecated for this model`. A narrow retry now drops `temperature` only for that specific rejection and never masks other errors.

## Limitations and next steps

- The decision cache uses exact structural matching. A semantic match would hit more often, but it risks auto-resolving a *similar-but-different* situation without review, so it's deliberately not done yet.
- The novelty signal needs history. It contributes nothing until a few clean runs have been recorded.
- The cost per agent turn is high (see the cost note above). Obvious next steps are a cheaper model for the mechanical signal calls, comparing resamples only against the primary rather than all pairs, and prompt caching.
- Scenarios are hand-written. A held-out set and real agent traces would be the next step toward numbers that mean something outside this repo.
