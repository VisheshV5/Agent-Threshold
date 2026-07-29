"""Tests for the decision cache, written before the implementation.

Pins the API surface:
- DecisionCache(path=None) -- in-memory by default, only touches disk
  with an explicit path (same opt-in pattern as DecisionLogger/NoveltyStore)
- .get(action) -> "proceed" | "abort" | None
- .record(action, resolution) -- persists if a path was given
- Keyed on (tool_name, arguments) only -- NOT agent_reasoning, matching
  self-consistency's stance that reasoning doesn't define the situation
- Deliberately EXACT-match, not fuzzy/semantic: a real embedding-based
  fuzzy match risks a false-positive cache hit auto-resolving a
  similar-but-different situation without human review -- the exact
  risk flagged when caching was first discussed. This is the
  conservative M4 stand-in, same pattern as every other "real semantic
  operation" in this project (self-consistency, novelty) having a
  heuristic placeholder.
"""

from threshold.decision_cache import DecisionCache
from threshold.types import ProposedAction


def make_action(tool_name: str, arguments: dict, reasoning: str = "test") -> ProposedAction:
    return ProposedAction(tool_name=tool_name, arguments=arguments, agent_reasoning=reasoning)


def test_get_returns_none_for_an_unrecorded_situation():
    cache = DecisionCache()
    action = make_action("update_customer_tier", {"customer_id": "C1", "new_tier": "gold"})
    assert cache.get(action) is None


def test_record_then_get_returns_the_resolution():
    cache = DecisionCache()
    action = make_action("update_customer_tier", {"customer_id": "C1", "new_tier": "gold"})
    cache.record(action, "proceed")
    assert cache.get(action) == "proceed"


def test_matching_is_exact_not_fuzzy():
    cache = DecisionCache()
    approved = make_action("update_customer_tier", {"customer_id": "C1", "new_tier": "gold"})
    similar_but_different = make_action("update_customer_tier", {"customer_id": "C2", "new_tier": "gold"})
    cache.record(approved, "proceed")
    assert cache.get(similar_but_different) is None  # must NOT auto-resolve a different customer


def test_agent_reasoning_does_not_affect_the_cache_key():
    cache = DecisionCache()
    action_a = make_action("read_file", {"path": "/x"}, reasoning="need to check config")
    action_b = make_action("read_file", {"path": "/x"}, reasoning="completely different justification")
    cache.record(action_a, "proceed")
    assert cache.get(action_b) == "proceed"


def test_no_path_does_not_touch_disk(tmp_path):
    cache = DecisionCache()
    cache.record(make_action("read_file", {"path": "/x"}), "proceed")
    assert list(tmp_path.iterdir()) == []


def test_persists_and_reloads_from_path(tmp_path):
    path = tmp_path / "resolutions.jsonl"
    cache1 = DecisionCache(path=path)
    action = make_action("update_customer_tier", {"customer_id": "C1", "new_tier": "gold"})
    cache1.record(action, "abort")

    cache2 = DecisionCache(path=path)  # fresh instance, same file
    assert cache2.get(action) == "abort"


def test_re_recording_a_situation_overwrites_the_prior_resolution():
    cache = DecisionCache()
    action = make_action("read_file", {"path": "/x"})
    cache.record(action, "proceed")
    cache.record(action, "abort")
    assert cache.get(action) == "abort"
