"""Builds the Policy used to evaluate the scenario suite.

Novelty needs a populated store to ever produce a real (non-fallback)
score, but Policy.from_profile() with no arguments always builds a
fresh, empty NoveltyStore -- meaning novelty would sit at its 0.5
missing-data fallback for every scenario forever, regardless of how
many scenarios exist. SEED_TRAJECTORIES represents "things this
hypothetical agent has done successfully before": the trajectory shape
of every scenario in the original 13 (pre-dating novelty's addition),
so scenarios reusing those same tool names read as unsurprising, while
scenarios built specifically to demonstrate novelty use trajectory
shapes that don't match anything here.
"""

from escalation.policy.policy import Policy
from escalation.signals.novelty import NoveltySignal, NoveltyStore

SEED_TRAJECTORIES: list[list[str]] = [
    ["read_file"],
    ["read_file", "write_file"],
    ["search_knowledge_base"],
    ["list_dir", "delete_file"],
    ["send_email"],
    ["execute_trade"],
    ["archive_old_project_files"],
    ["update_draft_pull_request"],
    ["search_knowledge_base", "delete_file"],
    ["update_shared_records"],
    ["update_customer_tier"],
    ["check_system_status"],
]


def build_seeded_novelty_store() -> NoveltyStore:
    store = NoveltyStore()
    for trajectory in SEED_TRAJECTORIES:
        store.record_success(trajectory)
    return store


def build_eval_policy(profile: str = "balanced") -> Policy:
    return Policy.from_profile(profile, novelty=NoveltySignal(store=build_seeded_novelty_store()))
