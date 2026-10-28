"""Internal timestamps: the hash chain depends on content, commit and order."""

from __future__ import annotations

from neuromps.report import timestamps as ts


def test_chain_changes_with_content_commit_and_order() -> None:
    a = {"sha256": "aa", "commit": "c1"}
    b = {"sha256": "bb", "commit": "c2"}
    base = ts.chain([a, b])
    assert len(base) == 2 and base[0] != base[1]
    assert ts.chain([{**a, "sha256": "ab"}, b])[-1] != base[-1]
    assert ts.chain([a, {**b, "commit": "c3"}])[-1] != base[-1]
    assert ts.chain([b, a])[-1] != base[-1]
    assert ts.chain([a, b]) == base


def test_summary_counts_checks_violations_and_uncovered_registrations() -> None:
    regs = [{"file": "a.json"}, {"file": "b.json"}, {"file": "c.json"}]
    checks = [
        {"registration": "a.json", "registered_first": True},
        {"registration": "a.json", "registered_first": True},
        {"registration": "b.json", "registered_first": False},
    ]
    got = ts.summary(regs, checks)
    assert got == {
        "n_registrations": 3,
        "n_outputs_checked": 3,
        "n_violations": 1,
        "not_covered": ["c.json"],
    }
