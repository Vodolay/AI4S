"""The asymmetric cross-laboratory test: only a low score means anything."""

import pandas as pd

from neuromps.models.cross_lab import (
    CHANCE_IS_MEANINGFUL_BELOW,
    shared_classes,
    summarise_cross_lab,
    train_test_split_by_source,
)


def _pairs() -> pd.DataFrame:
    """Four sources; only three classes appear in more than one of them."""
    rows = []
    for source, label, n in (
        ("kapucu2022", "gabaa_antagonist", 10),
        ("kapucu2022", "gabaa_agonist_or_pam", 4),
        ("kapucu2022", "glutamatergic_agonist", 6),
        ("kapucu2022", "na_channel_blocker", 26),
        ("screenneuropharm_bic", "gabaa_antagonist", 497),
        ("sharf2022", "gabaa_agonist_or_pam", 13),
        ("vassallo2017", "glutamatergic_agonist", 25),
    ):
        rows += [{"source": source, "label": label, "group": f"{source}-{i}"} for i in range(n)]
    return pd.DataFrame(rows)


def test_shared_classes_are_those_in_more_than_one_source() -> None:
    shared = shared_classes(_pairs())
    assert sorted(shared) == [
        "gabaa_agonist_or_pam",
        "gabaa_antagonist",
        "glutamatergic_agonist",
    ]
    assert "na_channel_blocker" not in shared, "a single-source class cannot be tested across labs"


def test_the_split_trains_on_one_source_and_tests_on_the_rest() -> None:
    train, test = train_test_split_by_source(_pairs(), train_source="kapucu2022")
    assert set(train["source"]) == {"kapucu2022"}
    assert "kapucu2022" not in set(test["source"])
    assert set(train["label"]) == set(test["label"]), "both sides must share the class set"
    assert len(train) == 20 and len(test) == 535


def test_the_summary_records_that_the_confound_helps_the_model() -> None:
    """In the test set each class comes from one laboratory, so laboratory
    identity alone scores perfectly. A high accuracy is therefore uninterpretable
    and only a low one carries information - the summary has to say so itself."""
    train, test = train_test_split_by_source(_pairs(), train_source="kapucu2022")
    out = summarise_cross_lab(train, test, balanced_accuracy=0.31, ci95=[0.2, 0.42])
    assert out["label_source_correspondence"] == 1.0
    assert out["confound_ceiling"] == 1.0
    assert out["interpretable"] is True  # 0.31 is low, so the failure is informative
    assert "only a low score" in out["reads_as"].lower()

    high = summarise_cross_lab(train, test, balanced_accuracy=0.95, ci95=[0.9, 0.99])
    assert high["interpretable"] is False
    assert CHANCE_IS_MEANINGFUL_BELOW < 0.95


def test_the_control_refuses_a_split_it_cannot_make() -> None:
    """With one pair in the smallest class there is no grouped split to run."""
    from neuromps.models.cross_lab import within_source_control

    thin = pd.DataFrame(
        {
            "label": ["a", "a", "b"],
            "group": ["g1", "g2", "g3"],
            "source": ["kapucu2022"] * 3,
        }
    )
    assert "status" in within_source_control(thin)


def test_max_shared_between_pairs_is_the_reason_pooling_is_refused() -> None:
    """Below two shared classes, no cross-laboratory classifier exists at all."""
    from neuromps.models.cross_lab import max_shared_between_pairs

    assert max_shared_between_pairs(_pairs()) == 1
    both = pd.DataFrame(
        {
            "label": ["a", "b", "a", "b"],
            "source": ["one", "one", "two", "two"],
            "group": list("wxyz"),
        }
    )
    assert max_shared_between_pairs(both) == 2
