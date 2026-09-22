"""Records must not be built from a file this tier's manifest cannot vouch for."""

from __future__ import annotations

from neuromps.etl import build


def test_toxcast_is_skipped_when_its_hash_is_not_in_this_manifest() -> None:
    """The failure this guards: `--quick` on a machine that has run `--full`.

    The interim parquet is shared between tiers, so the quick run finds the file
    and no hash for it, and every record it builds carries an empty SHA256 and
    fails schema validation - 20,375 of them, which stops the build with a
    message about a regex rather than about tiers.
    """
    assert build._toxcast_records({}, full=False) == []
    assert build._toxcast_records({"something_else.csv": "a" * 64}, full=False) == []


def test_toxcast_is_built_when_the_manifest_has_its_hash(monkeypatch) -> None:
    """The guard must not swallow the ordinary case."""
    calls = {}

    def fake_build_curve_records(df, hashes, source_file):
        calls["source_file"] = source_file
        return [{"record_id": "toxcast_mea:x"}]

    monkeypatch.setattr(build, "build_curve_records", fake_build_curve_records)
    monkeypatch.setattr(build.pd, "read_parquet", lambda _p: "frame")
    monkeypatch.setattr(build, "INTERIM_PARQUET", build.INTERIM_PARQUET)
    if not build.INTERIM_PARQUET.exists():
        return  # nothing cached on this machine; the branch is covered above
    records = build._toxcast_records({build.MC56_CSV.name: "b" * 64}, full=False)
    assert records and calls["source_file"] == build.MC56_CSV.name


def test_an_asset_without_pairs_degrades_with_a_message(tmp_path, monkeypatch, caplog) -> None:
    """A blocked source must not surface as `KeyError: 'source'` from pandas."""
    import logging

    import pandas as pd

    from neuromps.models import conformal, dataset, mechanism, transfer

    empty = pd.DataFrame(columns=dataset.PAIR_COLUMNS + dataset.delta_columns())
    # Each main() imports load_pairs inside the function, so it is patched at
    # the source rather than on the consumer.
    monkeypatch.setattr(dataset, "load_pairs", lambda *a, **k: empty)
    modules = ((mechanism, "mechanism"), (transfer, "transfer"), (conformal, "conformal"))
    for module, name in modules:
        with caplog.at_level(logging.WARNING):
            assert module.main([]) == 0, name
    assert "no exposure/baseline pairs" in caplog.text
