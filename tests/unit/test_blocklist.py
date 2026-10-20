import os

from geometry_to_physics.provenance.blocklist import (
    assert_allowed_source,
    classify_placement_file,
    is_blocked_historical_output,
    placement_data_path,
)


def test_blocklist_rejects_historical_outputs():
    assert is_blocked_historical_output("outputs/tcsvt_reference_anchor/properties/object_property_pairs.csv")
    assert not is_blocked_historical_output("metadata.json")


def test_blocklist_rejects_historical_placement_outside_canonical(tmp_path, monkeypatch):
    monkeypatch.delenv("GTP_CANONICAL_ROOT", raising=False)
    p = tmp_path / "placement_data.json"
    p.write_text("{}", encoding="utf-8")
    assert classify_placement_file(p) == "UNVERIFIED"
    try:
        assert_allowed_source(p)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_placement_allowed_under_canonical_root(tmp_path, monkeypatch):
    canon = tmp_path / "canonical"
    pl = canon / "DUSt3R" / "ycbv_000048_spread_scaled" / "placement_data.json"
    pl.parent.mkdir(parents=True)
    pl.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GTP_CANONICAL_ROOT", str(canon))
    assert classify_placement_file(pl) == "ALLOWED_REQUIRED"
    assert_allowed_source(pl)
    assert placement_data_path("DUSt3R", "ycbv_000048_spread_scaled") == pl
