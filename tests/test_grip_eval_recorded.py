"""Targeted tests for the grip-limit experiment viewer source (additive loader param)."""
import pytest
from inference.recorded import load_bic_recorded


def test_default_bic_unchanged():
    m = load_bic_recorded()["meta"]
    assert m["label"] == "Recorded: BIC" and m["steps"] == 7521 and m["run"] is None


def test_grip_run_loads_with_label():
    d = load_bic_recorded("bic_g1p0")
    assert d["meta"]["label"].startswith("EXPERIMENT — grip limit 1.0 G")
    assert d["meta"]["grip"]["limit_g"] == 1.0
    assert d["cols"]["crashed"][-1] is True


def test_unknown_run_rejected():
    with pytest.raises(KeyError):
        load_bic_recorded("bic_g9p9")
