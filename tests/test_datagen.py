"""Phase 1: generator determinism, integrity and anomaly rate."""

from __future__ import annotations

import pandas as pd

from supplysight.data import po_items, validate, weekly_consumption
from supplysight.datagen.generator import (
    ANOMALY_TYPES,
    GenConfig,
    generate,
    is_valid_gstin,
    season_factor,
)


def test_referential_integrity(small_tables):
    assert validate(small_tables) == []


def test_same_seed_same_output(small_cfg, small_tables):
    again = generate(small_cfg)
    for name in ("ekpo", "lfa1", "mseg", "anomaly_labels"):
        pd.testing.assert_frame_equal(small_tables[name], again[name])


def test_different_seed_differs(small_cfg, small_tables):
    other = generate(GenConfig(**{**small_cfg.__dict__, "seed": 8}))
    assert not small_tables["ekpo"].netpr.head(50).equals(other["ekpo"].netpr.head(50))


def test_anomaly_rate_in_range(small_tables):
    rate = len(small_tables["anomaly_labels"]) / len(small_tables["ekpo"])
    assert 0.025 <= rate <= 0.06
    assert set(small_tables["anomaly_labels"].anomaly_type) <= set(ANOMALY_TYPES)


def test_gstins_are_format_valid(small_tables):
    assert small_tables["lfa1"].stcd3.map(is_valid_gstin).all()


def test_consumption_derived_from_bom(small_tables):
    # every material issued to production must be a BOM component
    issued = set(small_tables["mseg"].query("bwart == '261'").matnr)
    assert issued <= set(small_tables["stpo"].idnrk)


def test_seasonality_profiles():
    from datetime import date

    assert season_factor("summer", date(2025, 5, 10)) > 1.5
    assert season_factor("festive", date(2025, 10, 10)) > season_factor("festive", date(2025, 6, 10))
    assert season_factor("flat", date(2025, 6, 10)) == 1.0


def test_po_items_and_weekly_consumption(small_tables):
    items = po_items(small_tables)
    assert len(items) == len(small_tables["ekpo"])
    wc = weekly_consumption(small_tables["mseg"])
    counts = wc.groupby(["matnr", "werks"]).size()
    assert counts.nunique() == 1  # dense weekly grid
    assert (wc.qty >= 0).all()
