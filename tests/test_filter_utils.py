# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Unit tests for metadata filter utilities and operators."""

from DeepImageSearch.core.filter_utils import apply_night_color_tolerance, matches_filters


def test_matches_filters_equality():
    meta = {"state": "stationary", "ship_color": "red"}
    assert matches_filters(meta, {"state": "stationary"})
    assert not matches_filters(meta, {"state": "moving"})
    assert not matches_filters(meta, {"unknown_key": "val"})


def test_matches_filters_range_numeric():
    meta = {"duration_sec": 2100}
    assert matches_filters(meta, {"duration_sec": {"$gte": 1200}})
    assert matches_filters(meta, {"duration_sec": {"$lte": 2500}})
    assert matches_filters(meta, {"duration_sec": {"$gte": 1200, "$lte": 2100}})
    assert not matches_filters(meta, {"duration_sec": {"$gt": 2100}})
    assert matches_filters(meta, {"duration_sec": {"$gt": 2000}})
    assert not matches_filters(meta, {"duration_sec": {"$lt": 2000}})


def test_matches_filters_range_datetime():
    meta = {"start_time": "2026-10-22 21:15:00"}
    assert matches_filters(
        meta,
        {"start_time": {"$gte": "2026-10-22 20:00:00", "$lte": "2026-10-22 22:00:00"}},
    )
    assert not matches_filters(
        meta,
        {"start_time": {"$gte": "2026-10-23 00:00:00"}},
    )
    assert matches_filters(
        meta,
        {"start_time": {"$lte": "2026-10-22 21:15:00"}},
    )


def test_matches_filters_in_and_nin():
    meta = {"ship_size": "small", "camera_location": "东港区3号货运码头", "ship_type": "fishing_boat"}
    assert matches_filters(meta, {"ship_size": {"$in": ["small", "medium"]}})
    assert not matches_filters(meta, {"ship_size": {"$in": ["large", "huge"]}})
    assert matches_filters(meta, {"ship_size": ["small", "medium"]})  # implicit list
    assert matches_filters(meta, {"camera_location": {"$in": ["东港区3号货运码头", "西港区"]}})
    assert matches_filters(meta, {"ship_type": {"$nin": ["tugboat", "cargo_ship"]}})
    assert not matches_filters(meta, {"ship_type": {"$nin": ["fishing_boat"]}})


def test_matches_filters_logical_or_and_and():
    meta_red = {"ship_color": "red", "color_confidence": "high", "state": "stationary"}
    meta_ir_night = {"ship_color": "unknown", "color_confidence": "low", "state": "stationary"}
    meta_blue = {"ship_color": "blue", "color_confidence": "high", "state": "moving"}

    or_cond = {"$or": [{"ship_color": "red"}, {"color_confidence": "low"}]}
    assert matches_filters(meta_red, or_cond)
    assert matches_filters(meta_ir_night, or_cond)
    assert not matches_filters(meta_blue, or_cond)

    and_cond = {
        "$and": [
            {"$or": [{"ship_color": "red"}, {"color_confidence": "low"}]},
            {"state": "stationary"},
        ]
    }
    assert matches_filters(meta_red, and_cond)
    assert matches_filters(meta_ir_night, and_cond)
    assert not matches_filters(meta_blue, and_cond)


def test_matches_filters_ne():
    meta = {"state": "stationary"}
    assert matches_filters(meta, {"state": {"$ne": "moving"}})
    assert not matches_filters(meta, {"state": {"$ne": "stationary"}})


def test_apply_night_color_tolerance():
    # 1. Transforms ship_color into $or
    f1 = {"ship_color": "red"}
    t1 = apply_night_color_tolerance(f1)
    assert t1 == {"$or": [{"ship_color": "red"}, {"color_confidence": "low"}]}

    # 2. Combined with numeric range
    f2 = {"ship_color": "red", "duration_sec": {"$gte": 1200}}
    t2 = apply_night_color_tolerance(f2)
    assert t2["duration_sec"] == {"$gte": 1200}
    assert t2["$or"] == [{"ship_color": "red"}, {"color_confidence": "low"}]

    # 3. No ship_color present
    f3 = {"duration_sec": {"$gte": 1200}}
    t3 = apply_night_color_tolerance(f3)
    assert t3 == f3
