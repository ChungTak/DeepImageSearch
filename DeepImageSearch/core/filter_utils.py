# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Metadata filtering utilities supporting logical and range operators."""

import copy
from datetime import datetime
from typing import Any, Dict, Optional


def _to_comparable(val: Any) -> Any:
    """Convert string/numeric values to comparable numbers or datetimes when possible."""
    if isinstance(val, (int, float, datetime)):
        return val
    if isinstance(val, str):
        # 1. Try numeric conversion
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            pass

        # 2. Try ISO datetime conversion (handling space or T separator)
        clean = val.strip().replace(" ", "T")
        try:
            return datetime.fromisoformat(clean)
        except (ValueError, TypeError):
            pass

        return val
    return val


def _compare(a: Any, b: Any, op: str) -> bool:
    """Compare two values with operator op ('>=', '<=', '>', '<')."""
    ca = _to_comparable(a)
    cb = _to_comparable(b)

    try:
        if op == ">=":
            return bool(ca >= cb)
        elif op == "<=":
            return bool(ca <= cb)
        elif op == ">":
            return bool(ca > cb)
        elif op == "<":
            return bool(ca < cb)
    except TypeError:
        # Fallback to string comparison if types are fundamentally mismatched
        try:
            sa, sb = str(a), str(b)
            if op == ">=":
                return sa >= sb
            elif op == "<=":
                return sa <= sb
            elif op == ">":
                return sa > sb
            elif op == "<":
                return sa < sb
        except Exception:
            return False
    return False


def matches_filters(meta: Optional[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> bool:
    """
    Check if a metadata dictionary matches the given filter specification.

    Supports:
      - Equality: {"state": "stationary"}
      - Implicit list membership: {"ship_size": ["small", "medium"]}
      - Comparison operators: {"duration_sec": {"$gte": 1200, "$lte": 3600}}
      - Set membership: {"ship_size": {"$in": ["small", "medium"]}}
      - Negated membership: {"ship_type": {"$nin": ["tugboat"]}}
      - Inequality: {"state": {"$ne": "moving"}}
      - Logical OR: {"$or": [{"ship_color": "red"}, {"color_confidence": "low"}]}
      - Logical AND: {"$and": [...]}
    """
    if not filters:
        return True
    if not isinstance(meta, dict):
        return False

    for key, cond in filters.items():
        if key == "$or":
            if not isinstance(cond, list) or not cond:
                return False
            if not any(matches_filters(meta, sub) for sub in cond):
                return False
            continue

        if key == "$and":
            if not isinstance(cond, list) or not cond:
                return False
            if not all(matches_filters(meta, sub) for sub in cond):
                return False
            continue

        # Target field
        if key not in meta:
            # Field does not exist in metadata
            if isinstance(cond, dict) and "$ne" in cond:
                continue
            if isinstance(cond, dict) and "$exists" in cond and not cond["$exists"]:
                continue
            return False

        meta_val = meta[key]

        if isinstance(cond, dict):
            for op, op_val in cond.items():
                if op == "$gte":
                    if not _compare(meta_val, op_val, ">="):
                        return False
                elif op == "$lte":
                    if not _compare(meta_val, op_val, "<="):
                        return False
                elif op == "$gt":
                    if not _compare(meta_val, op_val, ">"):
                        return False
                elif op == "$lt":
                    if not _compare(meta_val, op_val, "<"):
                        return False
                elif op == "$eq":
                    if meta_val != op_val:
                        return False
                elif op == "$ne":
                    if meta_val == op_val:
                        return False
                elif op == "$in":
                    if not isinstance(op_val, (list, tuple, set)):
                        return False
                    if isinstance(meta_val, (list, tuple, set)):
                        if not any(item in op_val for item in meta_val):
                            return False
                    else:
                        if meta_val not in op_val:
                            return False
                elif op == "$nin":
                    if not isinstance(op_val, (list, tuple, set)):
                        return False
                    if isinstance(meta_val, (list, tuple, set)):
                        if any(item in op_val for item in meta_val):
                            return False
                    else:
                        if meta_val in op_val:
                            return False
                elif op == "$exists":
                    if not bool(op_val):
                        return False
                else:
                    # Unrecognized op, treat as exact match
                    if meta_val != cond:
                        return False
        elif isinstance(cond, list):
            # Implicit $in (backwards-compatible with original DeepImageSearch)
            if isinstance(meta_val, (list, tuple, set)):
                if not any(item in cond for item in meta_val):
                    return False
            else:
                if meta_val not in cond:
                    return False
        else:
            # Scalar equality
            if meta_val != cond:
                return False

    return True


def apply_night_color_tolerance(
    filters: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Transform filters to enable night color tolerance.

    If 'ship_color' is specified in the filter, transforms it into an $or clause:
      {"$or": [{"ship_color": <color_cond>}, {"color_confidence": "low"}]}
    so that infrared/night detections with low color confidence are not excluded.
    """
    if not filters or not isinstance(filters, dict):
        return filters

    if "ship_color" not in filters:
        return copy.deepcopy(filters)

    new_filters = copy.deepcopy(filters)
    color_val = new_filters.pop("ship_color")

    color_or_clause: Dict[str, Any] = {
        "$or": [
            {"ship_color": color_val},
            {"color_confidence": "low"},
        ]
    }

    if "$and" in new_filters and isinstance(new_filters["$and"], list):
        new_filters["$and"].append(color_or_clause)
    elif "$or" in new_filters and isinstance(new_filters["$or"], list):
        existing_or = {"$or": new_filters.pop("$or")}
        new_filters["$and"] = [existing_or, color_or_clause]
    else:
        new_filters["$or"] = color_or_clause["$or"]

    return new_filters
