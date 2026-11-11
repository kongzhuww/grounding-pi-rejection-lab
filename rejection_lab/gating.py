"""Explicit existence-gate routing, not repairs of malformed localizer output."""

import unicodedata

from .data import answer_for


def route_prediction(query, baseline_prediction, present):
    if type(present) is not bool or query["id"] != baseline_prediction["id"]:
        raise ValueError("one boolean head decision and matching localizer ID required")
    if present:
        # In particular, do not turn an invalid localizer response into a valid one.
        return {**baseline_prediction, "gate_present": True, "origin": "frozen_localizer"}
    phrase = query["phrase"]
    if (
        not isinstance(phrase, str)
        or not phrase
        or unicodedata.normalize("NFC", phrase) != phrase
        or "<|" in phrase
        or "|>" in phrase
        or any(unicodedata.category(c) in ("Cc", "Cf") for c in phrase)
    ):
        raise ValueError("unsafe or noncanonical query label")
    # This is a classifier decision rendered as GAM, NOT a model-generated answer.
    # No ground-truth presence, coordinates, answer, or localizer validity is consulted.
    return {
        "id": query["id"],
        "valid": True,
        "coordinates": None,
        "raw_output": answer_for(phrase, None),
        "finish_reason": "gate_constructed",
        "gate_present": False,
        "origin": "existence_gate",
    }


def presence_counts(rows, choices):
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or set(choices) != set(ids):
        raise ValueError("presence predictions must match every unique row ID")
    if any(type(x) is not bool for x in choices.values()):
        raise ValueError("presence decisions must be booleans")
    return {
        "true_positive": sum(r["present"] and choices[r["id"]] for r in rows),
        "true_negative": sum(not r["present"] and not choices[r["id"]] for r in rows),
        "false_positive": sum(not r["present"] and choices[r["id"]] for r in rows),
        "false_negative": sum(r["present"] and not choices[r["id"]] for r in rows),
    }
