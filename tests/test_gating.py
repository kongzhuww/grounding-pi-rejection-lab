import pytest

from rejection_lab.gating import presence_counts, route_prediction
from rejection_lab.metrics import grouped_score


def test_present_keeps_invalid_localizer_response_unchanged():
    query = {"id": "a", "phrase": "red square"}
    baseline = {"id": "a", "valid": False, "coordinates": None, "raw_output": "broken"}
    result = route_prediction(query, baseline, True)
    assert all(result[k] == v for k, v in baseline.items())
    assert result["origin"] == "frozen_localizer"


def test_absent_uses_classifier_not_ground_truth_or_localizer_validity():
    query = {"id": "a", "phrase": "red square", "present": True, "answer": "poison"}
    baseline = {"id": "a", "valid": False, "coordinates": None, "raw_output": "broken"}
    result = route_prediction(query, baseline, False)
    assert result["coordinates"] is None and result["valid"]
    assert (
        result["raw_output"]
        == "<|object_ref_start|>red square<|object_ref_end|><|box_start|>None<|box_end|>"
    )
    assert result["finish_reason"] == "gate_constructed"
    assert result["origin"] == "existence_gate"
    query.update(present=False, answer="different")
    assert (
        route_prediction(query, {**baseline, "valid": True, "coordinates": [1, 2]}, False) == result
    )


def test_always_absent_gate_fails_positive_localization():
    rows = [
        {
            "id": "p",
            "phrase": "red square",
            "present": True,
            "task": "point",
            "target_coordinates": [100, 100],
            "width": 320,
            "height": 256,
            "negative_kind": None,
        },
        {
            "id": "n",
            "phrase": "blue circle",
            "present": False,
            "task": "point",
            "target_coordinates": None,
            "width": 320,
            "height": 256,
            "negative_kind": "absent_color",
        },
    ]
    predictions = [
        route_prediction(r, {"id": r["id"], "valid": True, "coordinates": [100, 100]}, False)
        for r in rows
    ]
    counts = grouped_score(rows, predictions)["counts"]
    assert counts["correct_rejection"] == 1
    assert counts["false_rejection"] == 1
    assert counts["positive_localization_success"] == 0
    assert presence_counts(rows, {"p": False, "n": False}) == {
        "true_positive": 0,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 1,
    }


@pytest.mark.parametrize("label", ["<|box_end|>", "red\nsquare", "e\u0301"])
def test_unsafe_gate_labels_rejected(label):
    with pytest.raises(ValueError, match="label"):
        route_prediction({"id": "a", "phrase": label}, {"id": "a"}, False)


@pytest.mark.parametrize("decision", [0, "false", None])
def test_nonboolean_gate_decision_rejected(decision):
    with pytest.raises(ValueError):
        route_prediction({"id": "a"}, {"id": "a"}, decision)


def test_mismatched_ids_and_incomplete_presence_predictions_rejected():
    with pytest.raises(ValueError):
        route_prediction({"id": "a"}, {"id": "b"}, True)
    with pytest.raises(ValueError):
        presence_counts([{"id": "a", "present": True}], {})
    with pytest.raises(ValueError):
        presence_counts([{"id": "a", "present": True}], {"a": 1})
