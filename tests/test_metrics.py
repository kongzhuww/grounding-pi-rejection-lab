import json

import pytest

from rejection_lab.baseline import normalize_result
from rejection_lab.data import build_dataset
from rejection_lab.metrics import score, valid_prediction


@pytest.fixture
def rows(tmp_path):
    build_dataset(tmp_path / "data", scenes=10)
    return [
        json.loads(line) for line in (tmp_path / "data/validation.jsonl").read_text().splitlines()
    ]


def test_perfect_prediction_scorer_unit_check_not_model_result(rows):
    predictions = [
        {"id": r["id"], "valid": True, "coordinates": r["target_coordinates"]} for r in rows
    ]
    result = score(rows, predictions)
    assert result["correct_rejection_rate"] == 1
    assert result["positive_localization_success_rate"] == 1
    assert result["negative_failure_rate_including_invalid"] == 0


def test_always_none_is_not_success(rows):
    predictions = [{"id": r["id"], "valid": True, "coordinates": None} for r in rows]
    result = score(rows, predictions)
    assert result["correct_rejection_rate"] == 1
    assert result["false_rejection_rate"] == 1
    assert result["positive_localization_success_rate"] == 0


def test_invalid_is_not_correct_rejection(rows):
    predictions = [{"id": r["id"], "valid": False, "coordinates": None} for r in rows]
    result = score(rows, predictions)
    assert result["invalid_output_rate"] == 1
    assert result["correct_rejection_rate"] == 0
    assert result["negative_failure_rate_including_invalid"] == 1


def test_always_coordinate_exposes_false_positives(rows):
    predictions = [
        {
            "id": r["id"],
            "valid": True,
            "coordinates": [0, 0] if r["task"] == "point" else [0, 0, 999, 999],
        }
        for r in rows
    ]
    result = score(rows, predictions)
    assert result["false_positive_rate"] == 1
    assert result["correct_rejection_rate"] == 0


def test_missing_duplicate_extra_ids_rejected(rows):
    predictions = [{"id": r["id"], "valid": True, "coordinates": None} for r in rows]
    for malformed in (
        predictions[:-1],
        predictions + [predictions[0]],
        predictions + [{"id": "extra", "valid": True, "coordinates": None}],
    ):
        with pytest.raises(ValueError, match="exactly"):
            score(rows, malformed)


@pytest.mark.parametrize(
    "prediction,task",
    [
        ({"valid": True, "coordinates": [True, 10]}, "point"),
        ({"valid": "true", "coordinates": None}, "point"),
        ({"valid": True}, "point"),
        ({"valid": True, "coordinates": [1000, 0]}, "point"),
        ({"valid": True, "coordinates": [100, 100, 100, 200]}, "bbox"),
    ],
)
def test_invalid_coordinate_structure(prediction, task):
    assert not valid_prediction(prediction, task)


def test_normalizer_requires_explicit_matching_single_entry():
    assert normalize_result(
        {"valid": True, "predictions": [{"label": "the red square", "coordinates": None}]},
        "the red square",
    ) == {"valid": True, "coordinates": None}
    assert normalize_result({"valid": True, "predictions": []}, "target")["valid"] is False
    assert (
        normalize_result(
            {"valid": True, "predictions": [{"label": "wrong target", "coordinates": None}]},
            "target",
        )["valid"]
        is False
    )
