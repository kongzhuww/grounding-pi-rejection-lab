"""Score normalized predictions without rewarding invalid output or always-None."""

import argparse
import json
import math
from pathlib import Path


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def valid_prediction(prediction, task):
    if prediction.get("valid") is not True:
        return False
    coords = prediction.get("coordinates")
    if "coordinates" not in prediction:
        return False
    if coords is None:
        return True
    if not isinstance(coords, list) or len(coords) != (2 if task == "point" else 4):
        return False
    if any(type(v) is not int or not 0 <= v <= 999 for v in coords):
        return False
    return task != "bbox" or (coords[0] < coords[2] and coords[1] < coords[3])


def score(rows, predictions, *, point_tolerance_pixels=15.0, minimum_iou=0.70):
    if point_tolerance_pixels < 0 or not 0 <= minimum_iou <= 1:
        raise ValueError("invalid threshold")
    truth = {row["id"]: row for row in rows}
    by_id = {pred["id"]: pred for pred in predictions}
    if (
        not rows
        or len(truth) != len(rows)
        or len(by_id) != len(predictions)
        or truth.keys() != by_id.keys()
    ):
        raise ValueError("unique prediction IDs must cover exactly the requested evaluation rows")
    counts = {
        key: 0
        for key in (
            "positive",
            "negative",
            "false_positive",
            "false_rejection",
            "correct_rejection",
            "positive_localization_success",
            "invalid",
        )
    }
    for row in rows:
        pred = by_id[row["id"]]
        exists = row["present"]
        counts["positive" if exists else "negative"] += 1
        if not valid_prediction(pred, row["task"]):
            counts["invalid"] += 1
            continue
        coords = pred["coordinates"]
        if not exists:
            counts["correct_rejection" if coords is None else "false_positive"] += 1
        elif coords is None:
            counts["false_rejection"] += 1
        else:
            target = row["target_coordinates"]
            if row["task"] == "point":
                dx = (coords[0] - target[0]) / 999 * (row["width"] - 1)
                dy = (coords[1] - target[1]) / 999 * (row["height"] - 1)
                good = math.hypot(dx, dy) <= point_tolerance_pixels
            else:
                good = box_iou(coords, target) >= minimum_iou
            counts["positive_localization_success"] += int(good)

    def rate(value, denominator):
        return value / denominator if denominator else None

    p, n = counts["positive"], counts["negative"]
    return {
        "counts": counts,
        "false_positive_rate": rate(counts["false_positive"], n),
        "negative_failure_rate_including_invalid": rate(n - counts["correct_rejection"], n),
        "correct_rejection_rate": rate(counts["correct_rejection"], n),
        "false_rejection_rate": rate(counts["false_rejection"], p),
        "positive_localization_success_rate": rate(counts["positive_localization_success"], p),
        "invalid_output_rate": rate(counts["invalid"], len(rows)),
        "point_tolerance_pixels": point_tolerance_pixels,
        "minimum_iou": minimum_iou,
        "scope": "supplied split only; synthetic scores are not natural-image benchmark accuracy",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True, help="A single split JSONL")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.truth.read_text().splitlines()]
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    report = score(rows, predictions)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
