import json

import pytest

from rejection_lab.data import build_dataset
from rejection_lab.sampling import select_training_scenes, training_schedule


def test_multiscene_selection_keeps_all_paired_questions_and_no_validation(tmp_path):
    build_dataset(tmp_path / "data", scenes=100, seed=2025)
    root = tmp_path / "data"
    train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (root / "validation.jsonl").read_text().splitlines()]
    selected = select_training_scenes(train, 8)
    assert len(selected) == 32
    assert sum(row["present"] for row in selected) == 16
    scene_ids = {row["scene_id"] for row in selected}
    assert len(scene_ids) == 8
    assert all(sum(row["scene_id"] == scene for row in selected) == 4 for scene in scene_ids)
    assert not scene_ids & {row["scene_id"] for row in validation}
    assert len(validation) == 40


def test_selection_rejects_heldout_or_insufficient_scenes():
    with pytest.raises(ValueError, match="training-split"):
        select_training_scenes([{"scene_id": "v", "split": "validation"}], 1)
    with pytest.raises(ValueError, match="more training scenes"):
        select_training_scenes([{"scene_id": "t", "split": "train"}], 2)
    with pytest.raises(ValueError, match="positive integer"):
        select_training_scenes([], 0)


def test_shuffled_epochs_visit_every_example_and_reproduce():
    first = training_schedule(32, 64, seed=42, shuffle=True)
    assert first == training_schedule(32, 64, seed=42, shuffle=True)
    assert first != training_schedule(32, 64, seed=43, shuffle=True)
    assert sorted(first[:32]) == sorted(first[32:]) == list(range(32))
    assert first[:32] != first[32:]


def test_default_sequential_schedule_preserves_original_smoke():
    assert training_schedule(4, 8) == [0, 1, 2, 3, 0, 1, 2, 3]


def test_invalid_schedule_rejected():
    with pytest.raises(ValueError, match="positive integers"):
        training_schedule(0, 10)
