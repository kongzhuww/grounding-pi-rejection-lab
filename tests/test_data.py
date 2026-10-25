import json

import pytest

from rejection_lab.data import SPLITS, answer_for, build_dataset, validate_dataset


def test_paired_data_deterministic_and_disjoint(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    result = build_dataset(first, scenes=10, seed=7)
    assert result["counts"] == {"train": 32, "validation": 4, "test": 4}
    assert result["scenes"] == result["unique_images"] == 10
    build_dataset(second, scenes=10, seed=7)
    for split in SPLITS:
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
        rows = [json.loads(line) for line in (first / f"{split}.jsonl").read_text().splitlines()]
        assert sum(row["present"] for row in rows) == len(rows) // 2
        for row in rows:
            assert row["answer"] == answer_for(row["phrase"], row["target_coordinates"])
            assert (first / row["image"]).read_bytes() == (second / row["image"]).read_bytes()


def test_refuse_overwrite_and_too_few_scenes(tmp_path):
    build_dataset(tmp_path / "data", scenes=10)
    with pytest.raises(FileExistsError):
        build_dataset(tmp_path / "data", scenes=10)
    with pytest.raises(ValueError):
        build_dataset(tmp_path / "tiny", scenes=9)


def test_scene_leakage_rejected(tmp_path):
    build_dataset(tmp_path / "data", scenes=10)
    root = tmp_path / "data"
    train = json.loads((root / "train.jsonl").read_text().splitlines()[0])
    file = root / "validation.jsonl"
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    rows[0]["scene_id"] = train["scene_id"]
    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset(root)


def test_image_tampering_rejected(tmp_path):
    build_dataset(tmp_path / "data", scenes=10)
    (tmp_path / "data/images/scene-000000.png").write_bytes(b"not the image")
    with pytest.raises(ValueError, match="digest"):
        validate_dataset(tmp_path / "data")


def test_absent_annotation_must_be_none(tmp_path):
    build_dataset(tmp_path / "data", scenes=10)
    path = tmp_path / "data/validation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["target_coordinates"] = [0, 0, 100, 100]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="absent target"):
        validate_dataset(tmp_path / "data")
