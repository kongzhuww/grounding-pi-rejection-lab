"""Generate original paired positive/negative scenes, split by image, not question."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {"red": (218, 48, 65), "blue": (42, 99, 215), "yellow": (240, 195, 35)}
SHAPES = ("square", "circle", "triangle")
SPLITS = ("train", "validation", "test")
SIZE = (320, 256)


def answer_for(phrase, coordinates):
    payload = "None" if coordinates is None else "".join(f"<{value}>" for value in coordinates)
    return f"<|object_ref_start|>{phrase}<|object_ref_end|><|box_start|>{payload}<|box_end|>"


def normalized(values):
    return [round(v / (SIZE[axis % 2] - 1) * 999) for axis, v in enumerate(values)]


def build_dataset(output, *, scenes=100, seed=42):
    output = Path(output)
    if scenes < 10:
        raise ValueError("At least 10 scenes are required for nonempty scene-disjoint splits")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    rows = {split: [] for split in SPLITS}
    all_pairs = list(itertools.product(COLORS, SHAPES))
    for index in range(scenes):
        scene = f"scene-{index:06d}"
        rng = random.Random(f"{seed}:{scene}")
        pairs = rng.sample(all_pairs, 2)
        image = Image.new("RGB", SIZE, (rng.randint(225, 250),) * 3)
        draw = ImageDraw.Draw(image)
        objects = []
        for slot, (color, shape) in enumerate(pairs):
            cx = (80 if slot == 0 else 240) + rng.randint(-20, 20)
            cy, radius = rng.randint(65, 190), rng.randint(22, 43)
            box = (cx - radius, cy - radius, cx + radius, cy + radius)
            point = (cx, cy)
            if shape == "square":
                draw.rectangle(box, fill=COLORS[color])
            elif shape == "circle":
                draw.ellipse(box, fill=COLORS[color])
            else:
                draw.polygon(
                    [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)],
                    fill=COLORS[color],
                )
                point = (cx, cy + radius / 3)
            objects.append({"color": color, "shape": shape, "point": point, "box": box})
        path = output / "images" / f"{scene}.png"
        image.save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        split = "train" if index % 10 < 8 else "validation" if index % 10 == 8 else "test"
        present = rng.choice(objects)
        absent = rng.choice([pair for pair in all_pairs if pair not in pairs])
        negative_kind = (
            "absent_color"
            if absent[0] not in {c for c, _ in pairs}
            else "absent_shape"
            if absent[1] not in {s for _, s in pairs}
            else "absent_combination"
        )
        for exists in (True, False):
            color, shape = (present["color"], present["shape"]) if exists else absent
            phrase = f"the {color} {shape}"
            for task in ("point", "bbox"):
                values = (
                    normalized(present["point" if task == "point" else "box"]) if exists else None
                )
                instruction = ("Point to" if task == "point" else "Locate") + (
                    f" the target referred to by the following description: {phrase}."
                )
                rows[split].append(
                    {
                        "id": f"{scene}-{'present' if exists else 'absent'}-{task}",
                        "scene_id": scene,
                        "split": split,
                        "image": str(path.relative_to(output)),
                        "image_sha256": digest,
                        "width": SIZE[0],
                        "height": SIZE[1],
                        "task": task,
                        "phrase": phrase,
                        "instruction": instruction,
                        "present": exists,
                        "negative_kind": None if exists else negative_kind,
                        "target_coordinates": values,
                        "answer": answer_for(phrase, values),
                    }
                )
    for split in SPLITS:
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[split])
        )
    manifest = {
        "version": 1,
        "seed": seed,
        "scenes": scenes,
        "counts": {split: len(items) for split, items in rows.items()},
        "split_unit": "scene/image",
        "source": "original synthetic shapes",
        "scope": "pipeline smoke only; not natural-image or benchmark validation",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return validate_dataset(output)


def validate_dataset(output):
    output = Path(output)
    ids, scene_splits, hash_splits, counts = set(), {}, {}, {}
    for split in SPLITS:
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        if not rows:
            raise ValueError(f"empty split: {split}")
        counts[split] = len(rows)
        for row in rows:
            if row["id"] in ids or row["split"] != split:
                raise ValueError("duplicate ID or inconsistent split")
            ids.add(row["id"])
            relative = Path(row["image"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("image must be a safe relative path")
            path = (output / relative).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError("image must remain in the dataset directory")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["image_sha256"]:
                raise ValueError("image digest mismatch")
            for table, key in ((scene_splits, row["scene_id"]), (hash_splits, digest)):
                if key in table and table[key] != split:
                    raise ValueError("scene/image leakage across splits")
                table[key] = split
            target = row["target_coordinates"]
            if type(row["present"]) is not bool or row["task"] not in ("point", "bbox"):
                raise ValueError("invalid presence/task")
            if row["present"]:
                if not isinstance(target, list) or len(target) != (
                    2 if row["task"] == "point" else 4
                ):
                    raise ValueError("invalid coordinate count")
                if any(type(v) is not int or not 0 <= v <= 999 for v in target):
                    raise ValueError("coordinates must be integers in 0..999")
                if row["task"] == "bbox" and not (target[0] < target[2] and target[1] < target[3]):
                    raise ValueError("degenerate box")
            elif target is not None:
                raise ValueError("absent target must have None coordinates")
            if row["answer"] != answer_for(row["phrase"], target):
                raise ValueError("answer/annotation mismatch")
    return {"counts": counts, "scenes": len(scene_splits), "unique_images": len(hash_splits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = (
        validate_dataset(args.output)
        if args.validate_only
        else build_dataset(args.output, scenes=args.scenes, seed=args.seed)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
