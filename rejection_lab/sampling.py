"""Deterministic scene selection and epoch schedules without model dependencies."""

import random


def select_training_scenes(rows, count):
    if type(count) is not int or count <= 0:
        raise ValueError("scene count must be a positive integer")
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("only nonempty training-split rows may be selected")
    scenes = sorted({row["scene_id"] for row in rows})
    if count > len(scenes):
        raise ValueError("requested more training scenes than available")
    selected = set(scenes[:count])
    return [row for row in rows if row["scene_id"] in selected]


def training_schedule(examples, steps, *, seed=42, shuffle=False):
    if type(examples) is not int or examples <= 0 or type(steps) is not int or steps <= 0:
        raise ValueError("examples and steps must be positive integers")
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < steps:
        epoch = list(range(examples))
        if shuffle:
            rng.shuffle(epoch)
        schedule.extend(epoch)
    return schedule[:steps]
