"""Bounded local existence-head experiment; original localizer is always frozen."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from .data import validate_dataset
from .gating import presence_counts, route_prediction
from .metrics import grouped_score
from .sampling import select_training_scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-repo", "model", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--head", type=Path, help="Evaluate an existing head only; no training")
    parser.add_argument(
        "--evaluation-role",
        choices=["reused-development", "fresh-synthetic-holdout"],
        default="reused-development",
        help="Provenance label only; fresh holdout requires a fixed head, never training",
    )
    parser.add_argument(
        "--feature-query", choices=["task-specific", "canonical"], default="task-specific"
    )
    parser.add_argument("--train-scenes", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--steps", type=int, choices=[32, 64, 128], default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-tokens", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--max-seconds", type=int, default=240)
    args = parser.parse_args()
    if args.evaluation_role == "fresh-synthetic-holdout" and not args.head:
        parser.error("fresh holdout evaluation requires --head; training is prohibited")
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        not 1 <= args.max_seconds <= 300
        or not 0 <= args.seed < 2**32
        or not math.isfinite(args.learning_rate)
        or not 0 < args.learning_rate <= 0.1
        or not math.isfinite(args.regularization)
        or not 0 <= args.regularization <= 1
        or not math.isfinite(args.positive_weight)
        or not 0 < args.positive_weight <= 10
    ):
        parser.error("invalid bounded training parameters")
    validate_dataset(args.dataset)
    repo = args.baseline_repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True):
        raise ValueError("baseline checkout must remain clean")
    pinned = json.loads((Path(__file__).parent.parent / "baseline.json").read_text())["commit"]
    if commit != pinned:
        raise ValueError("this experiment requires the pinned baseline commit")
    sys.path.insert(0, str(repo))
    import mlx.core as mx
    import mlx.optimizers as optim
    import mlx_grounding_pi
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_grounding_pi.coordinates import parse_answer
    from mlx_grounding_pi.generate import greedy_generate, parse_result
    from mlx_grounding_pi.loading import load_model
    from mlx_grounding_pi.processing import Processor

    from .baseline import normalize_result
    from .existence import (
        FEATURE_VERSIONS,
        ExistenceHead,
        feature_instruction,
        head_loss,
        load_head,
        prompt_feature,
        save_head,
    )
    from .training import model_fingerprint

    if not Path(mlx_grounding_pi.__file__).resolve().is_relative_to(repo):
        raise ValueError("unexpected baseline import path")
    mx.random.seed(args.seed)
    budget = min(8 * 1024**3, int(mx.device_info()["max_recommended_working_set_size"] * 0.75))
    mx.set_memory_limit(budget)
    mx.set_cache_limit(64 * 1024**2)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    if index["metadata"]["total_size"] > budget * 0.8:
        raise ValueError("model leaves insufficient memory headroom")
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    report = {
        "phase": "loading",
        "scope": (
            "fresh same-generator synthetic holdout; not natural-image or production acceptance"
            if args.evaluation_role == "fresh-synthetic-holdout"
            else "reused synthetic development; not final acceptance"
        ),
        "evaluation_role": args.evaluation_role,
        "baseline_commit": commit,
        "base_fingerprint": model_fingerprint(args.model),
        "feature_version": FEATURE_VERSIONS[args.feature_query],
        "feature_query": args.feature_query,
        "image_token_budget": args.image_tokens,
        "dataset_manifest": json.loads((args.dataset / "manifest.json").read_text()),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "positive_weight": args.positive_weight,
        "negative_weight": 1.0,
        "requested_steps": args.steps if not args.head else 0,
        "max_seconds_soft": args.max_seconds,
        "memory_guideline_bytes": budget,
        "training_source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "train_existence.py",
                "existence.py",
                "gating.py",
                "sampling.py",
                "metrics.py",
            )
        },
        "routing": "classifier argmax; uncalibrated scores; absent renders GAM None, present retains exact localizer output",
        "evaluation": "all baseline outputs generated once for control, then routing replay; no latency-saving claim",
        "steps": [],
        "checkpoints": [],
        "test_inference": False,
    }

    def write_report():
        report.update(
            elapsed_seconds=time.perf_counter() - started,
            peak_mlx_memory_bytes=mx.get_peak_memory(),
        )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    def check_budget():
        if time.perf_counter() - started > args.max_seconds:
            raise TimeoutError("local existence experiment time budget reached")
        if mx.get_active_memory() > budget:
            raise MemoryError("active MLX memory exceeded guideline")

    def write_jsonl(name, rows):
        (args.output / name).write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
        )

    write_report()
    model, config = load_model(args.model)
    model.freeze()
    model.eval()
    if tree_flatten(model.trainable_parameters()):
        raise ValueError("the entire original localizer must be frozen")
    frozen = {
        "vision": model.visual.vision_tower.patch_embed.proj.weight,
        "language": model.layers[0].self_attn.q_proj.weight,
    }
    processor = Processor(args.model, max_output_tokens=args.image_tokens)
    validation = [
        json.loads(x) for x in (args.dataset / "validation.jsonl").read_text().splitlines()
    ]
    if len(validation) > 64:
        raise ValueError("bounded development split permits at most 64 questions")
    train_rows = (
        []
        if args.head
        else select_training_scenes(
            [json.loads(x) for x in (args.dataset / "train.jsonl").read_text().splitlines()],
            args.train_scenes,
        )
    )
    if {r["scene_id"] for r in train_rows} & {r["scene_id"] for r in validation}:
        raise ValueError("training/validation scene overlap")
    report["train_ids"] = [r["id"] for r in train_rows]
    report["validation_ids"] = [r["id"] for r in validation]
    visual_cache = {}
    feature_cache = {}
    feature_audit = []

    def extract(rows):
        values = []
        for row in rows:
            check_budget()
            # Only image and user phrase/instruction: never gold presence/coordinates.
            instruction = feature_instruction(row["phrase"], row["instruction"], args.feature_query)
            feature_audit.append(
                {"id": row["id"], "image_sha256": row["image_sha256"], "instruction": instruction}
            )
            cache_key = (row["image_sha256"], instruction)
            if cache_key in feature_cache:
                values.append(feature_cache[cache_key])
                continue
            inputs = processor(args.dataset / row["image"], instruction)
            key = row["image_sha256"]
            if key not in visual_cache:
                visual_cache[key] = mx.stop_gradient(
                    model.visual(mx.array(inputs["pixel_values"]), inputs["image_grid_thw"])
                )
                mx.eval(visual_cache[key])
            feature = prompt_feature(model, inputs, visual_features=visual_cache[key])
            mx.eval(feature)
            if (
                not bool(mx.all(mx.isfinite(feature)).item())
                or abs(float(mx.sum(feature**2).item()) - 1) > 1e-4
            ):
                raise ValueError("feature must be finite and unit-length")
            feature_cache[cache_key] = feature
            values.append(feature)
        return mx.stack(values)

    xtrain = extract(train_rows) if train_rows else None
    xval = extract(validation)
    mx.eval(xval)
    write_jsonl("feature-inputs.jsonl", feature_audit)
    report["paired_features"] = {}
    for label, rows, features in [("train", train_rows, xtrain), ("validation", validation, xval)]:
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault((row["image_sha256"], row["phrase"]), []).append(i)
        pairs = [indices for indices in groups.values() if len(indices) == 2]
        difference = max(
            (float(mx.max(mx.abs(features[a] - features[b])).item()) for a, b in pairs),
            default=0.0,
        )
        report["paired_features"][label] = {"pairs": len(pairs), "max_abs_difference": difference}
        if args.feature_query == "canonical" and difference != 0:
            raise ValueError("canonical point/box paired features must be identical")
    baseline = []
    for row in validation:
        check_budget()
        inputs = processor(args.dataset / row["image"], row["instruction"])
        tokens, finish, _ = greedy_generate(
            model, inputs, eos_token_id=config["eos_token_id"], max_tokens=128, max_context=512
        )
        text = processor.tokenizer.decode(
            tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        parsed = parse_result(text, row["task"], inputs["image"].original_size, finish)
        baseline.append(
            {
                "id": row["id"],
                **normalize_result(parsed, row["phrase"]),
                "raw_output": text,
                "finish_reason": finish,
            }
        )
    write_jsonl("baseline-predictions.jsonl", baseline)
    report["baseline"] = grouped_score(validation, baseline)
    report["feature_dim"] = xval.shape[1]

    def evaluate(head, label):
        head.eval()
        scores = head(xval)
        decisions = mx.argmax(scores, axis=-1).tolist()
        choices = {r["id"]: bool(d) for r, d in zip(validation, decisions)}
        preds = [route_prediction(r, b, choices[r["id"]]) for r, b in zip(validation, baseline)]
        for row, pred in zip(validation, preds):
            if pred["origin"] == "existence_gate":
                entries = parse_answer(pred["raw_output"], row["task"])
                if len(entries) != 1 or entries[0].coordinates is not None:
                    raise ValueError("constructed gate output failed strict GAM validation")
        write_jsonl(f"predictions-{label}.jsonl", preds)
        write_jsonl(
            f"gate-{label}.jsonl",
            [
                {"id": r["id"], "present": choices[r["id"]], "uncalibrated_logits": s}
                for r, s in zip(validation, scores.tolist())
            ],
        )
        return {
            "classification": presence_counts(validation, choices),
            "joint": grouped_score(validation, preds),
        }

    if args.head:
        head, metadata = load_head(
            args.head,
            expected_fingerprint=report["base_fingerprint"],
            expected_commit=commit,
            expected_budget=args.image_tokens,
            expected_feature_version=FEATURE_VERSIONS[args.feature_query],
        )
        if metadata["feature_dim"] != xval.shape[1]:
            raise ValueError("loaded head and feature dimensions disagree")
        report["head_metadata"] = metadata
        report["positive_weight"] = metadata.get("positive_weight", 1.0)
        report["negative_weight"] = metadata.get("negative_weight", 1.0)
        report["regularization"] = metadata["regularization"]
        report["learning_rate"] = metadata["learning_rate"]
        report["evaluation_result"] = evaluate(head, "reloaded")
    else:
        y = mx.array([int(r["present"]) for r in train_rows], dtype=mx.int32)
        if int(mx.sum(y).item()) * 2 != len(train_rows):
            raise ValueError("this controlled probe expects balanced training classes")
        head = ExistenceHead(xtrain.shape[1])
        report["head_parameters"] = sum(x.size for _, x in tree_flatten(head.parameters()))
        report["train_objective_before"] = float(
            head_loss(head, xtrain, y, args.regularization, args.positive_weight)[0].item()
        )
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        value_grad = nn.value_and_grad(head, head_loss)
        report["phase"] = "training_head"
        mx.save_safetensors(
            str(args.output / "features.safetensors"), {"train": xtrain, "validation": xval}
        )
        for step in range(1, args.steps + 1):
            check_budget()
            (loss, auxiliary), grads = value_grad(
                head, xtrain, y, args.regularization, args.positive_weight
            )
            mx.eval(loss, auxiliary, grads)
            if not math.isfinite(float(loss.item())) or any(
                not bool(mx.all(mx.isfinite(g)).item()) for _, g in tree_flatten(grads)
            ):
                raise ValueError("nonfinite head loss/gradient")
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state)
            report["steps"].append(
                {
                    "step": step,
                    "objective": float(loss.item()),
                    "ce": float(auxiliary[0].item()),
                    "penalty": float(auxiliary[1].item()),
                }
            )
            if step in {32, 64, args.steps}:
                relative = f"checkpoints/step-{step:04d}"
                metadata = {
                    "format": "grounding-pi-existence-head-v1",
                    "feature_version": FEATURE_VERSIONS[args.feature_query],
                    "feature_query": args.feature_query,
                    "feature_dim": xtrain.shape[1],
                    "classes": ["absent", "present"],
                    "base_fingerprint": report["base_fingerprint"],
                    "baseline_commit": commit,
                    "image_token_budget": args.image_tokens,
                    "steps": step,
                    "regularization": args.regularization,
                    "positive_weight": args.positive_weight,
                    "negative_weight": 1.0,
                    "learning_rate": args.learning_rate,
                    "train_ids": report["train_ids"],
                    "training_source_sha256": report["training_source_sha256"],
                }
                save_head(head, args.output / relative, metadata)
                result = evaluate(head, f"step-{step:04d}")
                report["checkpoints"].append({"step": step, "head": relative, **result})
                head.train()
            write_report()
        report["train_objective_after"] = float(
            head_loss(head, xtrain, y, args.regularization, args.positive_weight)[0].item()
        )
        choices = {
            r["id"]: bool(d) for r, d in zip(train_rows, mx.argmax(head(xtrain), axis=-1).tolist())
        }
        report["train_classification"] = presence_counts(train_rows, choices)
    unchanged = bool(
        mx.array_equal(frozen["vision"], model.visual.vision_tower.patch_embed.proj.weight).item()
    ) and bool(mx.array_equal(frozen["language"], model.layers[0].self_attn.q_proj.weight).item())
    if not unchanged or tree_flatten(model.trainable_parameters()):
        raise ValueError("frozen original localizer unexpectedly changed")
    report["frozen_sample_arrays_unchanged"] = True
    report["phase"] = "completed"
    write_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
