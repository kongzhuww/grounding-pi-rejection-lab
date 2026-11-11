"""Bounded, local-only MLX multimodal LoRA smoke; not a production trainer."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from .data import validate_dataset
from .metrics import grouped_score
from .sampling import select_training_scenes, training_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=range(1, 129), default=8)
    parser.add_argument("--train-scenes", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle each training epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--image-tokens", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--distillation-strength", type=float, default=0.0)
    parser.add_argument(
        "--distillation-scope",
        choices=["positive-and-protocol", "positive-only"],
        default="positive-and-protocol",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        not 1 <= args.max_seconds <= 300
        or not 0 <= args.seed < 2**32
        or not math.isfinite(args.learning_rate)
        or not 0 < args.learning_rate <= 1e-3
        or not math.isfinite(args.positive_weight)
        or not 0 < args.positive_weight <= 10
        or not 0 <= args.checkpoint_every <= 128
        or not math.isfinite(args.distillation_strength)
        or not 0 <= args.distillation_strength <= 10
    ):
        parser.error(
            "invalid smoke budget, seed, learning rate, positive weight or checkpoint interval"
        )
    validate_dataset(args.dataset)
    train_rows = select_training_scenes(
        [json.loads(line) for line in (args.dataset / "train.jsonl").read_text().splitlines()],
        args.train_scenes,
    )
    validation = [
        json.loads(line) for line in (args.dataset / "validation.jsonl").read_text().splitlines()
    ]
    if len(validation) > 64:
        raise ValueError("bounded Mac runs allow <=64 complete validation questions")
    if {row["scene_id"] for row in train_rows} & {row["scene_id"] for row in validation}:
        raise ValueError("training/validation scene overlap")
    repo = args.baseline_repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True):
        raise ValueError("baseline checkout must be clean")
    sys.path.insert(0, str(repo))
    import mlx.core as mx
    import mlx.optimizers as optim
    import mlx_grounding_pi
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_grounding_pi.generate import greedy_generate, parse_result
    from mlx_grounding_pi.loading import load_model
    from mlx_grounding_pi.processing import Processor

    from .baseline import normalize_result
    from .training import (
        answer_loss,
        attach_lora,
        cache_teacher,
        gradient_summary,
        model_fingerprint,
        prepare_example,
        preservation_mask,
        protected_answer_loss,
        save_adapter,
    )

    if not Path(mlx_grounding_pi.__file__).resolve().is_relative_to(repo):
        raise ValueError("baseline import path mismatch")
    mx.random.seed(args.seed)
    budget = min(8 * 1024**3, int(mx.device_info()["max_recommended_working_set_size"] * 0.75))
    # MLX's memory limit is a guideline, not an OS hard cap. Also inspect usage between operations.
    mx.set_memory_limit(budget)
    mx.set_cache_limit(64 * 1024**2)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    if index["metadata"]["total_size"] > budget * 0.8:
        raise ValueError("weights leave insufficient headroom for this conservative Mac smoke")
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    report = {
        "phase": "loading",
        "baseline_commit": commit,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "scope": "bounded synthetic development experiment, not natural-image generalization evidence",
        "dataset_manifest": json.loads((args.dataset / "manifest.json").read_text()),
        "training_source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("train_mac.py", "training.py", "sampling.py")
        },
        "requested_steps": args.steps,
        "max_seconds_soft": args.max_seconds,
        "memory_guideline_bytes": budget,
        "image_token_budget": args.image_tokens,
        "learning_rate": args.learning_rate,
        "positive_loss_weight": args.positive_weight,
        "negative_loss_weight": 1.0,
        "distillation_strength": args.distillation_strength,
        "distillation_scope": args.distillation_scope,
        "distillation_policy": (
            "all positive targets; negative CE only; temperature=1"
            if args.distillation_scope == "positive-only"
            else "all positive targets; negative non-payload targets; temperature=1"
        ),
        "checkpoint_every": args.checkpoint_every,
        "checkpoints": [],
        "steps": [],
    }

    def write_report():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_mlx_memory_bytes"] = mx.get_peak_memory()
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    def check_budget():
        if time.perf_counter() - started > args.max_seconds:
            raise TimeoutError("local smoke wall budget reached between operations")
        if mx.get_active_memory() > budget:
            raise MemoryError("local smoke active MLX memory exceeded the guideline")

    write_report()
    model, config = load_model(args.model)
    processor = Processor(args.model, max_output_tokens=args.image_tokens)
    lora = attach_lora(model, layers=2, rank=8, scale=2.0)
    mx.eval(model.parameters())
    report["lora"] = lora
    report["base_fingerprint"] = model_fingerprint(args.model)
    frozen_samples = {
        "visual": model.visual.vision_tower.patch_embed.proj.weight,
        "base_language": model.layers[0].self_attn.q_proj.weight,
    }
    eos = config["eos_token_id"]
    eos = eos if isinstance(eos, int) else eos[0]
    features_cache = {}
    samples = []
    for row in train_rows:
        check_budget()
        sample = prepare_example(model, processor, row, args.dataset, eos, features_cache)
        sample["loss_weight"] = args.positive_weight if row["present"] else 1.0
        samples.append(sample)
    report["train_scenes"] = sorted({row["scene_id"] for row in train_rows})
    report["validation_scenes"] = sorted({row["scene_id"] for row in validation})
    if len(report["train_scenes"]) == 1:
        report["train_scene"] = report["train_scenes"][0]
    report["train_ids"] = [row["id"] for row in train_rows]
    report["validation_ids"] = [row["id"] for row in validation]
    report["supervised_tokens_per_example"] = [sample["targets"].size for sample in samples]

    def mean_loss():
        values = []
        for sample in samples:
            check_budget()
            values.append(float(answer_loss(model, sample).item()))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite teacher-forcing loss")
        return sum(values) / len(values)

    def evaluate(label):
        model.eval()
        predictions = []
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
            predictions.append(
                {
                    "id": row["id"],
                    **normalize_result(parsed, row["phrase"]),
                    "raw_output": text,
                    "finish_reason": finish,
                }
            )
        (args.output / f"validation-{label}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions)
        )
        return grouped_score(validation, predictions)

    report["train_loss_before"] = mean_loss()
    report["validation_before"] = evaluate("before")
    if args.distillation_strength:
        model.eval()
        if any(
            bool(mx.any(value != 0).item())
            for name, value in tree_flatten(model.trainable_parameters())
            if name.endswith(".lora_b")
        ):
            raise ValueError("teacher must be cached before any LoRA update")
        start_id = processor.tokenizer.convert_tokens_to_ids("<|box_start|>")
        end_id = processor.tokenizer.convert_tokens_to_ids("<|box_end|>")
        report["distillation_masked_positions"] = []
        for sample, row in zip(samples, train_rows):
            check_budget()
            mask = preservation_mask(
                sample["targets"][0].tolist(),
                present=row["present"],
                box_start_id=start_id,
                box_end_id=end_id,
                negative_scope="none"
                if args.distillation_scope == "positive-only"
                else "nonpayload",
            )
            if any(mask):
                cache_teacher(model, sample, mask, strength=args.distillation_strength)
            report["distillation_masked_positions"].append(
                {
                    "id": sample["id"],
                    "present": row["present"],
                    "preserved": int(sum(mask)),
                    "total": len(mask),
                }
            )
        report["teacher_cache_bytes"] = sum(
            s["teacher_log_probs"].nbytes for s in samples if "teacher_log_probs" in s
        )
    loss_and_grad = nn.value_and_grad(model, protected_answer_loss)
    checks = []
    for exists in (True, False):
        sample = next(
            sample for sample, row in zip(samples, train_rows) if row["present"] == exists
        )
        (loss, auxiliary), grads = loss_and_grad(model, sample)
        mx.eval(loss, auxiliary, grads)
        if not math.isfinite(float(loss.item())):
            raise ValueError("nonfinite gradient-check loss")
        checks.append(
            {
                "present": exists,
                "loss_weight": sample["loss_weight"],
                "initial_kl": float(auxiliary[1].item()),
                **gradient_summary(grads),
            }
        )
    del grads
    report["gradient_checks"] = checks
    report["phase"] = "training"
    write_report()
    model.train()
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    def adapter_metadata(steps):
        return {
            "format": "grounding-pi-rejection-lab-mlx-lora-v1",
            "lora": lora,
            "base_fingerprint": report["base_fingerprint"],
            "baseline_commit": commit,
            "image_token_budget": args.image_tokens,
            "steps": steps,
            "train_scene_count": len(report["train_scenes"]),
            "dataset_manifest": report["dataset_manifest"],
            "training_source_sha256": report["training_source_sha256"],
            "positive_loss_weight": args.positive_weight,
            "negative_loss_weight": 1.0,
            "distillation_strength": args.distillation_strength,
            "distillation_scope": args.distillation_scope,
            "distillation_policy": report["distillation_policy"],
            "note": "experimental development adapter; no natural-image generalization claim",
        }

    schedule = training_schedule(len(samples), args.steps, seed=args.seed, shuffle=args.shuffle)
    for step, sample_index in enumerate(schedule):
        check_budget()
        sample = samples[sample_index]
        (loss, auxiliary), grads = loss_and_grad(model, sample)
        mx.eval(loss, auxiliary, grads)
        objective_value = float(loss.item())
        value, kl_value = (float(x.item()) for x in auxiliary)
        weighted_value = value * sample["loss_weight"]
        if not all(math.isfinite(x) for x in (value, weighted_value, kl_value, objective_value)):
            raise ValueError("nonfinite training loss")
        gradients = gradient_summary(grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        report["steps"].append(
            {
                "step": step + 1,
                "id": sample["id"],
                "loss": value,
                "weighted_loss": weighted_value,
                "preservation_kl": kl_value,
                "objective_loss": objective_value,
                "loss_weight": sample["loss_weight"],
                "gradient_l2": gradients["l2_norm"],
            }
        )
        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            relative = f"checkpoints/step-{step + 1:04d}"
            save_adapter(model, args.output / relative, adapter_metadata(step + 1))
            report["checkpoints"].append({"step": step + 1, "adapter": relative})
        write_report()
        print(
            f"step {step + 1}/{args.steps}: loss={value:.6f} kl={kl_value:.6f} weight={sample['loss_weight']}",
            flush=True,
        )
    del grads
    model.eval()
    report["train_loss_after"] = mean_loss()
    report["validation_after"] = evaluate("after")
    unchanged = bool(
        mx.array_equal(
            frozen_samples["visual"], model.visual.vision_tower.patch_embed.proj.weight
        ).item()
    ) and bool(
        mx.array_equal(
            frozen_samples["base_language"], model.layers[0].self_attn.q_proj.weight
        ).item()
    )
    if not unchanged:
        raise ValueError("sampled frozen base arrays changed")
    report["frozen_sample_arrays_unchanged"] = unchanged
    report["trainable_array_names"] = [key for key, _ in tree_flatten(model.trainable_parameters())]
    save_adapter(model, args.output / "adapter", adapter_metadata(len(report["steps"])))
    report["phase"] = "completed"
    write_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
