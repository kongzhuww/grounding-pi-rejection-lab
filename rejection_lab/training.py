"""Experimental MLX multimodal LoRA primitives; imports require Apple Silicon.

Use mlx-lm's installed LoRA implementation, not a copied or text-only trainer.
Frozen visual features are cached; only answer/EOS positions enter the loss.
"""

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

TARGETS = ("q_proj", "v_proj")


def attach_lora(model, *, layers=2, rank=8, scale=2.0):
    if not 1 <= layers <= len(model.layers) or rank <= 0 or scale <= 0:
        raise ValueError("invalid LoRA layer count/rank/scale")
    if any(isinstance(m, LoRALinear) for _, m in model.named_modules()):
        raise ValueError("LoRA is already attached")
    model.freeze()
    for layer in model.layers[-layers:]:
        for name in TARGETS:
            base = getattr(layer.self_attn, name)
            setattr(
                layer.self_attn, name, LoRALinear.from_base(base, r=rank, scale=scale, dropout=0.0)
            )
    params = dict(tree_flatten(model.trainable_parameters()))
    if not params or any(not key.endswith((".lora_a", ".lora_b")) for key in params):
        raise ValueError("only LoRA arrays may remain trainable")
    return {
        "layers": layers,
        "rank": rank,
        "scale": scale,
        "dropout": 0.0,
        "targets": list(TARGETS),
        "trainable_parameters": sum(v.size for v in params.values()),
    }


def teacher_forcing(prompt_ids, answer_ids, eos_token_id):
    if not prompt_ids or not answer_ids or type(eos_token_id) is not int:
        raise ValueError("nonempty prompt/answer and one explicit EOS are required")
    targets = list(answer_ids) + [eos_token_id]
    return {
        "input_ids": mx.array([list(prompt_ids) + targets[:-1]], dtype=mx.int32),
        "targets": mx.array([targets], dtype=mx.int32),
        "answer_start": len(prompt_ids) - 1,
    }


def prepare_example(model, processor, row, dataset, eos_token_id, features_cache):
    inputs = processor(Path(dataset) / row["image"], row["instruction"])
    prompt = inputs["input_ids"][0].tolist()
    combined = processor.tokenizer.encode(
        inputs["prompt"] + row["answer"], add_special_tokens=False
    )
    if combined[: len(prompt)] != prompt:
        raise ValueError("answer tokenization changed the prompt boundary")
    sample = teacher_forcing(prompt, combined[len(prompt) :], eos_token_id)
    if sample["input_ids"].shape[1] > 512:
        raise ValueError("initial Mac smoke requires sequences <=512 tokens; no truncation")
    key = row["image_sha256"]
    if key not in features_cache:
        features = mx.stop_gradient(
            model.visual(mx.array(inputs["pixel_values"]), inputs["image_grid_thw"])
        )
        mx.eval(features)
        features_cache[key] = features
    sample["features"] = features_cache[key]
    sample["image_positions"] = [
        i for i, token in enumerate(prompt) if token == model.image_token_id
    ]
    if len(sample["image_positions"]) != sample["features"].shape[0]:
        raise ValueError("image slots/features mismatch")
    sample["id"] = row["id"]
    return sample


def answer_logits(model, sample):
    ids = sample["input_ids"]
    embeddings = model.language_model.model.embed_tokens(ids)
    features = mx.stop_gradient(sample["features"]).astype(embeddings.dtype)
    embeddings[0, mx.array(sample["image_positions"])] = features
    hidden = model.language_model.model(ids, input_embeddings=embeddings)
    hidden = hidden[:, sample["answer_start"] :]
    if model.language_model.args.tie_word_embeddings:
        logits = model.language_model.model.embed_tokens.as_linear(hidden)
    else:
        logits = model.language_model.lm_head(hidden)
    if logits.shape[:2] != sample["targets"].shape:
        raise ValueError("shifted answer loss shape mismatch")
    return logits.astype(mx.float32)


def answer_loss(model, sample):
    return nn.losses.cross_entropy(
        answer_logits(model, sample), sample["targets"], reduction="mean"
    )


def weighted_answer_loss(model, sample):
    """Example weighting; with batch-1 Adam this is not a literal update multiplier."""
    weight = sample.get("loss_weight", 1.0)
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("loss weight must be a finite positive scalar")
    return answer_loss(model, sample) * weight


def preservation_mask(
    target_ids, *, present, box_start_id, box_end_id, negative_scope="nonpayload"
):
    """Preserve positives; negative scope is nonpayload or none (CE-only, no teacher cache)."""
    if negative_scope not in ("nonpayload", "none"):
        raise ValueError("unknown negative distillation scope")
    if target_ids.count(box_start_id) != 1 or target_ids.count(box_end_id) != 1:
        raise ValueError("one spatial block is required for distillation masking")
    start, end = target_ids.index(box_start_id), target_ids.index(box_end_id)
    if start + 1 >= end:
        raise ValueError("nonempty ordered spatial payload required")
    return [
        float(present or (negative_scope == "nonpayload" and not start < i < end))
        for i in range(len(target_ids))
    ]


def cache_teacher(model, sample, mask, *, strength):
    """Call before any updates, with zero-delta LoRA; cache detached FP32 distributions."""
    if not math.isfinite(strength) or not 0 < strength <= 10:
        raise ValueError("distillation strength must be finite and in (0,10]")
    if (
        len(mask) != sample["targets"].shape[1]
        or any(x not in (0, 1) for x in mask)
        or not any(mask)
    ):
        raise ValueError("distillation mask must be nonempty binary answer-position mask")
    logits = answer_logits(model, sample)
    teacher = mx.stop_gradient(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    mx.eval(teacher)
    if not bool(mx.all(mx.isfinite(teacher)).item()):
        raise ValueError("nonfinite teacher distribution")
    sample.update(
        teacher_log_probs=teacher,
        preservation_mask=mx.array([mask], dtype=mx.float32),
        distillation_strength=strength,
    )


def masked_teacher_kl(logits, teacher_log_probs, mask):
    """FP32 forward KL, using a nonempty binary position mask validated when caching."""
    teacher = mx.stop_gradient(teacher_log_probs)
    if teacher.shape != logits.shape or mask.shape != logits.shape[:-1]:
        raise ValueError("teacher/student/mask shape mismatch")
    student = logits.astype(mx.float32)
    student = student - mx.logsumexp(student, axis=-1, keepdims=True)
    per_position = mx.sum(mx.exp(teacher) * (teacher - student), axis=-1)
    return mx.sum(mask * per_position) / mx.sum(mask)


def protected_answer_loss(model, sample):
    """Return weighted CE + lambda * masked KL(teacher || student), plus unweighted CE/KL."""
    weight = sample.get("loss_weight", 1.0)
    strength = sample.get("distillation_strength", 0.0)
    if (
        not math.isfinite(weight)
        or weight <= 0
        or not math.isfinite(strength)
        or not 0 <= strength <= 10
    ):
        raise ValueError("invalid loss coefficients")
    logits = answer_logits(model, sample)
    ce = nn.losses.cross_entropy(logits, sample["targets"], reduction="mean")
    kl = mx.array(0.0)
    if strength:
        kl = masked_teacher_kl(logits, sample["teacher_log_probs"], sample["preservation_mask"])
    return weight * ce + strength * kl, (ce, kl)


def gradient_summary(grads):
    arrays = dict(tree_flatten(grads))
    if not arrays or any(not name.endswith((".lora_a", ".lora_b")) for name in arrays):
        raise ValueError("unexpected non-LoRA gradient")
    finite = all(bool(mx.all(mx.isfinite(g)).item()) for g in arrays.values())
    norm = float(mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in arrays.values())).item())
    if not finite or not math.isfinite(norm) or norm <= 0:
        raise ValueError("gradients must be finite and at least one must be nonzero")
    # lora_a may legitimately have zero gradient initially because lora_b starts at zero.
    return {"finite": finite, "l2_norm": norm, "arrays": len(arrays)}


def model_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    for name in ("config.json", "conversion.json", "model.safetensors.index.json"):
        digest.update(name.encode())
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


def save_adapter(model, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = dict(tree_flatten(model.trainable_parameters()))
    if not weights or any(not key.endswith((".lora_a", ".lora_b")) for key in weights):
        raise ValueError("refusing to save anything except LoRA arrays")
    mx.eval(weights)
    mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
    (directory / "adapter_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_adapter(model, directory, *, expected_fingerprint):
    directory = Path(directory)
    metadata = json.loads((directory / "adapter_config.json").read_text())
    if metadata["base_fingerprint"] != expected_fingerprint:
        raise ValueError("adapter/base model fingerprint mismatch")
    weights = mx.load(str(directory / "adapters.safetensors"))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(weights) != set(expected) or any(
        weights[k].shape != expected[k].shape for k in expected
    ):
        raise ValueError("adapter parameter names/shapes mismatch")
    if any(not bool(mx.all(mx.isfinite(w)).item()) for w in weights.values()):
        raise ValueError("nonfinite adapter weights")
    model.load_weights(list(weights.items()), strict=False)
    return metadata
