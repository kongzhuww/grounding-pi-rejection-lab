"""Frozen prompt-only multimodal features and an independent regularized linear probe."""

import json
import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

FEATURE_VERSION = "frozen-final-prompt-hidden-l2-v1"
CANONICAL_FEATURE_VERSION = "frozen-final-canonical-existence-hidden-l2-v2"
FEATURE_VERSIONS = {
    "task-specific": FEATURE_VERSION,
    "canonical": CANONICAL_FEATURE_VERSION,
}


def feature_instruction(phrase, instruction, query_mode="task-specific"):
    """Only user-provided phrase/instruction enter; never labels or coordinates."""
    if query_mode not in FEATURE_VERSIONS:
        raise ValueError("unknown feature query mode")
    if query_mode == "task-specific":
        return instruction
    if not isinstance(phrase, str) or not phrase.strip() or "<|" in phrase:
        raise ValueError("nonempty plain target phrase required")
    return f"Is the following target present in the image? {phrase}. Answer yes or no."


def prompt_feature(model, inputs, *, visual_features=None):
    """No gold answer or generated tokens enter this feature. Output is stop-gradient."""
    ids = mx.array(inputs["input_ids"])
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("one nonempty prompt is required")
    positions = [i for i, token in enumerate(ids[0].tolist()) if token == model.image_token_id]
    if visual_features is None:
        visual_features = model.visual(mx.array(inputs["pixel_values"]), inputs["image_grid_thw"])
    if not positions or len(positions) != visual_features.shape[0]:
        raise ValueError("prompt image slots/features mismatch")
    embeddings = model.language_model.model.embed_tokens(ids)
    embeddings[0, mx.array(positions)] = mx.stop_gradient(visual_features).astype(embeddings.dtype)
    hidden = model.language_model.model(ids, input_embeddings=embeddings)
    feature = mx.stop_gradient(hidden[0, -1].astype(mx.float32))
    # No fitted validation statistics; every feature is normalized independently.
    return feature / mx.maximum(mx.sqrt(mx.sum(feature * feature)), 1e-8)


class ExistenceHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("positive feature dimension required")
        self.projection = nn.Linear(dimension, 2)
        # Balanced classes, no initial class preference. Exact ties select class 0.
        self.projection.weight = mx.zeros((2, dimension))
        self.projection.bias = mx.zeros((2,))

    def __call__(self, features):
        return self.projection(mx.stop_gradient(features))


def head_loss(head, features, labels, regularization=0.01, positive_weight=1.0):
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("finite nonnegative regularization required")
    if not math.isfinite(positive_weight) or not 0 < positive_weight <= 10:
        raise ValueError("positive class weight must be finite and in (0,10]")
    if positive_weight == 1:
        ce = nn.losses.cross_entropy(head(features), labels, reduction="mean")
    else:
        per_example = nn.losses.cross_entropy(head(features), labels, reduction="none")
        weights = mx.where(labels == 1, positive_weight, 1.0)
        ce = mx.sum(weights * per_example) / mx.sum(weights)
    penalty = 0.5 * regularization * mx.sum(head.projection.weight**2)
    return ce + penalty, (ce, penalty)


def save_head(head, directory, metadata):
    directory = Path(directory)
    weights = dict(tree_flatten(head.parameters()))
    if set(weights) != {"projection.weight", "projection.bias"}:
        raise ValueError("only the independent linear head may be saved")
    mx.eval(weights)
    if any(not bool(mx.all(mx.isfinite(x)).item()) for x in weights.values()):
        raise ValueError("nonfinite head weights")
    directory.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(str(directory / "head.safetensors"), weights)
    (directory / "head_config.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


def load_head(
    directory,
    *,
    expected_fingerprint,
    expected_commit,
    expected_budget,
    expected_feature_version=FEATURE_VERSION,
):
    directory = Path(directory)
    metadata = json.loads((directory / "head_config.json").read_text())
    if expected_feature_version not in FEATURE_VERSIONS.values():
        raise ValueError("unknown expected feature version")
    if (
        metadata["format"] != "grounding-pi-existence-head-v1"
        or metadata["feature_version"] != expected_feature_version
        or metadata["classes"] != ["absent", "present"]
        or metadata["base_fingerprint"] != expected_fingerprint
        or metadata["baseline_commit"] != expected_commit
        or metadata["image_token_budget"] != expected_budget
    ):
        raise ValueError("head/base code, feature, class or image budget mismatch")
    head = ExistenceHead(metadata["feature_dim"])
    expected = dict(tree_flatten(head.parameters()))
    weights = mx.load(str(directory / "head.safetensors"))
    if set(weights) != set(expected) or any(
        weights[k].shape != expected[k].shape for k in expected
    ):
        raise ValueError("head parameter names/shapes mismatch")
    if any(
        x.dtype != mx.float32 or not bool(mx.all(mx.isfinite(x)).item()) for x in weights.values()
    ):
        raise ValueError("head weights must be finite FP32")
    head.load_weights(list(weights.items()), strict=True)
    head.eval()
    return head, metadata
