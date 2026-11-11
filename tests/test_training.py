"""Optional real MLX autodiff tests; skip on hosts without the baseline/MLX."""

from dataclasses import asdict

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_grounding_pi")
np = pytest.importorskip("numpy")

import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_grounding_pi.config import VisionConfig
from mlx_grounding_pi.model import Model

from rejection_lab.existence import (
    CANONICAL_FEATURE_VERSION,
    FEATURE_VERSION,
    ExistenceHead,
    feature_instruction,
    head_loss,
    load_head,
    prompt_feature,
    save_head,
)
from rejection_lab.training import (
    answer_loss,
    attach_lora,
    cache_teacher,
    gradient_summary,
    load_adapter,
    masked_teacher_kl,
    preservation_mask,
    protected_answer_loss,
    save_adapter,
    teacher_forcing,
    weighted_answer_loss,
)


@pytest.fixture
def tiny():
    mx.random.seed(11)
    vision = VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        qkv_hidden_size=16,
        patch_size=2,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        projector_hidden_size=32,
        out_hidden_size=32,
    )
    config = {
        "image_token_id": 31,
        "vision_config": asdict(vision),
        "text_config": {
            "model_type": "qwen3",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
            "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
        },
    }
    model = Model(config)
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: p.startswith("language_model") and isinstance(m, nn.Linear),
    )
    cfg = attach_lora(model, layers=1, rank=2, scale=2.0)
    sample = teacher_forcing([1, 31, 2], [4, 5], 3)
    sample["image_positions"] = [1]
    sample["features"] = mx.stop_gradient(model.visual(mx.random.normal((4, 3, 2, 2)), [[1, 2, 2]]))
    mx.eval(model.parameters(), sample["features"])
    return model, sample, cfg


def test_answer_shift_masks_prompt_and_supervises_eos():
    sample = teacher_forcing([1, 31, 2], [4, 5], 3)
    assert sample["input_ids"].tolist() == [[1, 31, 2, 4, 5]]
    assert sample["targets"].tolist() == [[4, 5, 3]]
    assert sample["answer_start"] == 2


def test_answer_only_loss_matches_full_sequence_mask(tiny):
    model, sample, _ = tiny
    ids = sample["input_ids"]
    embeddings = model.language_model.model.embed_tokens(ids)
    embeddings[0, mx.array(sample["image_positions"])] = sample["features"].astype(embeddings.dtype)
    logits = model.language_model(ids, input_embeddings=embeddings)
    start = sample["answer_start"]
    targets = mx.concatenate([mx.zeros((1, start), dtype=mx.int32), sample["targets"]], axis=1)
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    masked = mx.mean(losses[:, start:])
    assert abs(float(masked.item()) - float(answer_loss(model, sample).item())) < 1e-6


def test_zero_lora_layer_count_rejected(tiny):
    model, _, _ = tiny
    with pytest.raises(ValueError, match="invalid LoRA"):
        attach_lora(model, layers=0)


def test_quantized_multimodal_gradient_only_updates_lora(tiny):
    model, sample, _ = tiny
    frozen = {
        k: v for k, v in tree_flatten(model.parameters()) if not k.endswith(("lora_a", "lora_b"))
    }
    loss, grads = nn.value_and_grad(model, answer_loss)(model, sample)
    mx.eval(loss, grads)
    assert gradient_summary(grads)["finite"]
    before = float(loss.item())
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    assert abs(float(answer_loss(model, sample).item()) - before) > 1e-7
    after = dict(tree_flatten(model.parameters()))
    assert all(bool(mx.array_equal(v, after[k]).item()) for k, v in frozen.items())


def test_visual_condition_used_but_features_are_stop_gradient(tiny):
    model, sample, _ = tiny
    without = {**sample, "features": mx.zeros_like(sample["features"])}
    assert (
        abs(float(answer_loss(model, sample).item()) - float(answer_loss(model, without).item()))
        > 1e-7
    )
    gradient = mx.grad(lambda f: answer_loss(model, {**sample, "features": f}))(sample["features"])
    assert bool(mx.all(gradient == 0).item())


def test_positive_weight_scales_loss_and_gradient_without_changing_targets(tiny):
    model, sample, _ = tiny
    original, gradients = nn.value_and_grad(model, answer_loss)(model, sample)
    weighted, scaled = nn.value_and_grad(model, weighted_answer_loss)(
        model, {**sample, "loss_weight": 3.0}
    )
    mx.eval(original, weighted, gradients, scaled)
    assert abs(float(weighted.item()) - 3 * float(original.item())) < 1e-5
    first, second = dict(tree_flatten(gradients)), dict(tree_flatten(scaled))
    assert first.keys() == second.keys()
    assert all(
        bool(mx.allclose(second[k], 3 * first[k], rtol=5e-4, atol=1e-5).item()) for k in first
    )
    assert sample["targets"].tolist() == [[4, 5, 3]]


def test_default_weight_preserves_original_loss(tiny):
    model, sample, _ = tiny
    assert float(weighted_answer_loss(model, sample).item()) == float(
        answer_loss(model, sample).item()
    )


@pytest.mark.parametrize("weight", [0, float("nan")])
def test_invalid_loss_weight_rejected(tiny, weight):
    model, sample, _ = tiny
    with pytest.raises(ValueError, match="finite positive"):
        weighted_answer_loss(model, {**sample, "loss_weight": weight})


def test_preservation_mask_keeps_delimiters_eos_but_not_negative_payload():
    targets = [10, 11, 12, 13, 20, 21, 14, 99]
    kw = {"box_start_id": 13, "box_end_id": 14}
    assert preservation_mask(targets, present=False, **kw) == [1, 1, 1, 1, 0, 0, 1, 1]
    assert preservation_mask(targets, present=True, **kw) == [1] * 8


def test_positive_only_scope_excludes_all_negative_distillation():
    targets = [10, 13, 20, 14, 99]
    kw = {"box_start_id": 13, "box_end_id": 14, "negative_scope": "none"}
    assert preservation_mask(targets, present=False, **kw) == [0] * len(targets)
    assert preservation_mask(targets, present=True, **kw) == [1] * len(targets)


def test_unknown_negative_distillation_scope_rejected():
    with pytest.raises(ValueError, match="scope"):
        preservation_mask(
            [13, 20, 14], present=False, box_start_id=13, box_end_id=14, negative_scope="bad"
        )


@pytest.mark.parametrize("targets", [[13, 20], [14, 20, 13], [13, 14]])
def test_bad_protocol_mask_rejected(targets):
    with pytest.raises(ValueError):
        preservation_mask(targets, present=False, box_start_id=13, box_end_id=14)


def test_masked_kl_matches_numpy_and_detaches_teacher_and_payload():
    logits = mx.array([[[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]]])
    teacher = mx.log(mx.array([[[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]]))
    mask = mx.array([[1.0, 0.0]])
    q = np.exp(np.array([1.0, 2.0, 3.0]) - 3)
    q /= q.sum()
    p = np.array([0.2, 0.3, 0.5])
    expected = np.sum(p * (np.log(p) - np.log(q)))
    assert abs(float(masked_teacher_kl(logits, teacher, mask).item()) - expected) < 1e-6
    gradient = mx.grad(lambda x: masked_teacher_kl(x, teacher, mask))(logits)
    assert bool(mx.all(gradient[:, 1] == 0).item())
    assert float(mx.sum(mx.abs(gradient[:, 0])).item()) > 0
    teacher_grad = mx.grad(lambda x: masked_teacher_kl(logits, x, mask))(teacher)
    assert bool(mx.all(teacher_grad == 0).item())


def test_cached_teacher_kl_zero_then_positive_and_cache_unchanged(tiny):
    model, sample, _ = tiny
    cache_teacher(model, sample, [1, 0, 1], strength=1.0)
    teacher = sample["teacher_log_probs"]
    (_, (ce, kl)), grads = nn.value_and_grad(model, protected_answer_loss)(model, sample)
    mx.eval(ce, kl, grads)
    assert abs(float(kl.item())) < 1e-6
    assert gradient_summary(grads)["finite"]
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    (_, (_, after_kl)) = protected_answer_loss(model, sample)
    assert float(after_kl.item()) > 1e-7
    assert bool(mx.array_equal(teacher, sample["teacher_log_probs"]).item())


def test_zero_strength_objective_and_gradient_match_weighted_ce(tiny):
    model, sample, _ = tiny
    sample = {**sample, "loss_weight": 3.0}
    a, ga = nn.value_and_grad(model, weighted_answer_loss)(model, sample)
    (b, (_, kl)), gb = nn.value_and_grad(model, protected_answer_loss)(model, sample)
    mx.eval(a, b, ga, gb, kl)
    assert float(a.item()) == float(b.item())
    assert float(kl.item()) == 0
    ga, gb = dict(tree_flatten(ga)), dict(tree_flatten(gb))
    assert all(bool(mx.array_equal(ga[k], gb[k]).item()) for k in ga)


@pytest.mark.parametrize("mask", [[0, 0, 0], [1], [1, 0.5, 1]])
def test_bad_teacher_masks_rejected(tiny, mask):
    model, sample, _ = tiny
    with pytest.raises(ValueError, match="mask"):
        cache_teacher(model, sample, mask, strength=1.0)


def test_frozen_prompt_feature_uses_query_and_image_and_stops_gradients(tiny):
    model, _, _ = tiny
    model.freeze()
    before = dict(tree_flatten(model.parameters()))
    inputs = {
        "input_ids": mx.array([[1, 31, 2]]),
        "pixel_values": mx.random.normal((4, 3, 2, 2)),
        "image_grid_thw": [[1, 2, 2]],
    }
    visual = model.visual(inputs["pixel_values"], inputs["image_grid_thw"])
    feature = prompt_feature(model, inputs, visual_features=visual)
    changed_query = prompt_feature(
        model, {**inputs, "input_ids": mx.array([[3, 31, 2]])}, visual_features=visual
    )
    changed_image = prompt_feature(model, inputs, visual_features=mx.zeros_like(visual))
    assert abs(float(mx.sum(feature**2).item()) - 1) < 1e-5
    assert float(mx.max(mx.abs(feature - changed_query)).item()) > 1e-6
    assert float(mx.max(mx.abs(feature - changed_image)).item()) > 1e-6
    gradient = mx.grad(lambda f: mx.sum(prompt_feature(model, inputs, visual_features=f)))(visual)
    assert bool(mx.all(gradient == 0).item())
    after = dict(tree_flatten(model.parameters()))
    assert all(bool(mx.array_equal(v, after[k]).item()) for k, v in before.items())
    assert not tree_flatten(model.trainable_parameters())


def test_independent_head_learns_toy_classes_without_feature_gradient():
    head = ExistenceHead(2)
    features = mx.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    labels = mx.array([1, 0, 0, 1])
    optimizer = optim.Adam(learning_rate=0.05)
    first = float(head_loss(head, features, labels)[0].item())
    for _ in range(16):
        (_, _), grads = nn.value_and_grad(head, head_loss)(head, features, labels, 0.01)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)
    assert float(head_loss(head, features, labels)[0].item()) < first
    assert mx.argmax(head(features), axis=-1).tolist() == labels.tolist()
    gradient = mx.grad(lambda f: head_loss(head, f, labels)[0])(features)
    assert bool(mx.all(gradient == 0).item())


def test_cost_sensitive_head_loss_matches_weighted_ce_and_keeps_default():
    head = ExistenceHead(2)
    head.projection.weight = mx.array([[0.2, -0.3], [-0.4, 0.1]])
    x = mx.array([[1.0, 0.0], [0.0, 1.0]])
    y = mx.array([0, 1])
    ce = nn.losses.cross_entropy(head(x), y, reduction="none")
    objective, _ = head_loss(head, x, y, 0.0, 3.0)
    assert abs(float(objective.item()) - float(((ce[0] + 3 * ce[1]) / 4).item())) < 1e-6
    original, _ = head_loss(head, x, y, 0.01)
    expected = mx.mean(ce) + 0.005 * mx.sum(head.projection.weight**2)
    assert float(original.item()) == float(expected.item())


def test_invalid_existence_class_cost_rejected():
    head = ExistenceHead(2)
    with pytest.raises(ValueError, match="class weight"):
        head_loss(head, mx.zeros((2, 2)), mx.array([0, 1]), 0.01, 0)


def test_canonical_feature_instruction_is_task_independent():
    phrase = "the red square"
    point = f"Point to the target referred to by the following description: {phrase}."
    box = f"Locate the target referred to by the following description: {phrase}."
    assert feature_instruction(phrase, point) == point
    assert feature_instruction(phrase, box) == box
    assert feature_instruction(phrase, point, "canonical") == feature_instruction(
        phrase, box, "canonical"
    )
    assert feature_instruction("the blue circle", point, "canonical") != feature_instruction(
        phrase, point, "canonical"
    )
    for bad in ["", "   ", "<|box_start|>", None]:
        with pytest.raises(ValueError, match="phrase"):
            feature_instruction(bad, point, "canonical")
    with pytest.raises(ValueError, match="mode"):
        feature_instruction(phrase, point, "unknown")


@pytest.mark.parametrize("feature_version", [FEATURE_VERSION, CANONICAL_FEATURE_VERSION])
def test_head_roundtrip_and_metadata_and_nonfinite_rejection(tmp_path, feature_version):
    head = ExistenceHead(2)
    head.projection.weight = mx.array([[1.0, -1.0], [-1.0, 1.0]])
    metadata = {
        "format": "grounding-pi-existence-head-v1",
        "feature_version": feature_version,
        "feature_dim": 2,
        "classes": ["absent", "present"],
        "base_fingerprint": "tiny",
        "baseline_commit": "abc",
        "image_token_budget": 64,
    }
    directory = tmp_path / "head"
    save_head(head, directory, metadata)
    kwargs = {
        "expected_fingerprint": "tiny",
        "expected_commit": "abc",
        "expected_budget": 64,
        "expected_feature_version": feature_version,
    }
    restored, _ = load_head(directory, **kwargs)
    assert bool(mx.array_equal(restored.projection.weight, head.projection.weight).item())
    for change in [
        {"expected_fingerprint": "other"},
        {"expected_commit": "other"},
        {"expected_budget": 32},
        {
            "expected_feature_version": CANONICAL_FEATURE_VERSION
            if feature_version == FEATURE_VERSION
            else FEATURE_VERSION
        },
    ]:
        with pytest.raises(ValueError, match="mismatch"):
            load_head(directory, **{**kwargs, **change})
    with pytest.raises(ValueError, match="version"):
        load_head(directory, **{**kwargs, "expected_feature_version": "unknown"})
    weights = mx.load(str(directory / "head.safetensors"))
    weights["projection.weight"] = mx.full((2, 2), float("nan"))
    mx.save_safetensors(str(directory / "head.safetensors"), weights)
    with pytest.raises(ValueError, match="finite"):
        load_head(directory, **kwargs)


def test_adapter_roundtrip_and_base_mismatch(tiny, tmp_path):
    model, sample, cfg = tiny
    _, grads = nn.value_and_grad(model, answer_loss)(model, sample)
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    expected = float(answer_loss(model, sample).item())
    directory = tmp_path / "adapter"
    save_adapter(model, directory, {"lora": cfg, "base_fingerprint": "tiny-test"})
    arrays = mx.load(str(directory / "adapters.safetensors"))
    assert all(k.endswith((".lora_a", ".lora_b")) for k in arrays)
    model.update(tree_unflatten([(k, mx.zeros_like(v)) for k, v in arrays.items()]))
    load_adapter(model, directory, expected_fingerprint="tiny-test")
    assert abs(float(answer_loss(model, sample).item()) - expected) < 1e-6
    with pytest.raises(ValueError, match="fingerprint"):
        load_adapter(model, directory, expected_fingerprint="different-base")
