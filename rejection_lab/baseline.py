"""Optional local MLX evaluation through a trusted sibling baseline checkout."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .data import validate_dataset
from .metrics import grouped_score


def normalize_result(result, phrase):
    entries = result.get("predictions", [])
    if not result.get("valid") or len(entries) != 1 or entries[0]["label"] != phrase:
        return {"valid": False, "coordinates": None}
    coords = entries[0]["coordinates"]
    if coords is None:
        return {"valid": True, "coordinates": None}
    if len(coords) != 1:
        return {"valid": False, "coordinates": None}
    return {"valid": True, "coordinates": list(coords[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter", type=Path, help="Optional locally trained lab adapter directory"
    )
    parser.add_argument("--image-tokens", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    validate_dataset(args.dataset)
    repo = args.baseline_repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError("baseline checkout must be clean for a reproducible reference")
    sys.path.insert(0, str(repo))
    import mlx.core as mx
    import mlx_grounding_pi
    from mlx_grounding_pi.generate import greedy_generate, parse_result
    from mlx_grounding_pi.loading import load_model
    from mlx_grounding_pi.processing import Processor

    if not Path(mlx_grounding_pi.__file__).resolve().is_relative_to(repo):
        raise ValueError("baseline imported from an unexpected checkout")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    recommended = mx.device_info().get("max_recommended_working_set_size", 0)
    if recommended and index["metadata"]["total_size"] > recommended * 0.75:
        raise ValueError("model exceeds conservative MLX memory preflight")
    truth_path = args.dataset / f"{args.split}.jsonl"
    rows = [json.loads(line) for line in truth_path.read_text().splitlines()]
    adapter_metadata = None
    if args.adapter:
        adapter_metadata = json.loads((args.adapter / "adapter_config.json").read_text())
        if adapter_metadata["baseline_commit"] != commit:
            raise ValueError("adapter/baseline code commit mismatch")
        trained_budget = adapter_metadata["image_token_budget"]
        if args.image_tokens is not None and args.image_tokens != trained_budget:
            raise ValueError("adapter evaluation must use the trained image-token budget")
        args.image_tokens = trained_budget
    processor = Processor(args.model, max_output_tokens=args.image_tokens)
    model, config = load_model(args.model)
    if args.adapter:
        from .training import attach_lora, load_adapter, model_fingerprint

        lora = adapter_metadata["lora"]
        attach_lora(model, layers=lora["layers"], rank=lora["rank"], scale=lora["scale"])
        load_adapter(model, args.adapter, expected_fingerprint=model_fingerprint(args.model))
    model.eval()
    args.output.mkdir(parents=True)
    predictions = []
    with (args.output / "predictions.jsonl").open("x") as stream:
        for row in rows:
            inputs = processor(args.dataset / row["image"], row["instruction"])
            tokens, finish, timing = greedy_generate(
                model,
                inputs,
                eos_token_id=config["eos_token_id"],
                max_tokens=128,
                max_context=min(4096, config["text_config"]["max_position_embeddings"]),
            )
            text = processor.tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            parsed = parse_result(text, row["task"], inputs["image"].original_size, finish)
            pred = {
                "id": row["id"],
                **normalize_result(parsed, row["phrase"]),
                "raw_output": text,
                "finish_reason": finish,
                "timing": timing,
            }
            predictions.append(pred)
            stream.write(json.dumps(pred) + "\n")
            stream.flush()
    report = grouped_score(rows, predictions)
    report["baseline_commit"] = commit
    report["adapter"] = adapter_metadata
    report["image_token_budget"] = args.image_tokens
    report["split"] = args.split
    report["dataset_manifest"] = json.loads((args.dataset / "manifest.json").read_text())
    report["model_conversion"] = json.loads((args.model / "conversion.json").read_text())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
