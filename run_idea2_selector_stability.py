from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from common import attention_modules, build_loader, load_model, save_json, seed_everything
from run_idea2_adapted_baselines import (
    MODEL_SPECS,
    apply_hparam_overrides,
    build_auto_rescue_selector,
    build_auto_selector,
    collect_bundle_with_resources,
    collect_model_logits,
    mode_tensor_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selector stability sweep for AttnPathQ")
    parser.add_argument("--dataset-root", type=str, default="/home/cxj/DyadicFold/data/imagenet")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/cxj/experiments/vit_quant_5ideas/results/idea2_adapted",
    )
    parser.add_argument("--run-name", type=str, default="selector_stability")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--models", type=str, nargs="+", default=["deit_small", "vit_base", "deit_tiny"])
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--calib-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--auto-qkv-margin", type=float, default=None)
    parser.add_argument("--rescue-max-layers", type=int, default=None)
    parser.add_argument("--rescue-min-layer-score", type=float, default=None)
    parser.add_argument("--rescue-min-total-score", type=float, default=None)
    parser.add_argument("--rescue-layer-penalty", type=float, default=None)
    parser.add_argument("--rescue-min-rel-improvement", type=float, default=None)
    return parser.parse_args()


def summarize_model(records: dict[str, dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for seed_record in records.values():
        for calib_record in seed_record["calibrations"].values():
            for bits, bit_record in calib_record["bits"].items():
                bucket = summary.setdefault(bits, {
                    "mode_counts": {},
                    "rescue_trigger_count": 0,
                    "total_cases": 0,
                    "selected_layer_patterns": {},
                })
                mode = bit_record["selected_mode"]
                bucket["mode_counts"][mode] = bucket["mode_counts"].get(mode, 0) + 1
                bucket["total_cases"] += 1
                if bit_record["rescue_triggered"]:
                    bucket["rescue_trigger_count"] += 1
                layer_pattern = ",".join(str(layer) for layer in bit_record["selected_layers"]) or "none"
                bucket["selected_layer_patterns"][layer_pattern] = bucket["selected_layer_patterns"].get(layer_pattern, 0) + 1
    return summary


def main() -> None:
    args = parse_args()
    hparams = apply_hparam_overrides(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {
        "setup": {
            "dataset_root": args.dataset_root,
            "device": str(device),
            "models": args.models,
            "bits": args.bits,
            "seeds": args.seeds,
            "calib_sizes": args.calib_sizes,
            "num_workers": args.num_workers,
            "run_name": args.run_name,
            "hparams": hparams,
        },
        "models": {},
    }

    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        batch_size = spec["batch_size"]
        model_records: dict[str, dict] = {}
        for seed in args.seeds:
            seed_everything(seed)
            seed_record = {"calibrations": {}}
            for calib_images in args.calib_sizes:
                calib_loader = build_loader(
                    dataset_root=args.dataset_root,
                    split="train",
                    batch_size=min(batch_size, 16),
                    num_workers=args.num_workers,
                    num_images=calib_images,
                    seed=seed + 17,
                    shuffle=False,
                    device=device,
                )
                calib_model = load_model(spec["model_name"], spec["checkpoint"], device=device)
                layers = list(range(len(attention_modules(calib_model))))
                attn_scale = float(attention_modules(calib_model)[0].scale)
                bundle, calib_stats = collect_bundle_with_resources(
                    calib_model,
                    calib_loader,
                    device=device,
                    layers=layers,
                    max_batches=max(1, math.ceil(calib_images / min(batch_size, 16))),
                )
                fp_calib_logits = collect_model_logits(calib_model, calib_loader, device)
                del calib_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                calib_record = {
                    "calibration": {
                        "num_images": calib_images,
                        "elapsed_sec": calib_stats["elapsed_sec"],
                        "peak_mem_mib": calib_stats["peak_mem_mib"],
                    },
                    "bits": {},
                }
                for bits in args.bits:
                    selector_start = time.perf_counter()
                    auto_selector = build_auto_selector(bundle, bits=bits, attn_scale=attn_scale)
                    selector_elapsed = float(time.perf_counter() - selector_start)
                    base_mode = auto_selector["chosen_mode"]
                    base_params, _ = mode_tensor_params(bundle, bits=bits, mode=base_mode)
                    repq_params, _ = mode_tensor_params(bundle, bits=bits, mode="repq_style_qkv")
                    rescue_start = time.perf_counter()
                    rescue_selector = build_auto_rescue_selector(
                        bundle,
                        bits=bits,
                        attn_scale=attn_scale,
                        base_mode=base_mode,
                        base_params=base_params,
                        repq_params=repq_params,
                        model_name=spec["model_name"],
                        checkpoint=spec["checkpoint"],
                        calib_loader=calib_loader,
                        device=device,
                        fp_calib_logits=fp_calib_logits,
                    )
                    rescue_elapsed = float(time.perf_counter() - rescue_start)
                    calib_record["bits"][str(bits)] = {
                        "selected_mode": base_mode,
                        "rescue_enabled": bool(rescue_selector["enabled"]),
                        "rescue_triggered": bool(rescue_selector["triggered"]),
                        "selected_layers": rescue_selector["selected_layers"],
                        "selected_overrides": rescue_selector["selected_overrides"],
                        "margin_early_v_minus_early_qk": auto_selector["decision"]["margin_early_v_minus_early_qk"],
                        "selector_elapsed_sec": selector_elapsed,
                        "rescue_selector_elapsed_sec": rescue_elapsed,
                        "best_rel_improvement": float(rescue_selector.get("best_rel_improvement", 0.0)),
                    }
                seed_record["calibrations"][str(calib_images)] = calib_record
            model_records[str(seed)] = seed_record
        results["models"][model_key] = {
            "per_seed": model_records,
            "summary": summarize_model(model_records),
        }

    out_path = Path(args.results_dir) / f"{args.run_name}.json"
    save_json(out_path, results)
    print(f"[Done] selector stability saved to {out_path}")


if __name__ == "__main__":
    main()
