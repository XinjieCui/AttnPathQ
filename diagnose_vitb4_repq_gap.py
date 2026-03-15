from __future__ import annotations

import argparse
import copy
import itertools
import math
from pathlib import Path

import torch
from torch import Tensor

from common import (
    attention_modules,
    build_loader,
    hadamard_transform,
    kurtosis,
    load_model,
    mse,
    quantize_with_scale,
    save_json,
    seed_everything,
)
from run_idea2_adapted_baselines import (
    MODEL_SPECS,
    collect_bundle_with_resources,
    evaluate_with_resources,
    mode_tensor_params,
    patch_qkv_method,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose why RepQ-style wins on ViT-B/16 4-bit")
    parser.add_argument("--dataset-root", type=str, default="/home/cxj/DyadicFold/data/imagenet")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/cxj/experiments/vit_quant_5ideas/results/idea2_analysis",
    )
    parser.add_argument("--run-name", type=str, default="vitb4_repq_gap_diagnostics")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="vit_base", choices=sorted(MODEL_SPECS))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--calib-images", type=int, default=128)
    parser.add_argument("--smoke-val-images", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-layers", type=int, default=4)
    return parser.parse_args()


def tensor_channel_stats(x: Tensor) -> dict[str, float]:
    flat = x.to(torch.float32).reshape(-1, x.shape[-1]).abs()
    q = torch.quantile(flat, 0.999, dim=0)
    mean_q = float(q.mean().item())
    median_q = float(q.median().item())
    scalar = flat.reshape(-1)
    if scalar.numel() > 2_000_000:
        step = max(1, math.ceil(scalar.numel() / 2_000_000))
        scalar = scalar[::step]
    return {
        "kurtosis": float(kurtosis(x)),
        "channel_scale_cv": float(q.std(unbiased=False).item() / max(mean_q, 1e-12)),
        "channel_scale_max_over_median": float(q.max().item() / max(median_q, 1e-12)),
        "abs_q999_over_abs_q95": float(
            torch.quantile(scalar, 0.999).item() / max(torch.quantile(scalar, 0.95).item(), 1e-12)
        ),
    }


def kl_rows(p: Tensor, q: Tensor) -> float:
    p32 = p.to(torch.float32).clamp_min(1e-8)
    q32 = q.to(torch.float32).clamp_min(1e-8)
    return float((p32 * (p32.log() - q32.log())).sum(dim=-1).mean().item())


def quantize_from_param(x: Tensor, bits: int, param: dict) -> Tensor:
    kind = param["kind"]
    scale = param["scale"]
    ratio = param.get("ratio")
    y = x.to(torch.float32)
    if kind == "rotated":
        y = hadamard_transform(y)
        y = quantize_with_scale(y, scale, bits)
        y = hadamard_transform(y)
        return y
    if kind == "rotated_repq":
        assert ratio is not None
        y = hadamard_transform(y)
        view = ratio.view(*([1] * (y.ndim - 1)), -1)
        y = quantize_with_scale(y / view, scale, bits) * view
        y = hadamard_transform(y)
        return y
    if kind == "repq":
        assert ratio is not None
        view = ratio.view(*([1] * (y.ndim - 1)), -1)
        return quantize_with_scale(y / view, scale, bits) * view
    return quantize_with_scale(y, scale, bits)


def layer_metric_record(
    payload: dict[str, Tensor],
    layer_param: dict[str, dict],
    bits: int,
    attn_scale: float,
) -> dict[str, float | dict[str, float]]:
    q = payload["q"].to(torch.float32)
    k = payload["k"].to(torch.float32)
    v = payload["v"].to(torch.float32)
    probs_fp = payload["attn_probs"].to(torch.float32)
    logits_fp = payload["attn_logits"].to(torch.float32)
    out_fp = probs_fp @ v

    q_hat = quantize_from_param(q, bits, layer_param["q"])
    k_hat = quantize_from_param(k, bits, layer_param["k"])
    v_hat = quantize_from_param(v, bits, layer_param["v"])

    logits_hat = (q_hat * attn_scale) @ k_hat.transpose(-2, -1)
    probs_hat = logits_hat.softmax(dim=-1)
    out_hat = probs_hat @ v_hat

    return {
        "q_mse": float(mse(q, q_hat)),
        "k_mse": float(mse(k, k_hat)),
        "v_mse": float(mse(v, v_hat)),
        "logits_mse": float(mse(logits_fp, logits_hat)),
        "probs_mse": float(mse(probs_fp, probs_hat)),
        "probs_kl": float(kl_rows(probs_fp, probs_hat)),
        "out_mse": float(mse(out_fp, out_hat)),
        "q_stats": tensor_channel_stats(q),
        "k_stats": tensor_channel_stats(k),
        "v_stats": tensor_channel_stats(v),
    }


def build_method_metrics(
    bundle: dict[int, dict[str, Tensor]],
    layer_params: dict[int, dict[str, dict]],
    bits: int,
    attn_scale: float,
) -> dict[int, dict[str, float | dict[str, float]]]:
    return {
        layer: layer_metric_record(payload, layer_params[layer], bits=bits, attn_scale=attn_scale)
        for layer, payload in bundle.items()
    }


def mean_of(values: list[float]) -> float:
    return float(sum(values) / len(values))


def summarize_method_gap(
    ours: dict[int, dict[str, float | dict[str, float]]],
    repq: dict[int, dict[str, float | dict[str, float]]],
) -> dict[str, object]:
    numeric_keys = ["q_mse", "k_mse", "v_mse", "logits_mse", "probs_mse", "probs_kl", "out_mse"]
    per_layer: dict[str, dict[str, float]] = {}
    for layer in ours:
        per_layer[str(layer)] = {}
        for key in numeric_keys:
            ours_value = float(ours[layer][key])
            repq_value = float(repq[layer][key])
            per_layer[str(layer)][f"ours_{key}"] = ours_value
            per_layer[str(layer)][f"repq_{key}"] = repq_value
            per_layer[str(layer)][f"repq_minus_ours_{key}"] = float(repq_value - ours_value)
            per_layer[str(layer)][f"repq_rel_gain_{key}"] = float((ours_value - repq_value) / max(ours_value, 1e-12))

    top_out_layers = sorted(
        range(len(ours)),
        key=lambda layer: per_layer[str(layer)]["repq_rel_gain_out_mse"],
        reverse=True,
    )
    top_q_layers = sorted(
        range(len(ours)),
        key=lambda layer: per_layer[str(layer)]["repq_rel_gain_q_mse"],
        reverse=True,
    )
    return {
        "per_layer": per_layer,
        "summary": {
            key: mean_of([per_layer[str(layer)][f"repq_rel_gain_{key}"] for layer in ours])
            for key in numeric_keys
        },
        "top_repq_out_layers": top_out_layers,
        "top_repq_q_layers": top_q_layers,
    }


def clone_layer_params(layer_params: dict[int, dict[str, dict]]) -> dict[int, dict[str, dict]]:
    return copy.deepcopy(layer_params)


def evaluate_mixed_case(
    model_name: str,
    checkpoint: str,
    bits: int,
    layer_params: dict[int, dict[str, dict]],
    val_loader,
    device: torch.device,
    tag: str,
) -> dict[str, float | int | None]:
    model = load_model(model_name, checkpoint, device=device)
    replaced = patch_qkv_method(model, bits=bits, layer_params=layer_params)
    stats = evaluate_with_resources(model, val_loader, device, tag)
    stats["replaced_layers"] = replaced
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec = MODEL_SPECS[args.model]

    calib_batch = min(spec["batch_size"], 16)
    calib_loader = build_loader(
        dataset_root=args.dataset_root,
        split="train",
        batch_size=calib_batch,
        num_workers=args.num_workers,
        num_images=args.calib_images,
        seed=args.seed + 17,
        shuffle=False,
        device=device,
    )
    val_loader = build_loader(
        dataset_root=args.dataset_root,
        split="val",
        batch_size=spec["batch_size"],
        num_workers=args.num_workers,
        num_images=args.smoke_val_images,
        seed=args.seed,
        shuffle=False,
        device=device,
    )

    model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    attn_scale = float(attention_modules(model)[0].scale)
    layers = list(range(len(attention_modules(model))))
    bundle, calib_stats = collect_bundle_with_resources(
        model,
        calib_loader,
        device=device,
        layers=layers,
        max_batches=max(1, math.ceil(args.calib_images / calib_batch)),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fp_model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    fp_stats = evaluate_with_resources(fp_model, val_loader, device, f"idea2_gap:{args.model}:fp_smoke")
    del fp_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    direct_params, _ = mode_tensor_params(bundle, args.bits, "direct_qkv")
    direct_stats = evaluate_mixed_case(
        spec["model_name"],
        spec["checkpoint"],
        bits=args.bits,
        layer_params=direct_params,
        val_loader=val_loader,
        device=device,
        tag=f"idea2_gap:{args.model}:direct",
    )
    ours_params, _ = mode_tensor_params(bundle, args.bits, "ours_rotated_qk")
    repq_params, _ = mode_tensor_params(bundle, args.bits, "repq_style_qkv")

    ours_stats = evaluate_mixed_case(
        spec["model_name"],
        spec["checkpoint"],
        bits=args.bits,
        layer_params=ours_params,
        val_loader=val_loader,
        device=device,
        tag=f"idea2_gap:{args.model}:ours",
    )
    repq_stats = evaluate_mixed_case(
        spec["model_name"],
        spec["checkpoint"],
        bits=args.bits,
        layer_params=repq_params,
        val_loader=val_loader,
        device=device,
        tag=f"idea2_gap:{args.model}:repq",
    )

    ours_metrics = build_method_metrics(bundle, ours_params, bits=args.bits, attn_scale=attn_scale)
    repq_metrics = build_method_metrics(bundle, repq_params, bits=args.bits, attn_scale=attn_scale)
    metric_gap = summarize_method_gap(ours_metrics, repq_metrics)

    swap_results = {
        "base": {
            "ours_top1": ours_stats["top1"],
            "repq_top1": repq_stats["top1"],
            "direct_top1": direct_stats["top1"],
            "fp_smoke_top1": fp_stats["top1"],
        },
        "ours_with_one_repq_layer": {},
        "repq_with_one_ours_layer": {},
    }
    for layer in layers:
        ours_mix = clone_layer_params(ours_params)
        ours_mix[layer] = copy.deepcopy(repq_params[layer])
        stats = evaluate_mixed_case(
            spec["model_name"],
            spec["checkpoint"],
            bits=args.bits,
            layer_params=ours_mix,
            val_loader=val_loader,
            device=device,
            tag=f"idea2_gap:{args.model}:ours_plus_repq_layer{layer}",
        )
        swap_results["ours_with_one_repq_layer"][str(layer)] = {
            "top1": stats["top1"],
            "delta_vs_ours": float(stats["top1"] - ours_stats["top1"]),
        }

        repq_mix = clone_layer_params(repq_params)
        repq_mix[layer] = copy.deepcopy(ours_params[layer])
        stats = evaluate_mixed_case(
            spec["model_name"],
            spec["checkpoint"],
            bits=args.bits,
            layer_params=repq_mix,
            val_loader=val_loader,
            device=device,
            tag=f"idea2_gap:{args.model}:repq_plus_ours_layer{layer}",
        )
        swap_results["repq_with_one_ours_layer"][str(layer)] = {
            "top1": stats["top1"],
            "delta_vs_repq": float(stats["top1"] - repq_stats["top1"]),
        }

    candidate_layers = metric_gap["top_repq_out_layers"][: args.top_layers]
    tensor_swap_results: dict[str, dict[str, dict[str, float]]] = {}
    tensor_groups = [
        ("q",),
        ("k",),
        ("v",),
        ("q", "k"),
        ("q", "v"),
        ("k", "v"),
        ("q", "k", "v"),
    ]
    for layer in candidate_layers:
        tensor_swap_results[str(layer)] = {}
        for tensor_names in tensor_groups:
            ours_mix = clone_layer_params(ours_params)
            for tensor_name in tensor_names:
                ours_mix[layer][tensor_name] = copy.deepcopy(repq_params[layer][tensor_name])
            stats = evaluate_mixed_case(
                spec["model_name"],
                spec["checkpoint"],
                bits=args.bits,
                layer_params=ours_mix,
                val_loader=val_loader,
                device=device,
                tag=f"idea2_gap:{args.model}:ours_plus_repq_{''.join(tensor_names)}_layer{layer}",
            )
            tensor_swap_results[str(layer)]["+".join(tensor_names)] = {
                "top1": stats["top1"],
                "delta_vs_ours": float(stats["top1"] - ours_stats["top1"]),
            }

    out = {
        "setup": {
            "model": args.model,
            "bits": args.bits,
            "seed": args.seed,
            "calib_images": args.calib_images,
            "smoke_val_images": args.smoke_val_images,
            "device": str(device),
        },
        "calibration": calib_stats,
        "smoke_top1": {
            "fp_smoke": fp_stats,
            "direct_qkv": direct_stats,
            "ours_rotated_qk": ours_stats,
            "repq_style_qkv": repq_stats,
        },
        "metric_gap": metric_gap,
        "swap_results": swap_results,
        "tensor_swap_results": tensor_swap_results,
    }
    out_path = Path(args.results_dir) / f"{args.run_name}.json"
    save_json(out_path, out)
    print(f"[Done] diagnostics saved to {out_path}")


if __name__ == "__main__":
    main()
