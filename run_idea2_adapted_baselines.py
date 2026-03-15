from __future__ import annotations

import argparse
import copy
import itertools
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from common import (
    attention_modules,
    build_loader,
    collect_attention_bundle,
    evaluate_classification,
    estimate_percentile_scale,
    hadamard_transform,
    load_model,
    mse,
    quantize_with_scale,
    save_json,
    seed_everything,
)


MODEL_SPECS = {
    "deit_tiny": {
        "model_name": "deit_tiny_patch16_224",
        "checkpoint": None,
        "batch_size": 128,
    },
    "deit_base": {
        "model_name": "deit_base_patch16_224",
        "checkpoint": None,
        "batch_size": 64,
    },
    "deit_small": {
        "model_name": "deit_small_patch16_224",
        "checkpoint": "/home/cxj/cdfquant/weights/model.safetensors",
        "batch_size": 128,
    },
    "deit3_small": {
        "model_name": "deit3_small_patch16_224.fb_in22k_ft_in1k",
        "checkpoint": None,
        "batch_size": 128,
    },
    "deit3_base": {
        "model_name": "deit3_base_patch16_224.fb_in22k_ft_in1k",
        "checkpoint": None,
        "batch_size": 64,
    },
    "vit_small": {
        "model_name": "vit_small_patch16_224",
        "checkpoint": None,
        "batch_size": 128,
    },
    "vit_base": {
        "model_name": "vit_base_patch16_224",
        "checkpoint": "/home/cxj/cdfquant/weights/vit_b16.safetensors",
        "batch_size": 64,
    },
    "vit_base32": {
        "model_name": "vit_base_patch32_224.augreg_in21k_ft_in1k",
        "checkpoint": None,
        "batch_size": 64,
    },
}

AUTO_QKV_MARGIN = 0.03
AUTO_RESCUE_MAX_LAYERS = 3
AUTO_RESCUE_MIN_LAYER_SCORE = 0.01
AUTO_RESCUE_MIN_TOTAL_SCORE = 0.02
AUTO_RESCUE_LAYER_PENALTY = 0.004
AUTO_RESCUE_MIN_REL_IMPROVEMENT = 0.005
AUTO_RESCUE_TOKEN_CHOICES = {
    "kv": ("k", "v"),
    "qkv": ("q", "k", "v"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-setting adapted baselines for idea2")
    parser.add_argument("--dataset-root", type=str, default="/home/cxj/DyadicFold/data/imagenet")
    parser.add_argument("--eval-dataset-root", type=str, default=None)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/cxj/experiments/vit_quant_5ideas/results/idea2_adapted",
    )
    parser.add_argument("--run-name", type=str, default="main_full")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--models", type=str, nargs="+", default=["deit_small", "vit_base"])
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[
            "direct_qkv",
            "ptq4vit_style_qkv",
            "repq_style_qkv",
            "ours_rotated_qk",
            "ours_rotated_qkv",
        ],
    )
    parser.add_argument("--val-images", type=int, default=50000)
    parser.add_argument("--calib-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--auto-qkv-margin", type=float, default=None)
    parser.add_argument("--rescue-max-layers", type=int, default=None)
    parser.add_argument("--rescue-min-layer-score", type=float, default=None)
    parser.add_argument("--rescue-min-total-score", type=float, default=None)
    parser.add_argument("--rescue-layer-penalty", type=float, default=None)
    parser.add_argument("--rescue-min-rel-improvement", type=float, default=None)
    return parser.parse_args()


def apply_hparam_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    global AUTO_QKV_MARGIN
    global AUTO_RESCUE_MAX_LAYERS
    global AUTO_RESCUE_MIN_LAYER_SCORE
    global AUTO_RESCUE_MIN_TOTAL_SCORE
    global AUTO_RESCUE_LAYER_PENALTY
    global AUTO_RESCUE_MIN_REL_IMPROVEMENT

    if args.auto_qkv_margin is not None:
        AUTO_QKV_MARGIN = float(args.auto_qkv_margin)
    if args.rescue_max_layers is not None:
        AUTO_RESCUE_MAX_LAYERS = int(args.rescue_max_layers)
    if args.rescue_min_layer_score is not None:
        AUTO_RESCUE_MIN_LAYER_SCORE = float(args.rescue_min_layer_score)
    if args.rescue_min_total_score is not None:
        AUTO_RESCUE_MIN_TOTAL_SCORE = float(args.rescue_min_total_score)
    if args.rescue_layer_penalty is not None:
        AUTO_RESCUE_LAYER_PENALTY = float(args.rescue_layer_penalty)
    if args.rescue_min_rel_improvement is not None:
        AUTO_RESCUE_MIN_REL_IMPROVEMENT = float(args.rescue_min_rel_improvement)

    return {
        "auto_qkv_margin": AUTO_QKV_MARGIN,
        "rescue_max_layers": AUTO_RESCUE_MAX_LAYERS,
        "rescue_min_layer_score": AUTO_RESCUE_MIN_LAYER_SCORE,
        "rescue_min_total_score": AUTO_RESCUE_MIN_TOTAL_SCORE,
        "rescue_layer_penalty": AUTO_RESCUE_LAYER_PENALTY,
        "rescue_min_rel_improvement": AUTO_RESCUE_MIN_REL_IMPROVEMENT,
    }


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_peak_mem_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def evaluate_with_resources(model: nn.Module, val_loader, device: torch.device, tag: str) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync_cuda(device)
    start = time.perf_counter()
    stats = evaluate_classification(model, val_loader, device, tag)
    sync_cuda(device)
    elapsed = time.perf_counter() - start
    return {
        "top1": stats.top1,
        "top5": stats.top5,
        "loss": stats.loss,
        "num_images": stats.num_images,
        "elapsed_sec": float(elapsed),
        "peak_mem_mib": get_peak_mem_mib(device),
    }


def collect_model_logits(model: nn.Module, loader, device: torch.device, max_batches: int | None = None) -> Tensor:
    outputs: list[Tensor] = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            outputs.append(logits.detach().to(torch.float32).cpu())
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    return torch.cat(outputs, dim=0)


def collect_bundle_with_resources(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    layers: list[int],
    max_batches: int,
) -> tuple[dict[int, dict[str, Tensor]], dict[str, float | None]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync_cuda(device)
    start = time.perf_counter()
    bundle = collect_attention_bundle(
        model,
        calib_loader,
        device=device,
        layers=layers,
        max_batches=max_batches,
        collect_qkv=True,
    )
    sync_cuda(device)
    elapsed = time.perf_counter() - start
    return bundle, {
        "elapsed_sec": float(elapsed),
        "peak_mem_mib": get_peak_mem_mib(device),
    }


def sample_flat(x: Tensor, max_samples: int = 200_000) -> Tensor:
    flat = x.to(torch.float32).reshape(-1)
    if flat.numel() > max_samples:
        step = max(1, math.ceil(flat.numel() / max_samples))
        flat = flat[::step]
    return flat


def channelwise_symmetric_scale(
    x: Tensor,
    bits: int,
    percentile: float = 0.999,
    max_rows: int = 200_000,
) -> Tensor:
    qmax = float((1 << (bits - 1)) - 1)
    flat = x.to(torch.float32).reshape(-1, x.shape[-1]).abs()
    if flat.shape[0] > max_rows:
        step = max(1, math.ceil(flat.shape[0] / max_rows))
        flat = flat[::step]
    scale = torch.quantile(flat, percentile, dim=0)
    return torch.clamp(scale / qmax, min=torch.finfo(torch.float32).tiny)


def search_scalar_scale(
    x: Tensor,
    bits: int,
    percentiles: tuple[float, ...] = (0.995, 0.999, 0.9995),
    multipliers: tuple[float, ...] = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5),
) -> Tensor:
    ref = sample_flat(x)
    best_scale = estimate_percentile_scale(ref, bits=bits, percentile=0.999)
    best_err = float("inf")
    for percentile in percentiles:
        base = estimate_percentile_scale(ref, bits=bits, percentile=percentile)
        for multiplier in multipliers:
            scale = torch.clamp(base * multiplier, min=torch.finfo(torch.float32).tiny)
            err = mse(ref, quantize_with_scale(ref, scale, bits))
            if err < best_err:
                best_err = err
                best_scale = scale
    return best_scale


def build_repq_params(x: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    channel_scale = channelwise_symmetric_scale(x, bits=bits)
    target_scale = torch.clamp(channel_scale.mean(), min=torch.finfo(torch.float32).tiny)
    ratio = torch.clamp(channel_scale / target_scale, min=0.125, max=8.0)
    return target_scale.reshape(()), ratio


def tensor_proxy_record(x: Tensor, bits: int, scale: Tensor, kind: str, ratio: Tensor | None = None) -> dict:
    if kind == "rotated":
        y = hadamard_transform(x)
        q = hadamard_transform(quantize_with_scale(y, scale, bits))
    elif kind == "rotated_repq":
        assert ratio is not None
        y = hadamard_transform(x).to(torch.float32)
        view = ratio.view(*([1] * (y.ndim - 1)), -1)
        q = hadamard_transform(quantize_with_scale(y / view, scale, bits) * view)
    elif kind == "repq":
        assert ratio is not None
        view = ratio.view(*([1] * (x.ndim - 1)), -1)
        q = quantize_with_scale(x.to(torch.float32) / view, scale, bits) * view
    else:
        q = quantize_with_scale(x, scale, bits)
    return {"mse": mse(x, q)}


def quantize_direct_or_rotated(
    x: Tensor,
    bits: int,
    direct_scale: Tensor,
    rotated_scale: Tensor,
    rotate: bool,
) -> Tensor:
    if rotate:
        rotated = hadamard_transform(x)
        return hadamard_transform(quantize_with_scale(rotated, rotated_scale, bits))
    return quantize_with_scale(x, direct_scale, bits)


def build_auto_selector(bundle: dict[int, dict[str, Tensor]], bits: int, attn_scale: float) -> dict:
    layer_metrics: dict[int, dict[str, float]] = {}
    num_layers = len(bundle)
    split = max(1, num_layers // 2)
    early_layers = list(range(split))
    late_layers = list(range(split, num_layers))

    for layer, payload in bundle.items():
        q = payload["q"].to(torch.float32)
        k = payload["k"].to(torch.float32)
        v = payload["v"].to(torch.float32)
        probs_fp = payload["attn_probs"].to(torch.float32)

        direct_q = estimate_percentile_scale(q, bits=bits)
        direct_k = estimate_percentile_scale(k, bits=bits)
        direct_v = estimate_percentile_scale(v, bits=bits)
        rot_q = estimate_percentile_scale(hadamard_transform(q), bits=bits)
        rot_k = estimate_percentile_scale(hadamard_transform(k), bits=bits)
        rot_v = estimate_percentile_scale(hadamard_transform(v), bits=bits)

        q_direct = quantize_direct_or_rotated(q, bits, direct_q, rot_q, rotate=False)
        q_rot = quantize_direct_or_rotated(q, bits, direct_q, rot_q, rotate=True)
        k_direct = quantize_direct_or_rotated(k, bits, direct_k, rot_k, rotate=False)
        k_rot = quantize_direct_or_rotated(k, bits, direct_k, rot_k, rotate=True)
        v_direct = quantize_direct_or_rotated(v, bits, direct_v, rot_v, rotate=False)
        v_rot = quantize_direct_or_rotated(v, bits, direct_v, rot_v, rotate=True)

        prob_direct = ((q_direct * attn_scale) @ k_direct.transpose(-2, -1)).softmax(dim=-1)
        prob_rot = ((q_rot * attn_scale) @ k_rot.transpose(-2, -1)).softmax(dim=-1)
        qk_direct_mse = mse(probs_fp, prob_direct)
        qk_rot_mse = mse(probs_fp, prob_rot)

        out_fp = probs_fp @ v
        out_direct = probs_fp @ v_direct
        out_rot = probs_fp @ v_rot
        v_direct_mse = mse(out_fp, out_direct)
        v_rot_mse = mse(out_fp, out_rot)

        layer_metrics[layer] = {
            "qk_prob_gain": float((qk_direct_mse - qk_rot_mse) / max(qk_direct_mse, 1e-12)),
            "v_out_gain": float((v_direct_mse - v_rot_mse) / max(v_direct_mse, 1e-12)),
            "qk_prob_direct_mse": float(qk_direct_mse),
            "qk_prob_rot_mse": float(qk_rot_mse),
            "v_out_direct_mse": float(v_direct_mse),
            "v_out_rot_mse": float(v_rot_mse),
        }

    mean_qk_gain = float(sum(layer_metrics[layer]["qk_prob_gain"] for layer in range(num_layers)) / num_layers)
    early_qk_gain = float(sum(layer_metrics[layer]["qk_prob_gain"] for layer in early_layers) / len(early_layers))
    early_v_gain = float(sum(layer_metrics[layer]["v_out_gain"] for layer in early_layers) / len(early_layers))
    late_v_gain = float(sum(layer_metrics[layer]["v_out_gain"] for layer in late_layers) / max(len(late_layers), 1))
    late_qk_gain = float(sum(layer_metrics[layer]["qk_prob_gain"] for layer in late_layers) / max(len(late_layers), 1))

    early_margin = float(early_v_gain - early_qk_gain)

    if mean_qk_gain <= 0.0:
        chosen_mode = "direct_qkv"
    elif (early_qk_gain <= 0.0 and early_v_gain > 0.0) or early_margin > AUTO_QKV_MARGIN:
        chosen_mode = "ours_rotated_qkv"
    else:
        chosen_mode = "ours_rotated_qk"
    return {
        "chosen_mode": chosen_mode,
        "decision": {
            "mean_qk_prob_gain": mean_qk_gain,
            "early_qk_prob_gain": early_qk_gain,
            "late_qk_prob_gain": late_qk_gain,
            "early_v_out_gain": early_v_gain,
            "late_v_out_gain": late_v_gain,
            "criterion": "choose_qkv_if_early_qk_is_nonpositive_and_early_v_positive_or_if_early_v_minus_early_qk_exceeds_margin_else_qk",
            "auto_qkv_margin": AUTO_QKV_MARGIN,
            "margin_early_v_minus_early_qk": early_margin,
            "margin_early_v_minus_mean_qk": float(early_v_gain - mean_qk_gain),
        },
        "layer_metrics": {str(layer): layer_metrics[layer] for layer in layer_metrics},
        "early_layers": early_layers,
        "late_layers": late_layers,
    }


def clone_layer_params(layer_params: dict[int, dict[str, dict]]) -> dict[int, dict[str, dict]]:
    return copy.deepcopy(layer_params)


def apply_overrides(
    base_params: dict[int, dict[str, dict]],
    ref_params: dict[int, dict[str, dict]],
    overrides: dict[int, tuple[str, ...]],
) -> dict[int, dict[str, dict]]:
    mixed = clone_layer_params(base_params)
    for layer, tensor_names in overrides.items():
        for tensor_name in tensor_names:
            mixed[layer][tensor_name] = copy.deepcopy(ref_params[layer][tensor_name])
    return mixed


def quantize_from_param(x: Tensor, bits: int, param: dict) -> Tensor:
    kind = param["kind"]
    scale = param["scale"]
    ratio = param.get("ratio")
    y = x.to(torch.float32)
    if kind == "rotated":
        y = hadamard_transform(y)
        y = quantize_with_scale(y, scale, bits)
        return hadamard_transform(y)
    if kind == "rotated_repq":
        assert ratio is not None
        y = hadamard_transform(y)
        view = ratio.view(*([1] * (y.ndim - 1)), -1)
        y = quantize_with_scale(y / view, scale, bits) * view
        return hadamard_transform(y)
    if kind == "repq":
        assert ratio is not None
        view = ratio.view(*([1] * (y.ndim - 1)), -1)
        return quantize_with_scale(y / view, scale, bits) * view
    return quantize_with_scale(y, scale, bits)


def layer_proxy_metrics(
    payload: dict[str, Tensor],
    layer_param: dict[str, dict],
    bits: int,
    attn_scale: float,
) -> dict[str, float]:
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
        "logits_mse": float(mse(logits_fp, logits_hat)),
        "probs_mse": float(mse(probs_fp, probs_hat)),
        "out_mse": float(mse(out_fp, out_hat)),
    }


def rescue_proxy_score(base_metrics: dict[str, float], candidate_metrics: dict[str, float]) -> dict[str, float]:
    logits_gain = float(
        (base_metrics["logits_mse"] - candidate_metrics["logits_mse"])
        / max(base_metrics["logits_mse"], 1e-12)
    )
    probs_gain = float(
        (base_metrics["probs_mse"] - candidate_metrics["probs_mse"])
        / max(base_metrics["probs_mse"], 1e-12)
    )
    out_gain = float(
        (base_metrics["out_mse"] - candidate_metrics["out_mse"])
        / max(base_metrics["out_mse"], 1e-12)
    )
    score = float(0.15 * logits_gain + 0.55 * probs_gain + 0.30 * out_gain)
    return {
        "score": score,
        "logits_gain": logits_gain,
        "probs_gain": probs_gain,
        "out_gain": out_gain,
    }


def logits_kl(fp_logits: Tensor, cand_logits: Tensor) -> float:
    fp_probs = fp_logits.to(torch.float32).softmax(dim=-1).clamp_min(1e-8)
    cand_probs = cand_logits.to(torch.float32).softmax(dim=-1).clamp_min(1e-8)
    return float((fp_probs * (fp_probs.log() - cand_probs.log())).sum(dim=-1).mean().item())


def format_overrides(overrides: dict[int, tuple[str, ...]]) -> str:
    if not overrides:
        return "none"
    return ",".join(f"{layer}:{''.join(tokens)}" for layer, tokens in sorted(overrides.items()))


def anchor_rescue_candidates(
    num_layers: int,
    token_choices: dict[str, tuple[str, ...]],
    max_layers: int,
) -> list[dict[int, tuple[str, ...]]]:
    anchors = list(range(0, max(1, num_layers // 2), 2))
    state_names = ["none", *token_choices.keys()]
    candidates: list[dict[int, tuple[str, ...]]] = []
    for combo in itertools.product(state_names, repeat=len(anchors)):
        overrides: dict[int, tuple[str, ...]] = {}
        active = 0
        for layer, state_name in zip(anchors, combo):
            if state_name == "none":
                continue
            overrides[layer] = token_choices[state_name]
            active += 1
        if active <= max_layers:
            candidates.append(overrides)
    unique: dict[str, dict[int, tuple[str, ...]]] = {}
    for overrides in candidates:
        unique[format_overrides(overrides)] = overrides
    return list(unique.values())


def build_auto_rescue_selector(
    bundle: dict[int, dict[str, Tensor]],
    bits: int,
    attn_scale: float,
    base_mode: str,
    base_params: dict[int, dict[str, dict]],
    repq_params: dict[int, dict[str, dict]],
    model_name: str,
    checkpoint: str | None,
    calib_loader,
    device: torch.device,
    fp_calib_logits: Tensor,
) -> dict:
    num_layers = len(bundle)
    early_layers = list(range(max(1, num_layers // 2)))
    base_metrics = {
        layer: layer_proxy_metrics(bundle[layer], base_params[layer], bits=bits, attn_scale=attn_scale)
        for layer in bundle
    }

    if base_mode != "ours_rotated_qk":
        return {
            "base_mode": base_mode,
            "enabled": False,
            "triggered": False,
            "selected_overrides": {},
            "selected_layers": [],
            "criterion": "rescue_only_applies_after_qk_base_mode",
            "layer_candidates": {},
            "prefix_scores": [],
            "candidate_scores": [],
        }

    layer_candidates: dict[int, dict[str, float | str | dict[str, float]]] = {}
    ranked_layers: list[tuple[int, float, str]] = []
    for layer in early_layers:
        best_choice_name = ""
        best_choice_tokens: tuple[str, ...] = ()
        best_choice_score = float("-inf")
        best_choice_metrics: dict[str, float] | None = None
        best_choice_gains: dict[str, float] | None = None
        choices: dict[str, dict[str, float]] = {}
        for choice_name, tensor_names in AUTO_RESCUE_TOKEN_CHOICES.items():
            mixed_param = copy.deepcopy(base_params[layer])
            for tensor_name in tensor_names:
                mixed_param[tensor_name] = copy.deepcopy(repq_params[layer][tensor_name])
            candidate_metrics = layer_proxy_metrics(bundle[layer], mixed_param, bits=bits, attn_scale=attn_scale)
            gains = rescue_proxy_score(base_metrics[layer], candidate_metrics)
            choices[choice_name] = {
                **candidate_metrics,
                **gains,
            }
            if gains["score"] > best_choice_score:
                best_choice_name = choice_name
                best_choice_tokens = tensor_names
                best_choice_score = gains["score"]
                best_choice_metrics = candidate_metrics
                best_choice_gains = gains

        layer_candidates[layer] = {
            "best_choice": best_choice_name,
            "best_tokens": list(best_choice_tokens),
            "best_score": float(best_choice_score),
            "base_metrics": base_metrics[layer],
            "best_metrics": best_choice_metrics or {},
            "best_gains": best_choice_gains or {},
            "choices": choices,
        }
        if best_choice_score > AUTO_RESCUE_MIN_LAYER_SCORE:
            ranked_layers.append((layer, best_choice_score, best_choice_name))

    ranked_layers.sort(key=lambda item: item[1], reverse=True)
    prefix_scores: list[dict[str, float | int]] = []
    running_score = 0.0
    best_prefix_score = float("-inf")
    for count, (_, layer_score, _) in enumerate(ranked_layers[:AUTO_RESCUE_MAX_LAYERS], start=1):
        running_score += layer_score
        adjusted_score = float(running_score - count * AUTO_RESCUE_LAYER_PENALTY)
        prefix_scores.append(
            {
                "count": count,
                "raw_score": float(running_score),
                "adjusted_score": adjusted_score,
            }
        )
        if adjusted_score > best_prefix_score:
            best_prefix_score = adjusted_score

    candidate_scores: list[dict[str, float | str | dict[str, list[str]]]] = []
    selected_overrides: dict[int, tuple[str, ...]] = {}
    base_candidate_score = None
    best_candidate_score = float("inf")
    candidate_overrides = anchor_rescue_candidates(num_layers, AUTO_RESCUE_TOKEN_CHOICES, AUTO_RESCUE_MAX_LAYERS)
    model = load_model(model_name, checkpoint, device=device)
    patch_qkv_method(model, bits=bits, layer_params=base_params)
    for overrides in candidate_overrides:
        layer_params = apply_overrides(base_params, repq_params, overrides)
        update_patched_qkv_method(model, layer_params)
        cand_logits = collect_model_logits(model, calib_loader, device)
        logits_mse = mse(fp_calib_logits, cand_logits)
        probs_kl = logits_kl(fp_calib_logits, cand_logits)
        score = logits_mse
        record = {
            "name": format_overrides(overrides),
            "logits_mse": float(logits_mse),
            "probs_kl": float(probs_kl),
            "score": float(score),
            "overrides": {str(layer): list(tokens) for layer, tokens in overrides.items()},
        }
        candidate_scores.append(record)
        if not overrides:
            base_candidate_score = float(score)
        if score < best_candidate_score:
            best_candidate_score = float(score)
            selected_overrides = overrides
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if base_candidate_score is None:
        raise RuntimeError("Base candidate score for rescue selector is missing")
    rel_improvement = float((base_candidate_score - best_candidate_score) / max(base_candidate_score, 1e-12))
    if rel_improvement < AUTO_RESCUE_MIN_REL_IMPROVEMENT:
        selected_overrides = {}

    return {
        "base_mode": base_mode,
        "enabled": True,
        "triggered": bool(selected_overrides),
        "selected_overrides": {str(layer): list(tokens) for layer, tokens in selected_overrides.items()},
        "selected_layers": list(selected_overrides.keys()),
        "criterion": "rank early layers by calibration proxy gain from selective RepQ rescue on top of QK base; keep best prefix if adjusted gain clears threshold",
        "min_layer_score": AUTO_RESCUE_MIN_LAYER_SCORE,
        "min_total_score": AUTO_RESCUE_MIN_TOTAL_SCORE,
        "layer_penalty": AUTO_RESCUE_LAYER_PENALTY,
        "max_layers": AUTO_RESCUE_MAX_LAYERS,
        "min_rel_improvement": AUTO_RESCUE_MIN_REL_IMPROVEMENT,
        "num_full_evaluated_candidates": len(candidate_scores),
        "early_layers": early_layers,
        "layer_candidates": {str(layer): layer_candidates[layer] for layer in layer_candidates},
        "prefix_scores": prefix_scores,
        "candidate_scores": sorted(candidate_scores, key=lambda row: float(row["score"])),
        "base_candidate_score": float(base_candidate_score),
        "best_candidate_score": float(best_candidate_score),
        "best_rel_improvement": rel_improvement,
    }


def mode_tensor_params(bundle: dict[int, dict[str, Tensor]], bits: int, mode: str) -> tuple[dict[int, dict[str, dict]], dict[int, dict[str, dict]]]:
    params: dict[int, dict[str, dict]] = {}
    proxy: dict[int, dict[str, dict]] = {}
    for layer, payload in bundle.items():
        params[layer] = {}
        proxy[layer] = {}
        for name in ("q", "k", "v"):
            x = payload[name]
            x_rot = hadamard_transform(x)
            direct_scale = estimate_percentile_scale(x, bits=bits)
            rotated_scale = estimate_percentile_scale(x_rot, bits=bits)
            ptq_scale = search_scalar_scale(x, bits=bits)
            repq_target, repq_ratio = build_repq_params(x, bits=bits)
            rot_repq_target, rot_repq_ratio = build_repq_params(x_rot, bits=bits)

            if mode == "direct_qkv":
                params[layer][name] = {"kind": "direct", "scale": direct_scale}
            elif mode == "ptq4vit_style_qkv":
                params[layer][name] = {"kind": "direct", "scale": ptq_scale}
            elif mode == "repq_style_qkv":
                params[layer][name] = {"kind": "repq", "scale": repq_target, "ratio": repq_ratio}
            elif mode == "ours_rotated_qk":
                if name in ("q", "k"):
                    params[layer][name] = {"kind": "rotated", "scale": rotated_scale}
                else:
                    params[layer][name] = {"kind": "direct", "scale": direct_scale}
            elif mode == "ours_rotated_qkv":
                params[layer][name] = {"kind": "rotated", "scale": rotated_scale}
            elif mode == "hybrid_q_rotrepq_k_rot_v_direct":
                if name == "q":
                    params[layer][name] = {"kind": "rotated_repq", "scale": rot_repq_target, "ratio": rot_repq_ratio}
                elif name == "k":
                    params[layer][name] = {"kind": "rotated", "scale": rotated_scale}
                else:
                    params[layer][name] = {"kind": "direct", "scale": direct_scale}
            elif mode == "hybrid_rotated_qk_repq_v":
                if name in ("q", "k"):
                    params[layer][name] = {"kind": "rotated", "scale": rotated_scale}
                else:
                    params[layer][name] = {"kind": "repq", "scale": repq_target, "ratio": repq_ratio}
            elif mode == "hybrid_rotated_repq_qk_repq_v":
                if name in ("q", "k"):
                    params[layer][name] = {"kind": "rotated_repq", "scale": rot_repq_target, "ratio": rot_repq_ratio}
                else:
                    params[layer][name] = {"kind": "repq", "scale": repq_target, "ratio": repq_ratio}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            info = params[layer][name]
            proxy[layer][name] = tensor_proxy_record(
                x,
                bits=bits,
                scale=info["scale"],
                kind=info["kind"],
                ratio=info.get("ratio"),
            )
        proxy[layer]["total_mse"] = {
            "mse": float(
                sum(proxy[layer][name]["mse"] for name in ("q", "k", "v")) / 3.0
            )
        }
    return params, proxy


def proxy_from_layer_params(
    bundle: dict[int, dict[str, Tensor]],
    bits: int,
    layer_params: dict[int, dict[str, dict]],
) -> dict[int, dict[str, dict]]:
    proxy: dict[int, dict[str, dict]] = {}
    for layer, payload in bundle.items():
        proxy[layer] = {}
        for name in ("q", "k", "v"):
            x = payload[name]
            info = layer_params[layer][name]
            proxy[layer][name] = tensor_proxy_record(
                x,
                bits=bits,
                scale=info["scale"],
                kind=info["kind"],
                ratio=info.get("ratio"),
            )
        proxy[layer]["total_mse"] = {
            "mse": float(sum(proxy[layer][name]["mse"] for name in ("q", "k", "v")) / 3.0)
        }
    return proxy


class AdaptedQKVAttentionWrapper(nn.Module):
    def __init__(self, base: nn.Module, bits: int, params: dict[str, dict]) -> None:
        super().__init__()
        self.num_heads = base.num_heads
        self.scale = float(base.scale)
        self.qkv = base.qkv
        self.q_norm = base.q_norm
        self.k_norm = base.k_norm
        self.attn_drop = base.attn_drop
        self.proj = base.proj
        self.proj_drop = base.proj_drop
        self.bits = bits
        self.kind_map = {name: params[name]["kind"] for name in ("q", "k", "v")}
        param_device = base.qkv.weight.device

        for name in ("q", "k", "v"):
            scale = params[name]["scale"].detach().to(device=param_device, dtype=torch.float32).reshape(())
            ratio = params[name].get("ratio")
            if ratio is None:
                ratio = torch.ones(base.qkv.out_features // (3 * self.num_heads), dtype=torch.float32)
            self.register_buffer(f"scale_{name}", scale)
            self.register_buffer(
                f"ratio_{name}",
                ratio.detach().to(device=param_device, dtype=torch.float32).reshape(-1),
            )

    def set_params(self, params: dict[str, dict]) -> None:
        param_device = self.scale_q.device
        for name in ("q", "k", "v"):
            self.kind_map[name] = params[name]["kind"]
            scale = params[name]["scale"].detach().to(device=param_device, dtype=torch.float32).reshape(())
            ratio = params[name].get("ratio")
            target_ratio = getattr(self, f"ratio_{name}")
            if ratio is None:
                ratio_tensor = torch.ones_like(target_ratio)
            else:
                ratio_tensor = ratio.detach().to(device=param_device, dtype=torch.float32).reshape(-1)
                if ratio_tensor.shape != target_ratio.shape:
                    raise ValueError(
                        f"Ratio shape mismatch for {name}: expected {tuple(target_ratio.shape)}, got {tuple(ratio_tensor.shape)}"
                    )
            getattr(self, f"scale_{name}").copy_(scale)
            target_ratio.copy_(ratio_tensor)

    def _quant(self, x: Tensor, name: str) -> Tensor:
        kind = self.kind_map[name]
        scale = getattr(self, f"scale_{name}")
        ratio = getattr(self, f"ratio_{name}")
        y = x.to(torch.float32)
        if kind == "rotated":
            y = hadamard_transform(y)
            y = quantize_with_scale(y, scale, self.bits)
            y = hadamard_transform(y)
            return y.to(x.dtype)
        if kind == "rotated_repq":
            y = hadamard_transform(y)
            view = ratio.view(*([1] * (y.ndim - 1)), -1)
            y = quantize_with_scale(y / view, scale, self.bits) * view
            y = hadamard_transform(y)
            return y.to(x.dtype)
        if kind == "repq":
            view = ratio.view(*([1] * (y.ndim - 1)), -1)
            y = quantize_with_scale(y / view, scale, self.bits) * view
            return y.to(x.dtype)
        y = quantize_with_scale(y, scale, self.bits)
        return y.to(x.dtype)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        del attn_mask
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            bsz,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self._quant(self.q_norm(q), "q")
        k = self._quant(self.k_norm(k), "k")
        v = self._quant(v, "v")
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def patch_qkv_method(model: nn.Module, bits: int, layer_params: dict[int, dict[str, dict]]) -> int:
    attn_list = attention_modules(model)
    if hasattr(model, "blocks") and len(model.blocks) == len(attn_list):
        for idx, block in enumerate(model.blocks):
            block.attn = AdaptedQKVAttentionWrapper(block.attn, bits=bits, params=layer_params[idx])
        return len(attn_list)

    replaced = 0

    def _patch(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if (
                child.__class__.__name__ == "Attention"
                and child.__class__.__module__.startswith("timm.")
            ):
                setattr(
                    parent,
                    name,
                    AdaptedQKVAttentionWrapper(child, bits=bits, params=layer_params[replaced]),
                )
                replaced += 1
            else:
                _patch(child)

    _patch(model)
    return replaced


def update_patched_qkv_method(model: nn.Module, layer_params: dict[int, dict[str, dict]]) -> int:
    if hasattr(model, "blocks"):
        attn_list = [block.attn for block in model.blocks if isinstance(block.attn, AdaptedQKVAttentionWrapper)]
    else:
        attn_list = [
            module
            for module in model.modules()
            if isinstance(module, AdaptedQKVAttentionWrapper)
        ]
    if not attn_list:
        raise RuntimeError("No AdaptedQKVAttentionWrapper modules found")
    if len(attn_list) != len(layer_params):
        raise ValueError(f"Layer count mismatch: {len(attn_list)} wrappers vs {len(layer_params)} params")
    for idx, module in enumerate(attn_list):
        module.set_params(layer_params[idx])
    return len(attn_list)


def mean_proxy(proxy: dict[int, dict[str, dict]]) -> float:
    values = [proxy[layer]["total_mse"]["mse"] for layer in proxy]
    return float(sum(values) / len(values))


def run_model(args: argparse.Namespace, model_key: str, device: torch.device) -> dict:
    spec = MODEL_SPECS[model_key]
    batch_size = spec["batch_size"]
    val_loader = build_loader(
        dataset_root=args.eval_dataset_root or args.dataset_root,
        split="val",
        batch_size=batch_size,
        num_workers=args.num_workers,
        num_images=args.val_images,
        seed=args.seed,
        shuffle=False,
        device=device,
    )

    model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    num_layers = len(attention_modules(model))
    layers = list(range(num_layers))
    fp_stats = evaluate_with_resources(model, val_loader, device, f"idea2_adapted:{model_key}:fp")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "model": spec["model_name"],
        "checkpoint": spec["checkpoint"],
        "num_layers": num_layers,
        "fp": fp_stats,
        "calibrations": {},
    }

    for calib_images in args.calib_sizes:
        calib_loader = build_loader(
            dataset_root=args.dataset_root,
            split="train",
            batch_size=min(batch_size, 16),
            num_workers=args.num_workers,
            num_images=calib_images,
            seed=args.seed + 17,
            shuffle=False,
            device=device,
        )
        calib_model = load_model(spec["model_name"], spec["checkpoint"], device=device)
        attn_scale = float(attention_modules(calib_model)[0].scale)
        bundle, calib_stats = collect_bundle_with_resources(
            calib_model,
            calib_loader,
            device=device,
            layers=layers,
            max_batches=max(1, math.ceil(calib_images / min(batch_size, 16))),
        )
        fp_calib_logits = None
        if any(mode in args.modes for mode in ("auto_place_rescue", "forced_qk_rescue")):
            fp_calib_logits = collect_model_logits(calib_model, calib_loader, device)
        del calib_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        calib_result = {
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
            auto_selector_elapsed = float(time.perf_counter() - selector_start)
            param_cache: dict[str, tuple[dict[int, dict[str, dict]], dict[int, dict[str, dict]]]] = {}

            def get_mode_params(mode_name: str) -> tuple[dict[int, dict[str, dict]], dict[int, dict[str, dict]]]:
                cached = param_cache.get(mode_name)
                if cached is None:
                    cached = mode_tensor_params(bundle, bits=bits, mode=mode_name)
                    param_cache[mode_name] = cached
                return cached

            rescue_selector = None
            rescue_selector_elapsed = 0.0
            if "auto_place_rescue" in args.modes:
                base_mode = auto_selector["chosen_mode"]
                base_params, _ = get_mode_params(base_mode)
                repq_params, _ = get_mode_params("repq_style_qkv")
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
                rescue_selector_elapsed = float(time.perf_counter() - rescue_start)
            forced_qk_rescue_selector = None
            forced_qk_rescue_selector_elapsed = 0.0
            if "forced_qk_rescue" in args.modes:
                qk_params, _ = get_mode_params("ours_rotated_qk")
                repq_params, _ = get_mode_params("repq_style_qkv")
                rescue_start = time.perf_counter()
                forced_qk_rescue_selector = build_auto_rescue_selector(
                    bundle,
                    bits=bits,
                    attn_scale=attn_scale,
                    base_mode="ours_rotated_qk",
                    base_params=qk_params,
                    repq_params=repq_params,
                    model_name=spec["model_name"],
                    checkpoint=spec["checkpoint"],
                    calib_loader=calib_loader,
                    device=device,
                    fp_calib_logits=fp_calib_logits,
                )
                forced_qk_rescue_selector_elapsed = float(time.perf_counter() - rescue_start)
            per_bit = {}
            direct_top1 = None
            for mode in args.modes:
                if mode == "auto_place":
                    effective_mode = auto_selector["chosen_mode"]
                    layer_params, proxy = get_mode_params(effective_mode)
                elif mode == "auto_place_rescue":
                    if rescue_selector is None:
                        raise RuntimeError("auto_place_rescue requested without rescue selector")
                    effective_mode = rescue_selector["base_mode"]
                    base_params, _ = get_mode_params(effective_mode)
                    repq_params, _ = get_mode_params("repq_style_qkv")
                    override_map = {
                        int(layer): tuple(tokens)
                        for layer, tokens in rescue_selector["selected_overrides"].items()
                    }
                    layer_params = apply_overrides(base_params, repq_params, override_map)
                    proxy = proxy_from_layer_params(bundle, bits=bits, layer_params=layer_params)
                elif mode == "forced_qk_rescue":
                    if forced_qk_rescue_selector is None:
                        raise RuntimeError("forced_qk_rescue requested without rescue selector")
                    effective_mode = "ours_rotated_qk"
                    base_params, _ = get_mode_params(effective_mode)
                    repq_params, _ = get_mode_params("repq_style_qkv")
                    override_map = {
                        int(layer): tuple(tokens)
                        for layer, tokens in forced_qk_rescue_selector["selected_overrides"].items()
                    }
                    layer_params = apply_overrides(base_params, repq_params, override_map)
                    proxy = proxy_from_layer_params(bundle, bits=bits, layer_params=layer_params)
                else:
                    effective_mode = mode
                    layer_params, proxy = get_mode_params(effective_mode)
                model_eval = load_model(spec["model_name"], spec["checkpoint"], device=device)
                replaced = patch_qkv_method(model_eval, bits=bits, layer_params=layer_params)
                stats = evaluate_with_resources(
                    model_eval,
                    val_loader,
                    device,
                    f"idea2_adapted:{model_key}:c{calib_images}:{mode}:b{bits}",
                )
                record = {
                    **stats,
                    "replaced_layers": replaced,
                    "proxy_mean_mse": mean_proxy(proxy),
                    "proxy_per_layer": {str(k): v for k, v in proxy.items()},
                    "delta_top1_vs_fp": float(stats["top1"] - fp_stats["top1"]),
                }
                if mode == "auto_place":
                    record["selected_mode"] = auto_selector["chosen_mode"]
                    record["selector"] = auto_selector
                    record["selector_elapsed_sec"] = auto_selector_elapsed
                if mode == "auto_place_rescue":
                    record["selected_mode"] = rescue_selector["base_mode"]
                    record["selector"] = auto_selector
                    record["selector_elapsed_sec"] = auto_selector_elapsed
                    record["rescue_selector"] = rescue_selector
                    record["rescue_selector_elapsed_sec"] = rescue_selector_elapsed
                if mode == "forced_qk_rescue":
                    record["selected_mode"] = effective_mode
                    record["forced_base_mode"] = "ours_rotated_qk"
                    record["rescue_selector"] = forced_qk_rescue_selector
                    record["rescue_selector_elapsed_sec"] = forced_qk_rescue_selector_elapsed
                if mode == "direct_qkv":
                    direct_top1 = stats["top1"]
                    record["delta_top1_vs_direct"] = 0.0
                elif direct_top1 is not None:
                    record["delta_top1_vs_direct"] = float(stats["top1"] - direct_top1)
                per_bit[mode] = record
                del model_eval
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            calib_result["bits"][str(bits)] = per_bit

        result["calibrations"][str(calib_images)] = calib_result
    return result


def main() -> None:
    args = parse_args()
    hparams = apply_hparam_overrides(args)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {
        "setup": {
            "dataset_root": args.dataset_root,
            "eval_dataset_root": args.eval_dataset_root or args.dataset_root,
            "seed": args.seed,
            "device": str(device),
            "val_images": args.val_images,
            "calib_sizes": args.calib_sizes,
            "bits": args.bits,
            "modes": args.modes,
            "models": args.models,
            "run_name": args.run_name,
            "hparams": hparams,
        },
        "models": {},
    }
    for model_key in args.models:
        results["models"][model_key] = run_model(args, model_key, device)

    out_path = Path(args.results_dir) / f"{args.run_name}.json"
    save_json(out_path, results)
    print(f"[Done] idea2 adapted baselines saved to {out_path}")


if __name__ == "__main__":
    main()
