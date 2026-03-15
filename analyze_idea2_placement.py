from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from common import (
    attention_modules,
    build_loader,
    collect_attention_bundle,
    estimate_percentile_scale,
    evaluate_classification,
    hadamard_transform,
    kurtosis,
    load_model,
    mse,
    quantize_with_scale,
    save_json,
    seed_everything,
)


MODEL_SPECS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze why idea2 placement differs across models")
    parser.add_argument("--dataset-root", type=str, default="/home/cxj/DyadicFold/data/imagenet")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/cxj/experiments/vit_quant_5ideas/results/idea2_analysis",
    )
    parser.add_argument("--run-name", type=str, default="placement_diagnostics")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--models", type=str, nargs="+", default=["deit_small", "vit_base"])
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--val-images", type=int, default=5000)
    parser.add_argument("--confirm-val-images", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-v-layers", type=int, default=4)
    return parser.parse_args()


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
            torch.quantile(scalar, 0.999).item()
            / max(torch.quantile(scalar, 0.95).item(), 1e-12)
        ),
    }


def quantize_tensor(x: Tensor, bits: int, rotate: bool) -> tuple[Tensor, Tensor]:
    if rotate:
        y = hadamard_transform(x)
        scale = estimate_percentile_scale(y, bits=bits)
        q = hadamard_transform(quantize_with_scale(y, scale, bits))
        return q, scale
    scale = estimate_percentile_scale(x, bits=bits)
    q = quantize_with_scale(x, scale, bits)
    return q, scale


def kl_rows(p: Tensor, q: Tensor) -> float:
    p32 = p.to(torch.float32).clamp_min(1e-8)
    q32 = q.to(torch.float32).clamp_min(1e-8)
    return float((p32 * (p32.log() - q32.log())).sum(dim=-1).mean().item())


def compute_layer_diagnostics(
    payload: dict[str, Tensor],
    bits: int,
    attn_scale: float,
) -> dict[str, float]:
    q = payload["q"].to(torch.float32)
    k = payload["k"].to(torch.float32)
    v = payload["v"].to(torch.float32)
    probs_fp = payload["attn_probs"].to(torch.float32)

    q_direct, _ = quantize_tensor(q, bits, rotate=False)
    q_rot, _ = quantize_tensor(q, bits, rotate=True)
    k_direct, _ = quantize_tensor(k, bits, rotate=False)
    k_rot, _ = quantize_tensor(k, bits, rotate=True)
    v_direct, _ = quantize_tensor(v, bits, rotate=False)
    v_rot, _ = quantize_tensor(v, bits, rotate=True)

    logits_fp = (q * attn_scale) @ k.transpose(-2, -1)

    q_direct_logits = (q_direct * attn_scale) @ k.transpose(-2, -1)
    q_rot_logits = (q_rot * attn_scale) @ k.transpose(-2, -1)
    k_direct_logits = (q * attn_scale) @ k_direct.transpose(-2, -1)
    k_rot_logits = (q * attn_scale) @ k_rot.transpose(-2, -1)
    qk_direct_logits = (q_direct * attn_scale) @ k_direct.transpose(-2, -1)
    qk_rot_logits = (q_rot * attn_scale) @ k_rot.transpose(-2, -1)

    q_direct_prob = q_direct_logits.softmax(dim=-1)
    q_rot_prob = q_rot_logits.softmax(dim=-1)
    k_direct_prob = k_direct_logits.softmax(dim=-1)
    k_rot_prob = k_rot_logits.softmax(dim=-1)
    qk_direct_prob = qk_direct_logits.softmax(dim=-1)
    qk_rot_prob = qk_rot_logits.softmax(dim=-1)

    out_fp = probs_fp @ v
    out_direct = probs_fp @ v_direct
    out_rot = probs_fp @ v_rot

    q_logit_direct = mse(logits_fp, q_direct_logits)
    q_logit_rot = mse(logits_fp, q_rot_logits)
    k_logit_direct = mse(logits_fp, k_direct_logits)
    k_logit_rot = mse(logits_fp, k_rot_logits)
    qk_prob_direct = mse(probs_fp, qk_direct_prob)
    qk_prob_rot = mse(probs_fp, qk_rot_prob)
    v_out_direct = mse(out_fp, out_direct)
    v_out_rot = mse(out_fp, out_rot)

    return {
        "q_tensor_gain": float((mse(q, q_direct) - mse(q, q_rot)) / max(mse(q, q_direct), 1e-12)),
        "k_tensor_gain": float((mse(k, k_direct) - mse(k, k_rot)) / max(mse(k, k_direct), 1e-12)),
        "v_tensor_gain": float((mse(v, v_direct) - mse(v, v_rot)) / max(mse(v, v_direct), 1e-12)),
        "q_logit_gain": float((q_logit_direct - q_logit_rot) / max(q_logit_direct, 1e-12)),
        "k_logit_gain": float((k_logit_direct - k_logit_rot) / max(k_logit_direct, 1e-12)),
        "qk_prob_gain": float((qk_prob_direct - qk_prob_rot) / max(qk_prob_direct, 1e-12)),
        "qk_prob_direct_mse": float(qk_prob_direct),
        "qk_prob_rot_mse": float(qk_prob_rot),
        "qk_prob_direct_kl": float(kl_rows(probs_fp, qk_direct_prob)),
        "qk_prob_rot_kl": float(kl_rows(probs_fp, qk_rot_prob)),
        "v_out_gain": float((v_out_direct - v_out_rot) / max(v_out_direct, 1e-12)),
        "v_out_direct_mse": float(v_out_direct),
        "v_out_rot_mse": float(v_out_rot),
        "q_stats": tensor_channel_stats(q),
        "k_stats": tensor_channel_stats(k),
        "v_stats": tensor_channel_stats(v),
    }


def summarize_layers(layer_metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    keys = [
        "q_tensor_gain",
        "k_tensor_gain",
        "v_tensor_gain",
        "q_logit_gain",
        "k_logit_gain",
        "qk_prob_gain",
        "v_out_gain",
    ]
    out: dict[str, float] = {}
    n = len(layer_metrics)
    split = max(1, n // 2)
    early_layers = list(range(split))
    late_layers = list(range(split, n))
    for key in keys:
        vals = [layer_metrics[layer][key] for layer in range(n)]
        out[f"{key}_mean"] = float(sum(vals) / len(vals))
        out[f"{key}_early_mean"] = float(sum(layer_metrics[layer][key] for layer in early_layers) / len(early_layers))
        out[f"{key}_late_mean"] = float(sum(layer_metrics[layer][key] for layer in late_layers) / len(late_layers))
    for tensor_name in ("q", "k", "v"):
        for stat_name in ("kurtosis", "channel_scale_cv", "channel_scale_max_over_median", "abs_q999_over_abs_q95"):
            vals = [layer_metrics[layer][f"{tensor_name}_stats"][stat_name] for layer in range(n)]
            out[f"{tensor_name}_{stat_name}_mean"] = float(sum(vals) / len(vals))
            out[f"{tensor_name}_{stat_name}_early_mean"] = float(
                sum(layer_metrics[layer][f"{tensor_name}_stats"][stat_name] for layer in early_layers) / len(early_layers)
            )
            out[f"{tensor_name}_{stat_name}_late_mean"] = float(
                sum(layer_metrics[layer][f"{tensor_name}_stats"][stat_name] for layer in late_layers) / len(late_layers)
            )
    return out


def build_scales(bundle: dict[int, dict[str, Tensor]], bits: int) -> dict[int, dict[str, Tensor]]:
    out: dict[int, dict[str, Tensor]] = {}
    for layer, payload in bundle.items():
        out[layer] = {}
        for name in ("q", "k", "v"):
            out[layer][name] = estimate_percentile_scale(payload[name], bits=bits)
            out[layer][f"{name}_rot"] = estimate_percentile_scale(hadamard_transform(payload[name]), bits=bits)
    return out


class LayerPlacementAttentionWrapper(nn.Module):
    def __init__(self, base: nn.Module, bits: int, scales: dict[str, Tensor], rotate_map: dict[str, bool]) -> None:
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
        self.rotate_map = rotate_map
        param_device = base.qkv.weight.device
        self.register_buffer("scale_q", scales["q"].detach().to(device=param_device, dtype=torch.float32).reshape(()))
        self.register_buffer("scale_k", scales["k"].detach().to(device=param_device, dtype=torch.float32).reshape(()))
        self.register_buffer("scale_v", scales["v"].detach().to(device=param_device, dtype=torch.float32).reshape(()))

    def _quant(self, x: Tensor, scale: Tensor, rotate: bool) -> Tensor:
        y = x.to(torch.float32)
        if rotate:
            y = hadamard_transform(y)
        y = quantize_with_scale(y, scale, self.bits)
        if rotate:
            y = hadamard_transform(y)
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
        q = self._quant(self.q_norm(q), self.scale_q, self.rotate_map["q"])
        k = self._quant(self.k_norm(k), self.scale_k, self.rotate_map["k"])
        v = self._quant(v, self.scale_v, self.rotate_map["v"])
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def patch_layerwise_placement(
    model: nn.Module,
    bits: int,
    scales: dict[int, dict[str, Tensor]],
    rotate_by_layer: dict[int, dict[str, bool]],
) -> int:
    attn_list = attention_modules(model)
    if hasattr(model, "blocks") and len(model.blocks) == len(attn_list):
        for idx, block in enumerate(model.blocks):
            chosen = {
                "q": scales[idx]["q_rot" if rotate_by_layer[idx]["q"] else "q"],
                "k": scales[idx]["k_rot" if rotate_by_layer[idx]["k"] else "k"],
                "v": scales[idx]["v_rot" if rotate_by_layer[idx]["v"] else "v"],
            }
            block.attn = LayerPlacementAttentionWrapper(
                block.attn,
                bits=bits,
                scales=chosen,
                rotate_map=rotate_by_layer[idx],
            )
        return len(attn_list)
    raise RuntimeError("Expected timm ViT blocks layout")


def make_rotate_by_layer(num_layers: int, q_layers: set[int], k_layers: set[int], v_layers: set[int]) -> dict[int, dict[str, bool]]:
    return {
        layer: {
            "q": layer in q_layers,
            "k": layer in k_layers,
            "v": layer in v_layers,
        }
        for layer in range(num_layers)
    }


def run_eval_config(
    spec: dict,
    bits: int,
    scales: dict[int, dict[str, Tensor]],
    rotate_by_layer: dict[int, dict[str, bool]],
    val_loader,
    device: torch.device,
    tag: str,
    fp_top1: float,
) -> dict:
    model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    replaced = patch_layerwise_placement(model, bits=bits, scales=scales, rotate_by_layer=rotate_by_layer)
    stats = evaluate_with_resources(model, val_loader, device, tag)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        **stats,
        "replaced_layers": replaced,
        "delta_top1_vs_fp": float(stats["top1"] - fp_top1),
    }


def pick_top_v_layers(layer_metrics: dict[int, dict[str, float]], top_k: int) -> list[int]:
    ranked = sorted(layer_metrics, key=lambda layer: layer_metrics[layer]["v_out_gain"], reverse=True)
    return ranked[:top_k]


def pick_bottom_v_layers(layer_metrics: dict[int, dict[str, float]], top_k: int) -> list[int]:
    ranked = sorted(layer_metrics, key=lambda layer: layer_metrics[layer]["v_out_gain"])
    return ranked[:top_k]


def model_arch_summary(model: nn.Module) -> dict[str, int]:
    blocks = getattr(model, "blocks", None)
    num_blocks = len(blocks) if blocks is not None else len(attention_modules(model))
    embed_dim = int(getattr(model, "embed_dim", 0))
    num_heads = int(getattr(model.blocks[0].attn, "num_heads", 0))
    mlp_hidden = int(getattr(model.blocks[0].mlp.fc1, "out_features", 0))
    return {
        "num_layers": num_blocks,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": int(embed_dim // max(num_heads, 1)),
        "mlp_hidden_dim": mlp_hidden,
    }


def run_model(args: argparse.Namespace, model_key: str, device: torch.device) -> dict:
    spec = MODEL_SPECS[model_key]
    batch_size = spec["batch_size"]

    calib_loader = build_loader(
        dataset_root=args.dataset_root,
        split="train",
        batch_size=min(batch_size, 16),
        num_workers=args.num_workers,
        num_images=args.calib_size,
        seed=args.seed + 101,
        shuffle=False,
        device=device,
    )
    val_loader = build_loader(
        dataset_root=args.dataset_root,
        split="val",
        batch_size=batch_size,
        num_workers=args.num_workers,
        num_images=args.val_images,
        seed=args.seed,
        shuffle=False,
        device=device,
    )
    confirm_val_loader = build_loader(
        dataset_root=args.dataset_root,
        split="val",
        batch_size=batch_size,
        num_workers=args.num_workers,
        num_images=args.confirm_val_images,
        seed=args.seed,
        shuffle=False,
        device=device,
    )

    model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    arch = model_arch_summary(model)
    layers = list(range(arch["num_layers"]))
    fp_5k = evaluate_with_resources(model, val_loader, device, f"idea2_analysis:{model_key}:fp5k")
    fp_50k = evaluate_with_resources(model, confirm_val_loader, device, f"idea2_analysis:{model_key}:fp50k")
    bundle, calib_stats = collect_bundle_with_resources(
        model,
        calib_loader,
        device=device,
        layers=layers,
        max_batches=max(1, math.ceil(args.calib_size / min(batch_size, 16))),
    )
    attn_scale = float(model.blocks[0].attn.scale)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "model": spec["model_name"],
        "checkpoint": spec["checkpoint"],
        "arch": arch,
        "fp_5k": fp_5k,
        "fp_50k": fp_50k,
        "calibration": {
            "num_images": args.calib_size,
            "elapsed_sec": calib_stats["elapsed_sec"],
            "peak_mem_mib": calib_stats["peak_mem_mib"],
        },
        "bits": {},
    }

    for bits in args.bits:
        layer_metrics = {
            layer: compute_layer_diagnostics(bundle[layer], bits=bits, attn_scale=attn_scale)
            for layer in layers
        }
        scales = build_scales(bundle, bits=bits)
        top_v_layers = pick_top_v_layers(layer_metrics, args.top_v_layers)
        bottom_v_layers = pick_bottom_v_layers(layer_metrics, args.top_v_layers)
        qk_layers = set(layers)
        no_layers: set[int] = set()
        configs = {
            "direct_qkv": make_rotate_by_layer(arch["num_layers"], no_layers, no_layers, no_layers),
            "rotated_v_only": make_rotate_by_layer(arch["num_layers"], no_layers, no_layers, set(layers)),
            "rotated_qk": make_rotate_by_layer(arch["num_layers"], qk_layers, qk_layers, no_layers),
            "rotated_qkv": make_rotate_by_layer(arch["num_layers"], qk_layers, qk_layers, set(layers)),
            "rotated_qk_topv": make_rotate_by_layer(arch["num_layers"], qk_layers, qk_layers, set(top_v_layers)),
            "rotated_qk_bottomv": make_rotate_by_layer(arch["num_layers"], qk_layers, qk_layers, set(bottom_v_layers)),
        }

        eval_5k = {}
        for name, rotate_by_layer in configs.items():
            eval_5k[name] = run_eval_config(
                spec=spec,
                bits=bits,
                scales=scales,
                rotate_by_layer=rotate_by_layer,
                val_loader=val_loader,
                device=device,
                tag=f"idea2_analysis:{model_key}:{name}:b{bits}:5k",
                fp_top1=fp_5k["top1"],
            )

        eval_50k = {}
        for name in ("rotated_qk", "rotated_qkv", "rotated_qk_topv", "rotated_qk_bottomv"):
            eval_50k[name] = run_eval_config(
                spec=spec,
                bits=bits,
                scales=scales,
                rotate_by_layer=configs[name],
                val_loader=confirm_val_loader,
                device=device,
                tag=f"idea2_analysis:{model_key}:{name}:b{bits}:50k",
                fp_top1=fp_50k["top1"],
            )

        result["bits"][str(bits)] = {
            "summary": summarize_layers(layer_metrics),
            "top_v_layers": top_v_layers,
            "bottom_v_layers": bottom_v_layers,
            "layer_metrics": {str(layer): layer_metrics[layer] for layer in layers},
            "diagnostic_eval_5k": eval_5k,
            "confirm_eval_50k": eval_50k,
        }

    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {
        "setup": {
            "dataset_root": args.dataset_root,
            "seed": args.seed,
            "device": str(device),
            "models": args.models,
            "bits": args.bits,
            "calib_size": args.calib_size,
            "val_images": args.val_images,
            "confirm_val_images": args.confirm_val_images,
            "top_v_layers": args.top_v_layers,
            "run_name": args.run_name,
        },
        "models": {},
    }
    out_path = Path(args.results_dir) / f"{args.run_name}.json"
    for model_key in args.models:
        results["models"][model_key] = run_model(args, model_key, device)
        save_json(out_path, results)
    save_json(out_path, results)
    print(f"[Done] idea2 placement analysis saved to {out_path}")


if __name__ == "__main__":
    main()
