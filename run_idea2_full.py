from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from common import (
    attention_modules,
    build_loader,
    collect_attention_bundle,
    evaluate_classification,
    hadamard_transform,
    load_model,
    mse,
    quantize_with_scale,
    save_json,
    seed_everything,
    estimate_percentile_scale,
)


MODEL_SPECS = {
    "deit_tiny": {
        "model_name": "deit_tiny_patch16_224",
        "checkpoint": None,
        "batch_size": 128,
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


class FixedScaleQKVQuantAttentionWrapper(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        bits: int,
        scales: dict[str, Tensor],
        rotate_map: dict[str, bool],
    ) -> None:
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
        self.register_buffer("scale_q", scales["q"].detach().to(torch.float32).reshape(()))
        self.register_buffer("scale_k", scales["k"].detach().to(torch.float32).reshape(()))
        self.register_buffer("scale_v", scales["v"].detach().to(torch.float32).reshape(()))

    def _quant(self, x: Tensor, scale: Tensor, rotate: bool) -> Tensor:
        orig_dtype = x.dtype
        y = x.to(torch.float32)
        if rotate:
            y = hadamard_transform(y)
        y = quantize_with_scale(y, scale, self.bits)
        if rotate:
            y = hadamard_transform(y)
        return y.to(orig_dtype)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal idea2 experiments: Q/K rotation for ViT PTQ")
    parser.add_argument("--dataset-root", type=str, default="/home/cxj/DyadicFold/data/imagenet")
    parser.add_argument("--results-dir", type=str, default="/home/cxj/experiments/vit_quant_5ideas/results/idea2")
    parser.add_argument("--run-name", type=str, default="summary")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--models", type=str, nargs="+", default=["deit_small", "vit_base"])
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["direct_qkv", "rotated_q", "rotated_k", "rotated_qk", "rotated_qkv", "auto_place"],
    )
    parser.add_argument("--val-images", type=int, default=50000)
    parser.add_argument("--calib-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def patch_qkv_quant(
    model: nn.Module,
    bits: int,
    scales: dict[int, dict[str, Tensor]],
    rotate_map: dict[str, bool],
) -> int:
    attn_list = attention_modules(model)
    if hasattr(model, "blocks") and len(model.blocks) == len(attn_list):
        for idx, block in enumerate(model.blocks):
            block.attn = FixedScaleQKVQuantAttentionWrapper(
                block.attn,
                bits=bits,
                scales=scales[idx],
                rotate_map=rotate_map,
            )
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
                    FixedScaleQKVQuantAttentionWrapper(
                        child,
                        bits=bits,
                        scales=scales[replaced],
                        rotate_map=rotate_map,
                    ),
                )
                replaced += 1
            else:
                _patch(child)

    _patch(model)
    return replaced


def build_scales_and_proxy(
    bundle: dict[int, dict[str, Tensor]],
    bits: int,
) -> tuple[dict[int, dict[str, Tensor]], dict[int, dict[str, float]]]:
    scales = {}
    proxy = {}
    for layer, payload in bundle.items():
        scales[layer] = {}
        proxy[layer] = {}
        for name in ("q", "k", "v"):
            tensor = payload[name]
            direct_scale = estimate_percentile_scale(tensor, bits=bits)
            rotated_tensor = hadamard_transform(tensor)
            rotated_scale = estimate_percentile_scale(rotated_tensor, bits=bits)
            direct_quant = quantize_with_scale(tensor, direct_scale, bits)
            rotated_quant = hadamard_transform(
                quantize_with_scale(rotated_tensor, rotated_scale, bits)
            )
            direct_mse = mse(tensor, direct_quant)
            rotated_mse = mse(tensor, rotated_quant)
            scales[layer][name] = direct_scale
            scales[layer][f"{name}_rot"] = rotated_scale
            proxy[layer][f"{name}_direct_mse"] = direct_mse
            proxy[layer][f"{name}_rotated_mse"] = rotated_mse
            proxy[layer][f"{name}_relative_gain"] = (
                (direct_mse - rotated_mse) / max(direct_mse, 1e-12)
            )
    return scales, proxy


def reshape_scales(
    scales: dict[int, dict[str, Tensor]],
    rotate_map: dict[str, bool],
) -> dict[int, dict[str, Tensor]]:
    return {
        layer: {
            "q": values["q_rot" if rotate_map["q"] else "q"],
            "k": values["k_rot" if rotate_map["k"] else "k"],
            "v": values["v_rot" if rotate_map["v"] else "v"],
        }
        for layer, values in scales.items()
    }


def mean_proxy_gain(proxy: dict[int, dict[str, float]]) -> dict[str, float]:
    out = {}
    for name in ("q", "k", "v"):
        values = [layer[f"{name}_relative_gain"] for layer in proxy.values()]
        out[name] = float(sum(values) / len(values))
    out["overall"] = float((out["q"] + out["k"] + out["v"]) / 3.0)
    return out


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
    stats = {
        "elapsed_sec": float(elapsed),
        "peak_mem_mib": get_peak_mem_mib(device),
    }
    return bundle, stats


def rotate_map_from_mode(mode_name: str) -> dict[str, bool]:
    mode_map = {
        "direct_qkv": {"q": False, "k": False, "v": False},
        "rotated_q": {"q": True, "k": False, "v": False},
        "rotated_k": {"q": False, "k": True, "v": False},
        "rotated_qk": {"q": True, "k": True, "v": False},
        "rotated_qkv": {"q": True, "k": True, "v": True},
    }
    if mode_name not in mode_map:
        raise ValueError(f"Unsupported mode: {mode_name}")
    return mode_map[mode_name]


def quantize_tensor_with_plan(
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


def build_auto_selector_info(
    bundle: dict[int, dict[str, Tensor]],
    raw_scales: dict[int, dict[str, Tensor]],
    bits: int,
    attn_scale: float,
) -> dict:
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
        scales = raw_scales[layer]

        q_direct = quantize_tensor_with_plan(q, bits, scales["q"], scales["q_rot"], rotate=False)
        q_rot = quantize_tensor_with_plan(q, bits, scales["q"], scales["q_rot"], rotate=True)
        k_direct = quantize_tensor_with_plan(k, bits, scales["k"], scales["k_rot"], rotate=False)
        k_rot = quantize_tensor_with_plan(k, bits, scales["k"], scales["k_rot"], rotate=True)
        v_direct = quantize_tensor_with_plan(v, bits, scales["v"], scales["v_rot"], rotate=False)
        v_rot = quantize_tensor_with_plan(v, bits, scales["v"], scales["v_rot"], rotate=True)

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

    if mean_qk_gain <= 0.0:
        chosen_mode = "direct_qkv"
    elif early_v_gain > early_qk_gain:
        chosen_mode = "rotated_qkv"
    else:
        chosen_mode = "rotated_qk"

    return {
        "chosen_mode": chosen_mode,
        "rotate_map": rotate_map_from_mode(chosen_mode),
        "decision": {
            "mean_qk_prob_gain": mean_qk_gain,
            "early_qk_prob_gain": early_qk_gain,
            "late_qk_prob_gain": late_qk_gain,
            "early_v_out_gain": early_v_gain,
            "late_v_out_gain": late_v_gain,
            "criterion": "choose_qkv_if_early_v_out_gain_exceeds_early_qk_prob_gain_else_qk",
            "margin_early_v_minus_early_qk": float(early_v_gain - early_qk_gain),
            "margin_early_v_minus_mean_qk": float(early_v_gain - mean_qk_gain),
        },
        "layer_metrics": {str(layer): layer_metrics[layer] for layer in range(num_layers)},
        "early_layers": early_layers,
        "late_layers": late_layers,
    }


def run_model(args: argparse.Namespace, model_key: str, device: torch.device) -> dict:
    spec = MODEL_SPECS[model_key]
    batch_size = spec["batch_size"]
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

    base_model = load_model(spec["model_name"], spec["checkpoint"], device=device)
    num_layers = len(attention_modules(base_model))
    layers = list(range(num_layers))
    attn_scale = float(base_model.blocks[0].attn.scale) if hasattr(base_model, "blocks") else 1.0
    fp_stats = evaluate_with_resources(base_model, val_loader, device, f"idea2:{model_key}:fp")
    del base_model
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
            seed=args.seed + 11,
            shuffle=False,
            device=device,
        )
        calib_model = load_model(spec["model_name"], spec["checkpoint"], device=device)
        calib_bundle, calib_stats = collect_bundle_with_resources(
            calib_model,
            calib_loader,
            device,
            layers=layers,
            max_batches=max(1, calib_images // min(batch_size, 16)),
        )
        del calib_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        calib_key = str(calib_images)
        calib_result = {
            "calibration": {
                "num_images": calib_images,
                "elapsed_sec": calib_stats["elapsed_sec"],
                "peak_mem_mib": calib_stats["peak_mem_mib"],
            },
            "bits": {},
        }

        for bits in args.bits:
            raw_scales, proxy = build_scales_and_proxy(calib_bundle, bits)
            auto_selector = build_auto_selector_info(
                calib_bundle,
                raw_scales,
                bits=bits,
                attn_scale=attn_scale,
            )
            per_bit = {
                "proxy": {
                    "per_layer": proxy,
                    "mean_relative_gain": mean_proxy_gain(proxy),
                }
            }

            direct_top1 = None
            for mode_name in args.modes:
                if mode_name == "auto_place":
                    rotate_map = auto_selector["rotate_map"]
                else:
                    rotate_map = rotate_map_from_mode(mode_name)
                scales = reshape_scales(raw_scales, rotate_map)
                model_eval = load_model(spec["model_name"], spec["checkpoint"], device=device)
                replaced = patch_qkv_quant(
                    model_eval,
                    bits=bits,
                    scales=scales,
                    rotate_map=rotate_map,
                )
                stats = evaluate_with_resources(
                    model_eval,
                    val_loader,
                    device,
                    f"idea2:{model_key}:c{calib_images}:{mode_name}:b{bits}",
                )
                record = {
                    **stats,
                    "replaced_layers": replaced,
                    "rotate_map": rotate_map,
                    "delta_top1_vs_fp": float(stats["top1"] - fp_stats["top1"]),
                }
                if mode_name == "auto_place":
                    record["selector"] = auto_selector
                    record["selected_mode"] = auto_selector["chosen_mode"]
                if mode_name == "direct_qkv":
                    direct_top1 = stats["top1"]
                    record["delta_top1_vs_direct"] = 0.0
                elif direct_top1 is not None:
                    record["delta_top1_vs_direct"] = float(stats["top1"] - direct_top1)
                per_bit[mode_name] = record
                del model_eval
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            calib_result["bits"][str(bits)] = per_bit

        result["calibrations"][calib_key] = calib_result

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
            "val_images": args.val_images,
            "calib_sizes": args.calib_sizes,
            "bits": args.bits,
            "modes": args.modes,
            "models": args.models,
            "run_name": args.run_name,
        },
        "models": {},
    }

    for model_key in args.models:
        results["models"][model_key] = run_model(args, model_key, device)

    out_path = Path(args.results_dir) / f"{args.run_name}.json"
    save_json(out_path, results)
    print(f"[Done] idea2 results saved to {out_path}")


if __name__ == "__main__":
    main()
