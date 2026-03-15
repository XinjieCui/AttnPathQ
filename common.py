from __future__ import annotations

import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import timm
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

CDFQUANT_ROOT = Path("/home/cxj/cdfquant")
if str(CDFQUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(CDFQUANT_ROOT))

from cdfquant import CDFQuant  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/home/cxj/DyadicFold/data/imagenet")
DEFAULT_CHECKPOINT = Path("/home/cxj/cdfquant/weights/model.safetensors")
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class EvalStats:
    top1: float
    top5: float
    loss: float
    num_images: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, Tensor]:
    checkpoint_path = str(checkpoint_path)
    suffix = Path(checkpoint_path).suffix.lower()
    if suffix == ".safetensors":
        checkpoint = safetensors_load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    cleaned: dict[str, Tensor] = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def disable_fused_attention(model: nn.Module) -> int:
    changed = 0
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            try:
                if bool(getattr(module, "fused_attn")):
                    setattr(module, "fused_attn", False)
                    changed += 1
            except Exception:
                continue
    return changed


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _subset_indices(length: int, count: int | None, seed: int) -> list[int]:
    if count is None or count >= length:
        return list(range(length))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.randperm(length, generator=gen).tolist()
    return perm[:count]


def build_loader(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    split: str = "val",
    batch_size: int = 32,
    num_workers: int = 4,
    num_images: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
    device: torch.device | None = None,
) -> DataLoader:
    root = Path(dataset_root) / split
    dataset = datasets.ImageFolder(str(root), transform=build_transform())
    indices = _subset_indices(len(dataset), num_images, seed)
    dataset = Subset(dataset, indices)
    pin_memory = bool(device is not None and device.type == "cuda")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def load_model(
    model_name: str = "deit_small_patch16_224",
    checkpoint: str | Path | None = DEFAULT_CHECKPOINT,
    device: torch.device | str = "cuda",
) -> nn.Module:
    device = torch.device(device)
    use_pretrained = checkpoint in (None, "")
    if not use_pretrained:
        ckpt_path = Path(str(checkpoint))
        use_pretrained = not ckpt_path.exists()
    model = timm.create_model(model_name, pretrained=use_pretrained)
    if not use_pretrained:
        state_dict = _load_checkpoint_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[Info] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    disable_fused_attention(model)
    model.to(device)
    model.eval()
    return model


def attention_modules(model: nn.Module) -> list[nn.Module]:
    modules: list[nn.Module] = []
    for module in model.modules():
        if (
            module.__class__.__name__ == "Attention"
            and module.__class__.__module__.startswith("timm.")
        ):
            modules.append(module)
    if not modules:
        raise RuntimeError("No timm Attention modules found")
    return modules


def symmetric_quantize(x: Tensor, bits: int, dim: int | tuple[int, ...] | None = None) -> Tensor:
    qmax = float((1 << (bits - 1)) - 1)
    if qmax < 1:
        raise ValueError(f"bits must be >= 2 for symmetric quantization, got {bits}")
    x_fp32 = x.to(torch.float32)
    if dim is None:
        max_abs = x_fp32.abs().amax()
    else:
        max_abs = x_fp32.abs().amax(dim=dim, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=torch.finfo(torch.float32).tiny)
    q = torch.round(x_fp32 / scale).clamp(-qmax, qmax)
    return q * scale


def uniform_prob_quantize(attn_probs: Tensor, bits: int, renorm: bool = True) -> Tensor:
    levels = (1 << bits) - 1
    p_hat = torch.round(attn_probs.to(torch.float32) * levels).clamp(0, levels) / levels
    if renorm:
        row_sum = p_hat.sum(dim=-1, keepdim=True)
        p_hat = torch.where(row_sum > 0, p_hat / row_sum, attn_probs.to(torch.float32))
    return p_hat.to(attn_probs.dtype)


def transport_particle_quantize(attn_probs: Tensor, bits: int) -> Tensor:
    levels = (1 << bits) - 1
    x = attn_probs.to(torch.float32)
    flat = x.reshape(-1, x.shape[-1])
    cdf = torch.cumsum(flat, dim=-1).contiguous()
    centers = (
        (torch.arange(levels, device=x.device, dtype=torch.float32) + 0.5) / levels
    ).unsqueeze(0).expand(flat.shape[0], -1).contiguous()
    idx = torch.searchsorted(cdf, centers, right=False)
    idx = idx.clamp_max(flat.shape[-1] - 1)
    out = torch.zeros_like(flat)
    out.scatter_add_(
        dim=-1,
        index=idx,
        src=torch.full_like(idx, fill_value=1.0 / levels, dtype=torch.float32),
    )
    return out.reshape_as(x).to(attn_probs.dtype)


def mse(a: Tensor, b: Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).pow(2).mean().item())


def mae(a: Tensor, b: Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean().item())


def kl_divergence(p: Tensor, q: Tensor) -> float:
    p_fp32 = p.to(torch.float32).clamp_min(1e-8)
    q_fp32 = q.to(torch.float32).clamp_min(1e-8)
    return float((p_fp32 * (p_fp32.log() - q_fp32.log())).sum(dim=-1).mean().item())


def kurtosis(x: Tensor) -> float:
    x_fp32 = x.to(torch.float32)
    centered = x_fp32 - x_fp32.mean()
    var = centered.pow(2).mean().clamp_min(1e-8)
    return float((centered.pow(4).mean() / (var * var)).item())


def hadamard_transform(x: Tensor) -> Tensor:
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Last dim must be power of two for Hadamard transform, got {n}")
    h = x.to(torch.float32)
    dim = 1
    while dim < n:
        new_shape = h.shape[:-1] + (n // (dim * 2), 2, dim)
        h = h.reshape(new_shape)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.cat((a + b, a - b), dim=-1)
        h = h.reshape(*x.shape[:-1], n)
        dim *= 2
    return h / math.sqrt(n)


def fit_quant_error_law(bits: Iterable[int], errors: Iterable[float]) -> dict[str, float]:
    bit_list = list(bits)
    err_list = list(errors)
    x = np.array([2.0 ** (-2.0 * b) for b in bit_list], dtype=np.float64)
    y = np.array(err_list, dtype=np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coeff
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum()) + 1e-12
    return {
        "alpha": float(coeff[0]),
        "beta": float(coeff[1]),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def kcenter_indices(embeddings: Tensor, budget: int) -> list[int]:
    if budget >= embeddings.shape[0]:
        return list(range(embeddings.shape[0]))
    embs = embeddings.to(torch.float32)
    selected = [0]
    distances = torch.cdist(embs[0:1], embs).squeeze(0)
    for _ in range(1, budget):
        idx = int(torch.argmax(distances).item())
        selected.append(idx)
        distances = torch.minimum(
            distances,
            torch.cdist(embs[idx: idx + 1], embs).squeeze(0),
        )
    return selected


def mean_min_distance(embeddings: Tensor, subset_indices: list[int]) -> float:
    embs = embeddings.to(torch.float32)
    subset = embs[subset_indices]
    dist = torch.cdist(embs, subset)
    return float(dist.min(dim=1).values.mean().item())


def flatten_for_scale(x: Tensor) -> Tensor:
    return x.to(torch.float32).reshape(-1)


def estimate_percentile_scale(
    x: Tensor,
    bits: int,
    percentile: float = 0.999,
    max_samples: int = 2_000_000,
) -> Tensor:
    qmax = float((1 << (bits - 1)) - 1)
    flat = flatten_for_scale(x).abs()
    if flat.numel() > max_samples:
        step = max(1, math.ceil(flat.numel() / max_samples))
        flat = flat[::step]
    value = torch.quantile(flat, percentile)
    return torch.clamp(value / qmax, min=torch.finfo(torch.float32).tiny)


def quantize_with_scale(x: Tensor, scale: Tensor, bits: int) -> Tensor:
    qmax = float((1 << (bits - 1)) - 1)
    x_fp32 = x.to(torch.float32)
    q = torch.round(x_fp32 / scale).clamp(-qmax, qmax)
    return q * scale


class ProbabilityQuantAttentionWrapper(nn.Module):
    def __init__(self, base: nn.Module, quantizer: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.num_heads = base.num_heads
        self.scale = float(base.scale)
        self.qkv = base.qkv
        self.q_norm = base.q_norm
        self.k_norm = base.k_norm
        self.attn_drop = base.attn_drop
        self.proj = base.proj
        self.proj_drop = base.proj_drop
        self.quantizer = quantizer

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
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.quantizer(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def patch_probability_quantizer(
    model: nn.Module,
    quantizer: Callable[[Tensor], Tensor],
) -> int:
    replaced = 0

    def _patch(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if (
                child.__class__.__name__ == "Attention"
                and child.__class__.__module__.startswith("timm.")
            ):
                setattr(parent, name, ProbabilityQuantAttentionWrapper(child, quantizer))
                replaced += 1
            else:
                _patch(child)

    _patch(model)
    return replaced


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> EvalStats:
    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    total = 0
    top1 = 0.0
    top5 = 0.0
    iterator = tqdm(loader, desc=desc, leave=False)
    for images, target in iterator:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        _, pred = output.topk(5, 1, True, True)
        correct = pred.eq(target.view(-1, 1))
        batch = images.size(0)
        total += batch
        total_loss += float(loss.item()) * batch
        top1 += float(correct[:, :1].any(dim=1).float().sum().item())
        top5 += float(correct.any(dim=1).float().sum().item())
    return EvalStats(
        top1=100.0 * top1 / max(total, 1),
        top5=100.0 * top5 / max(total, 1),
        loss=total_loss / max(total, 1),
        num_images=total,
    )


def collect_attention_bundle(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layers: list[int],
    max_batches: int,
    collect_qkv: bool = False,
) -> dict[int, dict[str, Tensor]]:
    attn_list = attention_modules(model)
    storage: dict[int, dict[str, list[Tensor]]] = {
        layer: {"tokens": [], "attn_logits": [], "attn_probs": []} for layer in layers
    }
    if collect_qkv:
        for layer in layers:
            storage[layer]["q"] = []
            storage[layer]["k"] = []
            storage[layer]["v"] = []

    hooks = []

    def _make_hook(layer_idx: int):
        def _hook(module: nn.Module, args: tuple[Tensor, ...]) -> None:
            x = args[0].detach()
            bsz, num_tokens, channels = x.shape
            qkv = module.qkv(x).reshape(
                bsz,
                num_tokens,
                3,
                module.num_heads,
                channels // module.num_heads,
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = module.q_norm(q)
            k = module.k_norm(k)
            logits = (q * float(module.scale)) @ k.transpose(-2, -1)
            probs = logits.softmax(dim=-1)
            storage[layer_idx]["tokens"].append(x.cpu())
            storage[layer_idx]["attn_logits"].append(logits.cpu())
            storage[layer_idx]["attn_probs"].append(probs.cpu())
            if collect_qkv:
                storage[layer_idx]["q"].append(q.cpu())
                storage[layer_idx]["k"].append(k.cpu())
                storage[layer_idx]["v"].append(v.cpu())

        return _hook

    for layer in layers:
        hooks.append(attn_list[layer].register_forward_pre_hook(_make_hook(layer)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _target) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            _ = model(images.to(device, non_blocking=True))

    for hook in hooks:
        hook.remove()

    packed: dict[int, dict[str, Tensor]] = {}
    for layer, parts in storage.items():
        packed[layer] = {name: torch.cat(chunks, dim=0) for name, chunks in parts.items()}
    return packed


@torch.no_grad()
def collect_cls_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tensor:
    outputs: list[Tensor] = []
    model.eval()
    for batch_idx, (images, _target) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        features = model.forward_features(images)
        if features.ndim == 3:
            features = features[:, 0]
        outputs.append(features.detach().cpu())
    return torch.cat(outputs, dim=0)


def clone_model(model: nn.Module, device: torch.device) -> nn.Module:
    cloned = copy.deepcopy(model)
    cloned.to(device)
    cloned.eval()
    return cloned


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
