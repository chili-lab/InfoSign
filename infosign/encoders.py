#!/usr/bin/env python3
"""LANISTR-style multimodal backbone for the canonical OT2 V2 artifacts.

The model intentionally mirrors the *useful* part of LANISTR: each source is
encoded in its native representation before a small, presence-aware fusion
Transformer.  It does not inherit the legacy fixed shapes or run quadratic
attention over every ECG sample. ECG windows use a learned mask-aware patch
projection that preserves within-window morphology rather than an average.

The primary public API is :class:`Task2V2Model`::

    output = model(batch, task="moca_binary", disabled_groups={"retinal_flio"})

``batch`` may contain one mapping per primary modality (the preferred
contract), or the canonical V2 tensor names at top level.  Missing mappings are
valid.  Every encoder derives presence from observation/QC masks, and an
optional ``source_present`` tensor or mapping can further restrict it.

This module only defines model behavior.  Scaling, label construction, split
selection, and artifact validation belong to the V2 dataset/preprocessing
pipeline.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn


PRIMARY_MODALITY_GROUPS = (
    "clinical_data",
    "cardiac_ecg",
    "environment",
    "retinal_flio",
    "retinal_oct",
    "retinal_octa",
    "retinal_photography",
    "wearable_activity_monitor",
    "wearable_blood_glucose",
)

RETINAL_GROUPS = (
    "retinal_flio",
    "retinal_oct",
    "retinal_octa",
    "retinal_photography",
)

DEFAULT_BINARY_TASKS = (
    "moca_binary",
    "cvd_broad_binary",
    "cvd_strict_binary",
)


@dataclasses.dataclass(frozen=True)
class Task2V2ModelConfig:
    """Shape and architecture contract for :class:`Task2V2Model`.

    Feature dimensions are explicit so a task-specific clinical schema (or a
    future preprocessing revision) cannot silently reuse an old 370-feature
    model.  Sequence lengths and retinal asset/view counts remain dynamic.
    """

    clinical_dim: int
    environment_dim: int
    activity_dim: int
    cgm_dim: int
    ecg_waveform_dim: int
    ecg_numeric_dim: int
    d_model: int = 256
    temporal_layers: int = 2
    fusion_layers: int = 2
    num_heads: int = 4
    ffn_multiplier: int = 2
    dropout: float = 0.2
    environment_patch_size: int = 12
    activity_patch_size: int = 12
    cgm_patch_size: int = 12
    # 100 samples at the current 500 Hz acquisition rate = 0.2 seconds and
    # reduces a 5,500-sample record to 55 attention tokens.
    ecg_patch_size: int = 100
    retinal_backbone: str = "small_cnn"
    retinal_backbone_dim: int = 128
    retinal_weights: Optional[str] = None
    retinal_weights_path: Optional[str] = None
    retinal_backbone_frozen: bool = False
    retinal_input_size: int = 224
    retinal_input_normalization: str = "none"
    retinal_backbone_forward_batch_size: int = 64
    task_names: tuple[str, ...] = DEFAULT_BINARY_TASKS

    def __post_init__(self) -> None:
        dimension_fields = (
            "clinical_dim",
            "environment_dim",
            "activity_dim",
            "cgm_dim",
            "ecg_waveform_dim",
            "ecg_numeric_dim",
            "d_model",
            "retinal_backbone_dim",
            "retinal_input_size",
            "retinal_backbone_forward_batch_size",
        )
        for name in dimension_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "environment_patch_size",
            "activity_patch_size",
            "cgm_patch_size",
            "ecg_patch_size",
            "temporal_layers",
            "fusion_layers",
            "num_heads",
            "ffn_multiplier",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.task_names:
            raise ValueError("task_names cannot be empty")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must be unique")
        if not isinstance(self.retinal_backbone_frozen, bool):
            raise ValueError("retinal_backbone_frozen must be boolean")

        backbone = str(self.retinal_backbone).strip().lower()
        normalization = str(self.retinal_input_normalization).strip().lower()
        weights = (
            None
            if self.retinal_weights is None
            else str(self.retinal_weights).strip().upper()
        )
        weights_path = (
            None
            if self.retinal_weights_path is None
            else str(self.retinal_weights_path).strip()
        )
        if backbone not in {"small_cnn", "resnet18"}:
            raise ValueError(
                "retinal_backbone must be 'small_cnn' or 'resnet18'"
            )
        if weights_path == "":
            raise ValueError("retinal_weights_path cannot be empty")
        if weights is not None and weights_path is not None:
            raise ValueError(
                "set at most one of retinal_weights and retinal_weights_path"
            )
        if weights not in {None, "IMAGENET1K_V1"}:
            raise ValueError(
                "retinal_weights must be null or 'IMAGENET1K_V1'; "
                "DEFAULT is intentionally unsupported because it is version-dependent"
            )
        if backbone == "small_cnn":
            if weights is not None or weights_path is not None:
                raise ValueError("small_cnn does not accept pretrained weights")
            if normalization != "none":
                raise ValueError(
                    "small_cnn requires retinal_input_normalization='none'"
                )
        else:
            if self.retinal_backbone_dim != 512:
                raise ValueError("resnet18 has a fixed retinal_backbone_dim of 512")
            if self.retinal_input_size != 224:
                raise ValueError(
                    "resnet18 requires 224x224 V2 sidecar views "
                    "(retinal_input_size=224)"
                )
            if normalization != "imagenet1k_v1":
                raise ValueError(
                    "resnet18 requires retinal_input_normalization="
                    "'imagenet1k_v1'"
                )


def _first(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _bundle(batch: Mapping[str, Any], group: str) -> Mapping[str, Any]:
    value = batch.get(group)
    return value if isinstance(value, Mapping) else batch


def _require_rank(tensor: torch.Tensor, rank: int, name: str) -> None:
    if tensor.ndim != rank:
        raise ValueError(
            f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}"
        )


def _finite_values_and_mask(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite values and a boolean mask broadcast to ``values``."""

    finite = torch.isfinite(values)
    clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is None:
        return clean, finite
    declared = mask.to(device=values.device).bool()
    while declared.ndim < values.ndim:
        declared = declared.unsqueeze(-1)
    try:
        declared = torch.broadcast_to(declared, values.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to "
            f"values shape {tuple(values.shape)}"
        ) from exc
    return clean, finite & declared


def _combine_masks(
    values: torch.Tensor,
    masks: Iterable[Optional[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    clean, valid = _finite_values_and_mask(values, None)
    for mask in masks:
        if mask is None:
            continue
        _, declared = _finite_values_and_mask(values, mask)
        valid &= declared
    return clean, valid


def _sinusoidal_positions(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dynamic sinusoidal positions; no fixed sequence length is assumed."""

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if width > 1:
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


@contextlib.contextmanager
def _safe_transformer_context(reference: torch.Tensor):
    """Avoid PyTorch's incompatible CPU-bf16 native MHA inference fastpath.

    Some supported PyTorch releases enter the fused eval/no-grad fastpath with
    a bfloat16 activation but float32 weights, then fail inside a matrix
    multiply.  Disabling only that fastpath for the duration of a CPU autocast
    Transformer call keeps ordinary CPU/CUDA execution unchanged.
    """

    try:
        cpu_autocast = bool(torch.is_autocast_enabled("cpu"))
    except TypeError:  # Older supported PyTorch signature.
        cpu_autocast = bool(torch.is_autocast_cpu_enabled())
    backend = getattr(torch.backends, "mha", None)
    needs_guard = bool(
        reference.device.type == "cpu"
        and cpu_autocast
        and not torch.is_grad_enabled()
        and backend is not None
        and hasattr(backend, "get_fastpath_enabled")
        and hasattr(backend, "set_fastpath_enabled")
        and backend.get_fastpath_enabled()
    )
    if needs_guard:
        backend.set_fastpath_enabled(False)
    try:
        yield
    finally:
        if needs_guard:
            backend.set_fastpath_enabled(True)


class MaskAwareVectorEncoder(nn.Module):
    """MLP that keeps observed-at-center distinct from unobserved."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        hidden = int(hidden_dim or max(d_model * 2, min(512, input_dim)))
        self.net = nn.Sequential(
            nn.Linear(2 * self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_rank(values, 2, "vector values")
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} vector features, got {values.shape[-1]}"
            )
        clean, valid = _finite_values_and_mask(values, mask)
        valid_float = valid.to(dtype=clean.dtype)
        encoded = self.net(torch.cat([clean * valid_float, valid_float], dim=-1))
        present = valid.any(dim=-1)
        return encoded, present


class MaskedTemporalPatchEncoder(nn.Module):
    """Mask-weighted patch pooling followed by a compact Transformer.

    Each patch stores the mean observed value and its per-channel support.  For
    a 5,500-sample ECG with patch size 100, self-attention sees 55 patch tokens
    rather than 5,500 raw samples.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        *,
        patch_size: int,
        num_heads: int,
        num_layers: int,
        ffn_multiplier: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.patch_projection = nn.Sequential(
            nn.Linear(2 * self.feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def patchify(
        self,
        values: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require_rank(values, 3, "temporal values")
        if values.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} temporal features, "
                f"got {values.shape[-1]}"
            )
        clean, valid = _combine_masks(values, (feature_mask, time_mask))
        batch_size, time_steps, feature_dim = clean.shape
        n_patches = max(1, math.ceil(time_steps / self.patch_size))
        padded_steps = n_patches * self.patch_size
        if padded_steps != time_steps:
            pad = padded_steps - time_steps
            clean = torch.cat(
                [clean, clean.new_zeros(batch_size, pad, feature_dim)], dim=1
            )
            valid = torch.cat(
                [
                    valid,
                    torch.zeros(
                        batch_size,
                        pad,
                        feature_dim,
                        device=valid.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
        clean = clean.reshape(
            batch_size, n_patches, self.patch_size, feature_dim
        )
        valid = valid.reshape(
            batch_size, n_patches, self.patch_size, feature_dim
        )
        weights = valid.to(dtype=clean.dtype)
        counts = weights.sum(dim=2)
        means = (clean * weights).sum(dim=2) / counts.clamp(min=1.0)
        support = counts / float(self.patch_size)
        patch_valid = valid.any(dim=(2, 3))
        patch_input = torch.cat([means, support], dim=-1)
        return patch_input, patch_valid, valid

    def forward(
        self,
        values: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
        *,
        return_patch_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        patch_input, patch_valid, _ = self.patchify(
            values, feature_mask=feature_mask, time_mask=time_mask
        )
        patch_tokens = self.patch_projection(patch_input)
        patch_tokens = patch_tokens + _sinusoidal_positions(
            patch_tokens.shape[1],
            patch_tokens.shape[2],
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        ).unsqueeze(0)
        cls = self.cls_token.expand(values.shape[0], -1, -1)
        sequence = torch.cat([cls, patch_tokens], dim=1)
        padding = torch.cat(
            [
                torch.zeros(
                    values.shape[0], 1, device=values.device, dtype=torch.bool
                ),
                ~patch_valid,
            ],
            dim=1,
        )
        with _safe_transformer_context(sequence):
            output = self.transformer(sequence, src_key_padding_mask=padding)
        output = self.output_norm(output)
        token = output[:, 0]
        present = patch_valid.any(dim=1)
        if return_patch_tokens:
            return token, present, output[:, 1:], patch_valid
        return token, present


class MaskedWaveformPatchEncoder(MaskedTemporalPatchEncoder):
    """Preserve within-patch waveform morphology before temporal attention.

    Longitudinal sensor channels can be summarized by a support-weighted mean
    over a short patch.  Doing that to an ECG would erase QRS morphology.  An
    ECG patch therefore exposes every normalized sample and every corresponding
    validity bit to a learned projection.  Attention still sees only 55 tokens
    for a 5,500-sample record with 100-sample patches.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        *,
        patch_size: int,
        num_heads: int,
        num_layers: int,
        ffn_multiplier: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            feature_dim,
            d_model,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )
        self.patch_projection = nn.Sequential(
            nn.Linear(2 * self.patch_size * self.feature_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def patchify(
        self,
        values: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require_rank(values, 3, "waveform values")
        if values.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} waveform leads, got {values.shape[-1]}"
            )
        clean, valid = _combine_masks(values, (feature_mask, time_mask))
        batch_size, time_steps, feature_dim = clean.shape
        n_patches = max(1, math.ceil(time_steps / self.patch_size))
        padded_steps = n_patches * self.patch_size
        if padded_steps != time_steps:
            pad = padded_steps - time_steps
            clean = torch.cat(
                [clean, clean.new_zeros(batch_size, pad, feature_dim)], dim=1
            )
            valid = torch.cat(
                [
                    valid,
                    torch.zeros(
                        batch_size,
                        pad,
                        feature_dim,
                        device=valid.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
        clean = clean.reshape(batch_size, n_patches, self.patch_size, feature_dim)
        valid = valid.reshape(batch_size, n_patches, self.patch_size, feature_dim)
        valid_float = valid.to(dtype=clean.dtype)
        # Flatten in time-major, lead-minor order.  The paired validity vector
        # keeps a true normalized zero distinct from an unobserved storage zero.
        patch_input = torch.cat(
            [
                (clean * valid_float).flatten(start_dim=2),
                valid_float.flatten(start_dim=2),
            ],
            dim=-1,
        )
        patch_valid = valid.any(dim=(2, 3))
        return patch_input, patch_valid, valid


class ECGEncoder(nn.Module):
    """Encode waveform patches and the numeric ECG report separately."""

    def __init__(self, config: Task2V2ModelConfig) -> None:
        super().__init__()
        self.waveform = MaskedWaveformPatchEncoder(
            config.ecg_waveform_dim,
            config.d_model,
            patch_size=config.ecg_patch_size,
            num_heads=config.num_heads,
            num_layers=config.temporal_layers,
            ffn_multiplier=config.ffn_multiplier,
            dropout=config.dropout,
        )
        self.numeric = MaskAwareVectorEncoder(
            config.ecg_numeric_dim,
            config.d_model,
            dropout=config.dropout,
        )
        self.combine = nn.Sequential(
            nn.Linear(2 * config.d_model + 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_feature_mask: Optional[torch.Tensor],
        sample_mask: Optional[torch.Tensor],
        lead_mask: Optional[torch.Tensor],
        numeric: torch.Tensor,
        numeric_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A two-dimensional ECG mask can describe time [B,T] or leads [B,C].
        # Expand the lead mask explicitly so it cannot be mistaken for time.
        expanded_lead_mask = None
        if lead_mask is not None:
            expanded_lead_mask = lead_mask
            if lead_mask.ndim == 2:
                expanded_lead_mask = lead_mask.unsqueeze(1)
        waveform_clean, waveform_valid = _combine_masks(
            waveform, (waveform_feature_mask, sample_mask, expanded_lead_mask)
        )
        wave_token, wave_present = self.waveform(
            waveform_clean, feature_mask=waveform_valid
        )
        numeric_token, numeric_present = self.numeric(numeric, numeric_mask)
        wave_gate = wave_present.to(dtype=wave_token.dtype).unsqueeze(-1)
        numeric_gate = numeric_present.to(dtype=numeric_token.dtype).unsqueeze(-1)
        combined = self.combine(
            torch.cat(
                [
                    wave_token * wave_gate,
                    numeric_token * numeric_gate,
                    wave_gate,
                    numeric_gate,
                ],
                dim=-1,
            )
        )
        return combined, wave_present | numeric_present


class SharedRetinalCNN(nn.Module):
    """Compact from-scratch smoke backbone shared by all retinal groups.

    This remains the dependency-free default for tests and pipeline smoke
    runs.  It is not presented as a scientifically comparable replacement for
    an ImageNet-pretrained backbone in retinal-modality importance studies.
    """

    def __init__(self, output_dim: int, *, frozen: bool = False) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.frozen = bool(frozen)
        widths = (32, 64, self.output_dim)
        layers: list[nn.Module] = []
        in_channels = 3
        for width in widths:
            groups = min(8, width)
            while width % groups:
                groups -= 1
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, width, kernel_size=3, stride=2, padding=1
                    ),
                    nn.GroupNorm(groups, width),
                    nn.GELU(),
                ]
            )
            in_channels = width
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.network = nn.Sequential(*layers)
        if self.frozen:
            self.network.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del pixel_mask  # Already applied by the hierarchical encoder.
        return self.network(images)

    def provenance(self) -> Dict[str, Any]:
        return {
            "backbone": "small_cnn",
            "torch_version": str(torch.__version__),
            "weights_source": "random_initialization",
            "weights_identifier": None,
            "weights_path": None,
            "weights_sha256": None,
            "output_dim": self.output_dim,
            "frozen": self.frozen,
            "input_size": None,
            "input_normalization": "none",
        }


class ImageNet1KInputNormalization(nn.Module):
    """Fixed ImageNet-1K normalization for V2 sidecar views.

    Sidecar images have already been aspect-preserving resized and padded to
    224x224.  Repeating torchvision's resize/center-crop would alter geometry,
    so this layer performs only the weight-matched channel normalization.  A
    pixel mask is reapplied *after* normalization, keeping padded/invalid
    support at zero instead of turning input zero into a negative constant.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, *, input_size: int = 224) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.register_buffer(
            "mean", torch.tensor(self.MEAN, dtype=torch.float32).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self.STD, dtype=torch.float32).reshape(1, 3, 1, 1)
        )

    def forward(
        self,
        images: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _require_rank(images, 4, "retinal backbone images")
        if images.shape[1] != 3:
            raise ValueError("retinal backbone images must have three channels")
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                "ImageNet retinal input must be fixed at "
                f"{self.input_size}x{self.input_size}, got "
                f"{tuple(images.shape[-2:])}"
            )
        if not images.is_floating_point():
            raise ValueError("retinal sidecar images must be floating point in [0,1]")
        if not torch.isfinite(images).all():
            raise ValueError("retinal sidecar images contain non-finite values")

        support: Optional[torch.Tensor] = None
        values_for_range = images
        if pixel_mask is not None:
            support = pixel_mask.to(device=images.device).bool()
            try:
                support = torch.broadcast_to(support, images.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"retinal pixel mask {tuple(pixel_mask.shape)} cannot "
                    f"broadcast to {tuple(images.shape)}"
                ) from exc
            if support.any():
                values_for_range = images[support]
            else:
                values_for_range = images.new_zeros(1)

        minimum = float(values_for_range.detach().amin().item())
        maximum = float(values_for_range.detach().amax().item())
        # float16 sidecars represent endpoints exactly; a small tolerance only
        # protects harmless conversion noise, never arbitrary rescaling.
        if minimum < -1e-4 or maximum > 1.0001:
            raise ValueError(
                "retinal sidecar images must lie in [0,1] before ImageNet "
                f"normalization; observed [{minimum:.6g}, {maximum:.6g}]"
            )
        mean = self.mean.to(dtype=images.dtype)
        std = self.std.to(dtype=images.dtype)
        normalized = (images - mean) / std
        if support is not None:
            normalized = normalized * support.to(dtype=normalized.dtype)
        return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    """Accept a raw torchvision state dict or an explicit state_dict wrapper."""

    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("local ResNet-18 weights must contain a non-empty state dict")
    state = dict(payload)
    if all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("local ResNet-18 state dict must map strings to tensors")
    return state


def _load_resnet18_network(
    *,
    weights: Optional[str],
    weights_path: Optional[str],
) -> tuple[nn.Module, Dict[str, Any]]:
    """Build ResNet-18 without ever silently dropping requested weights."""

    try:
        import torchvision  # type: ignore
        from torchvision import models  # type: ignore
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "retinal_backbone='resnet18' requires a working torchvision "
            "installation compatible with the installed torch"
        ) from exc

    metadata: Dict[str, Any] = {
        "torchvision_version": str(getattr(torchvision, "__version__", "unknown")),
        "weights_source": "random_initialization",
        "weights_identifier": None,
        "weights_path": None,
        "weights_sha256": None,
    }
    if weights_path is not None:
        resolved = Path(weights_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"local ResNet-18 weights do not exist: {resolved}")
        network = models.resnet18(weights=None)
        try:
            payload = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "safe local-weight loading requires torch.load(weights_only=True)"
            ) from exc
        state = _unwrap_state_dict(payload)
        try:
            network.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"local weights are not a strict torchvision ResNet-18 state dict: {resolved}"
            ) from exc
        metadata.update(
            {
                "weights_source": "local_file",
                "weights_identifier": None,
                "weights_path": str(resolved),
                "weights_sha256": _sha256_file(resolved),
            }
        )
    elif weights is not None:
        if weights != "IMAGENET1K_V1":
            raise ValueError(f"unsupported ResNet-18 weights identifier: {weights!r}")
        enum_value = models.ResNet18_Weights.IMAGENET1K_V1
        try:
            network = models.resnet18(weights=enum_value)
        except Exception as exc:
            raise RuntimeError(
                "requested torchvision ResNet18_Weights.IMAGENET1K_V1 could "
                "not be loaded; no random-weight fallback was used. Pre-cache "
                "the torchvision weights or provide weights_path."
            ) from exc
        metadata.update(
            {
                "weights_source": "torchvision_enum",
                "weights_identifier": "ResNet18_Weights.IMAGENET1K_V1",
                "weights_url": str(getattr(enum_value, "url", "")) or None,
            }
        )
    else:
        network = models.resnet18(weights=None)
    network.fc = nn.Identity()
    return network, metadata


class ResNet18RetinalBackbone(nn.Module):
    """Torchvision ResNet-18 with a strict offline/auditable weight contract."""

    output_dim = 512

    def __init__(
        self,
        *,
        weights: Optional[str],
        weights_path: Optional[str],
        frozen: bool,
        input_size: int,
    ) -> None:
        super().__init__()
        self.frozen = bool(frozen)
        self.normalization = ImageNet1KInputNormalization(input_size=input_size)
        self.network, metadata = _load_resnet18_network(
            weights=weights,
            weights_path=weights_path,
        )
        self._provenance = {
            "backbone": "resnet18",
            "torch_version": str(torch.__version__),
            **metadata,
            "output_dim": self.output_dim,
            "frozen": self.frozen,
            "input_size": int(input_size),
            "input_normalization": "imagenet1k_v1",
            "pixel_mask_reapplied_after_normalization": True,
        }
        if self.frozen:
            self.network.requires_grad_(False)
            self.network.eval()

    def train(self, mode: bool = True) -> "ResNet18RetinalBackbone":
        super().train(mode)
        # Freezing includes BatchNorm running statistics, not only gradients.
        if self.frozen:
            self.network.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.network(self.normalization(images, pixel_mask))

    def provenance(self) -> Dict[str, Any]:
        return dict(self._provenance)


def build_retinal_backbone(config: Task2V2ModelConfig) -> nn.Module:
    backbone = str(config.retinal_backbone).strip().lower()
    if backbone == "small_cnn":
        return SharedRetinalCNN(
            config.retinal_backbone_dim,
            frozen=config.retinal_backbone_frozen,
        )
    if backbone == "resnet18":
        weights = (
            None
            if config.retinal_weights is None
            else str(config.retinal_weights).strip().upper()
        )
        return ResNet18RetinalBackbone(
            weights=weights,
            weights_path=config.retinal_weights_path,
            frozen=config.retinal_backbone_frozen,
            input_size=config.retinal_input_size,
        )
    raise ValueError(f"unsupported retinal backbone: {config.retinal_backbone!r}")


class MaskedAttentionPool(nn.Module):
    """Attention pooling that returns exact zero for an all-masked set."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1)
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 3 or mask.shape != tokens.shape[:-1]:
            raise ValueError(
                f"pool mask {tuple(mask.shape)} must match token prefix "
                f"{tuple(tokens.shape[:-1])}"
            )
        mask = mask.bool()
        scores = self.score(tokens).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return (tokens * weights.unsqueeze(-1)).sum(dim=-2)


class HierarchicalRetinalEncoder(nn.Module):
    """Shared backbone view -> asset -> retinal-modality set encoder."""

    def __init__(self, config: Task2V2ModelConfig) -> None:
        super().__init__()
        self.backbone = build_retinal_backbone(config)
        self.backbone_forward_batch_size = int(
            config.retinal_backbone_forward_batch_size
        )
        self.view_projection = nn.Sequential(
            nn.Linear(config.retinal_backbone_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.view_pool = MaskedAttentionPool(config.d_model)
        self.asset_pools = nn.ModuleDict(
            {group: MaskedAttentionPool(config.d_model) for group in RETINAL_GROUPS}
        )
        self.group_projections = nn.ModuleDict(
            {
                group: nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.LayerNorm(config.d_model),
                )
                for group in RETINAL_GROUPS
            }
        )

    def forward(
        self,
        group: str,
        images: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        asset_mask: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if group not in RETINAL_GROUPS:
            raise ValueError(f"unknown retinal group {group!r}")
        _require_rank(images, 6, f"{group} images")
        if images.shape[3] != 3:
            raise ValueError(
                f"{group} images must have three channels at axis 3, "
                f"got {tuple(images.shape)}"
            )
        batch_size, n_assets, n_views, channels, height, width = images.shape
        finite_input = torch.isfinite(images)
        clean = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
        if asset_mask is None:
            asset_valid = torch.ones(
                batch_size, n_assets, device=images.device, dtype=torch.bool
            )
        else:
            asset_valid = asset_mask.to(device=images.device).bool()
            if asset_valid.shape != (batch_size, n_assets):
                raise ValueError(
                    f"{group} asset mask must have shape {(batch_size, n_assets)}, "
                    f"got {tuple(asset_valid.shape)}"
                )
        if view_mask is None:
            view_valid = torch.ones(
                batch_size,
                n_assets,
                n_views,
                device=images.device,
                dtype=torch.bool,
            )
        else:
            view_valid = view_mask.to(device=images.device).bool()
            if view_valid.shape != (batch_size, n_assets, n_views):
                raise ValueError(
                    f"{group} view mask must have shape "
                    f"{(batch_size, n_assets, n_views)}, "
                    f"got {tuple(view_valid.shape)}"
                )
        view_valid &= asset_valid.unsqueeze(-1)

        declared: Optional[torch.Tensor] = None
        if pixel_mask is not None:
            declared = pixel_mask.to(device=images.device).bool()
            while declared.ndim < clean.ndim:
                declared = declared.unsqueeze(-3)
            try:
                declared = torch.broadcast_to(declared, clean.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"{group} pixel mask {tuple(pixel_mask.shape)} cannot "
                    f"broadcast to {tuple(clean.shape)}"
                ) from exc
            clean = clean * declared.to(dtype=clean.dtype)
            view_valid &= declared.any(dim=(3, 4, 5))

        if isinstance(self.backbone, ResNet18RetinalBackbone):
            supported = finite_input if declared is None else finite_input | ~declared
            if not bool(supported.all().item()):
                raise ValueError(
                    f"{group} contains non-finite values under a true pixel mask"
                )

        flat = clean.reshape(
            batch_size * n_assets * n_views, channels, height, width
        )
        flat_pixel_mask = (
            None
            if declared is None
            else declared.reshape(
                batch_size * n_assets * n_views, channels, height, width
            )
        )
        flat_valid = view_valid.reshape(-1)
        flat_tokens = clean.new_zeros(
            flat.shape[0], self.view_projection[0].out_features
        )
        valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
        if valid_indices.numel():
            encoded_chunks = []
            for indices in valid_indices.split(self.backbone_forward_batch_size):
                chunk_mask = (
                    None
                    if flat_pixel_mask is None
                    else flat_pixel_mask.index_select(0, indices)
                )
                encoded_chunks.append(
                    self.backbone(flat.index_select(0, indices), chunk_mask)
                )
            encoded = self.view_projection(torch.cat(encoded_chunks, dim=0))
            flat_tokens = flat_tokens.index_copy(0, valid_indices, encoded)
        view_tokens = flat_tokens.reshape(
            batch_size, n_assets, n_views, flat_tokens.shape[-1]
        )
        asset_tokens = self.view_pool(view_tokens, view_valid)
        effective_assets = asset_valid & view_valid.any(dim=-1)
        group_token = self.asset_pools[group](asset_tokens, effective_assets)
        group_token = self.group_projections[group](group_token)
        return group_token, effective_assets.any(dim=-1)

    def backbone_provenance(self) -> Dict[str, Any]:
        provenance = getattr(self.backbone, "provenance", None)
        if not callable(provenance):
            raise RuntimeError("retinal backbone does not expose provenance")
        result = dict(provenance())
        result["shared_across_retinal_groups"] = True
        result["retinal_groups"] = list(RETINAL_GROUPS)
        result["backbone_forward_batch_size"] = self.backbone_forward_batch_size
        result["hierarchy"] = "view_then_asset_then_retinal_group"
        result["view_pool"] = "masked_attention"
        result["asset_pool"] = "masked_attention"
        return result


class PresenceAwareFusion(nn.Module):
    """Fuse exactly one token per primary modality plus a patient CLS token."""

    def __init__(self, config: Task2V2ModelConfig) -> None:
        super().__init__()
        n_sources = len(PRIMARY_MODALITY_GROUPS)
        self.source_embeddings = nn.Parameter(
            torch.zeros(1, n_sources, config.d_model)
        )
        self.missing_tokens = nn.Parameter(
            torch.zeros(1, n_sources, config.d_model)
        )
        self.patient_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.source_embeddings, std=0.02)
        nn.init.trunc_normal_(self.missing_tokens, std=0.02)
        nn.init.trunc_normal_(self.patient_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * config.ffn_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.fusion_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_tokens.ndim != 3:
            raise ValueError("source_tokens must have shape [B, S, D]")
        if source_tokens.shape[1] != len(PRIMARY_MODALITY_GROUPS):
            raise ValueError(
                f"expected {len(PRIMARY_MODALITY_GROUPS)} source tokens, "
                f"got {source_tokens.shape[1]}"
            )
        if source_present.shape != source_tokens.shape[:2]:
            raise ValueError("source_present must have shape [B, S]")
        present = source_present.bool().unsqueeze(-1)
        missing = self.missing_tokens.expand(source_tokens.shape[0], -1, -1)
        fused_sources = torch.where(present, source_tokens, missing)
        fused_sources = fused_sources + self.source_embeddings
        sequence = torch.cat(
            [self.patient_token.expand(source_tokens.shape[0], -1, -1), fused_sources],
            dim=1,
        )
        with _safe_transformer_context(sequence):
            output = self.output_norm(self.transformer(sequence))
        return output[:, 0], output[:, 1:]


class BinaryTaskHead(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding).squeeze(-1)


class Task2V2Model(nn.Module):
    """Nine-source, mask-aware, extensible LANISTR-style OT2 model."""

    def __init__(
        self, config: Task2V2ModelConfig | Mapping[str, Any]
    ) -> None:
        super().__init__()
        if isinstance(config, Mapping):
            config = Task2V2ModelConfig(**dict(config))
        if not isinstance(config, Task2V2ModelConfig):
            raise TypeError("config must be Task2V2ModelConfig or a mapping")
        self.config = config
        self.source_order = PRIMARY_MODALITY_GROUPS
        self.source_index = {name: i for i, name in enumerate(self.source_order)}

        self.clinical_encoder = MaskAwareVectorEncoder(
            config.clinical_dim, config.d_model, dropout=config.dropout
        )
        self.ecg_encoder = ECGEncoder(config)
        self.temporal_encoders = nn.ModuleDict(
            {
                "environment": MaskedTemporalPatchEncoder(
                    config.environment_dim,
                    config.d_model,
                    patch_size=config.environment_patch_size,
                    num_heads=config.num_heads,
                    num_layers=config.temporal_layers,
                    ffn_multiplier=config.ffn_multiplier,
                    dropout=config.dropout,
                ),
                "wearable_activity_monitor": MaskedTemporalPatchEncoder(
                    config.activity_dim,
                    config.d_model,
                    patch_size=config.activity_patch_size,
                    num_heads=config.num_heads,
                    num_layers=config.temporal_layers,
                    ffn_multiplier=config.ffn_multiplier,
                    dropout=config.dropout,
                ),
                "wearable_blood_glucose": MaskedTemporalPatchEncoder(
                    config.cgm_dim,
                    config.d_model,
                    patch_size=config.cgm_patch_size,
                    num_heads=config.num_heads,
                    num_layers=config.temporal_layers,
                    ffn_multiplier=config.ffn_multiplier,
                    dropout=config.dropout,
                ),
            }
        )
        self.retinal_encoder = HierarchicalRetinalEncoder(config)
        self.fusion = PresenceAwareFusion(config)
        self.heads = nn.ModuleDict(
            {
                task: BinaryTaskHead(config.d_model, config.dropout)
                for task in config.task_names
            }
        )

    def retinal_backbone_provenance(self) -> Dict[str, Any]:
        """Resolved initialization/preprocessing contract for checkpoints."""

        return self.retinal_encoder.backbone_provenance()

    def add_binary_task(self, task: str) -> None:
        """Register a future audited binary downstream task head."""

        task = str(task)
        if not task:
            raise ValueError("task name cannot be empty")
        if task in self.heads:
            raise ValueError(f"task {task!r} is already registered")
        self.heads[task] = BinaryTaskHead(self.config.d_model, self.config.dropout)

    def _batch_reference(self, batch: Mapping[str, Any]) -> tuple[int, torch.device]:
        def tensors(value: Any) -> Iterable[torch.Tensor]:
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, Mapping):
                for nested in value.values():
                    yield from tensors(nested)
            elif isinstance(value, (tuple, list)):
                for nested in value:
                    yield from tensors(nested)

        for tensor in tensors(batch):
            if tensor.ndim > 0:
                return int(tensor.shape[0]), tensor.device
        raise ValueError("batch must contain at least one batched tensor")

    def _declared_presence(
        self,
        batch: Mapping[str, Any],
        group: str,
        inferred: torch.Tensor,
    ) -> torch.Tensor:
        declared_all = batch.get("source_present")
        declared: Optional[torch.Tensor] = None
        if isinstance(declared_all, Mapping):
            candidate = declared_all.get(group)
            if isinstance(candidate, torch.Tensor):
                declared = candidate
        elif isinstance(declared_all, torch.Tensor):
            if declared_all.ndim != 2 or declared_all.shape[1] != len(
                self.source_order
            ):
                raise ValueError("source_present tensor must have shape [B, 9]")
            declared = declared_all[:, self.source_index[group]]

        bundle = batch.get(group)
        if isinstance(bundle, Mapping):
            candidate = _first(bundle, ("source_present", "present", f"has_{group}"))
            if isinstance(candidate, torch.Tensor):
                declared = candidate
        if declared is None:
            aliases = {
                "clinical_data": ("has_clinical",),
                "cardiac_ecg": ("has_ecg",),
                "environment": ("has_environment",),
                "retinal_flio": ("has_retinal_flio",),
                "retinal_oct": ("has_retinal_oct",),
                "retinal_octa": ("has_retinal_octa",),
                "retinal_photography": ("has_retinal_photography",),
                "wearable_activity_monitor": ("has_activity",),
                "wearable_blood_glucose": ("has_cgm",),
            }
            candidate = _first(batch, aliases[group])
            if isinstance(candidate, torch.Tensor):
                declared = candidate
        if declared is None:
            return inferred.bool()
        declared = declared.to(device=inferred.device).bool().reshape(-1)
        if declared.shape != inferred.shape:
            raise ValueError(
                f"declared presence for {group} has shape {tuple(declared.shape)}, "
                f"expected {tuple(inferred.shape)}"
            )
        # The collator derives acquisition availability from the validated
        # master index, while ``inferred`` represents usable support after all
        # observation/QC masks.  A true measured zero remains present because
        # its mask is true; an acquired but entirely QC-rejected source must
        # not be passed to fusion as an observed token.
        return declared & inferred.bool()

    def _clinical(
        self, batch: Mapping[str, Any]
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        values = _first(
            _bundle(batch, "clinical_data"), ("values", "clinical_values", "clinical")
        )
        if not isinstance(values, torch.Tensor):
            return None
        mask = _first(
            _bundle(batch, "clinical_data"),
            ("mask", "feature_mask", "clinical_mask", "clinical_feature_mask"),
        )
        return self.clinical_encoder(values, mask)

    def _ecg(
        self, batch: Mapping[str, Any]
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        source = _bundle(batch, "cardiac_ecg")
        waveform = _first(source, ("waveform", "values", "ecg_waveform"))
        numeric = _first(source, ("numeric", "ecg_numeric"))
        if not isinstance(waveform, torch.Tensor) or not isinstance(
            numeric, torch.Tensor
        ):
            return None
        return self.ecg_encoder(
            waveform,
            _first(source, ("value_mask", "feature_mask", "ecg_value_mask")),
            _first(source, ("sample_mask", "ecg_sample_mask", "ecg_time_mask")),
            _first(source, ("lead_mask", "ecg_lead_mask", "quality_lead_mask")),
            numeric,
            _first(source, ("numeric_mask", "ecg_numeric_mask")),
        )

    def _temporal(
        self,
        batch: Mapping[str, Any],
        group: str,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        source = _bundle(batch, group)
        aliases = {
            "environment": (
                ("values", "environment"),
                ("feature_mask", "environment_feature_mask"),
                ("time_mask", "slot_mask", "environment_slot_mask"),
            ),
            "wearable_activity_monitor": (
                ("values", "activity"),
                ("feature_mask", "activity_feature_mask"),
                (
                    "time_mask",
                    "valid_mask",
                    "activity_valid_mask",
                    "activity_observed_mask",
                ),
            ),
            "wearable_blood_glucose": (
                ("values", "cgm"),
                ("feature_mask", "cgm_feature_mask"),
                ("time_mask", "valid_mask", "cgm_valid_mask", "cgm_observed_mask"),
            ),
        }
        value_names, feature_names, time_names = aliases[group]
        values = _first(source, value_names)
        if not isinstance(values, torch.Tensor):
            return None
        time_mask = _first(source, time_names)
        if group == "wearable_activity_monitor":
            observed = _first(source, ("activity_observed_mask", "observed_mask"))
            valid = _first(source, ("activity_valid_mask", "valid_mask"))
            if isinstance(observed, torch.Tensor) and isinstance(valid, torch.Tensor):
                time_mask = observed.bool() & valid.bool()
        elif group == "wearable_blood_glucose":
            observed = _first(source, ("cgm_observed_mask", "observed_mask"))
            valid = _first(source, ("cgm_valid_mask", "valid_mask"))
            if isinstance(observed, torch.Tensor) and isinstance(valid, torch.Tensor):
                time_mask = observed.bool() & valid.bool()
        return self.temporal_encoders[group](
            values,
            feature_mask=_first(source, feature_names),
            time_mask=time_mask,
        )

    def _retinal(
        self,
        batch: Mapping[str, Any],
        group: str,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        source = _bundle(batch, group)
        images = _first(
            source,
            (
                group,
                "values",
                "images",
                "retinal_assets",
                "model_views",
                f"{group}_images",
            ),
        )
        if not isinstance(images, torch.Tensor):
            return None
        if images.ndim == 5:
            images = images.unsqueeze(1)
        return self.retinal_encoder(
            group,
            images,
            view_mask=_first(
                source,
                (
                    "view_mask",
                    "retinal_view_mask",
                    "view_quality_mask",
                    f"{group}_view_mask",
                ),
            ),
            asset_mask=_first(
                source,
                ("asset_mask", "retinal_asset_mask", f"{group}_asset_mask"),
            ),
            pixel_mask=_first(
                source,
                ("pixel_mask", "model_pixel_mask", f"{group}_pixel_mask"),
            ),
        )

    def encode_sources(
        self,
        batch: Mapping[str, Any],
        *,
        disabled_groups: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode exactly one native token and presence bit per source.

        ``disabled_groups`` is a true model-level ablation: disabled encoders
        are not evaluated and their presence is forced false.  Fusion is kept
        out of this method deliberately, so a frozen source-token cache can
        reuse the *same* public model path as ordinary forward evaluation.
        """

        disabled = frozenset(disabled_groups or ())
        unknown = disabled.difference(self.source_order)
        if unknown:
            raise ValueError(f"unknown disabled_groups: {sorted(unknown)}")
        batch_size, device = self._batch_reference(batch)
        parameter = next(self.parameters())
        dtype = parameter.dtype
        tokens: list[torch.Tensor] = []
        presences: list[torch.Tensor] = []
        for group in self.source_order:
            encoded: Optional[tuple[torch.Tensor, torch.Tensor]]
            if group in disabled:
                encoded = None
            elif group == "clinical_data":
                encoded = self._clinical(batch)
            elif group == "cardiac_ecg":
                encoded = self._ecg(batch)
            elif group in RETINAL_GROUPS:
                encoded = self._retinal(batch, group)
            else:
                encoded = self._temporal(batch, group)

            if encoded is None:
                token = torch.zeros(
                    batch_size, self.config.d_model, device=device, dtype=dtype
                )
                present = torch.zeros(batch_size, device=device, dtype=torch.bool)
            else:
                token, present = encoded
                if token.shape != (batch_size, self.config.d_model):
                    raise ValueError(
                        f"{group} encoder returned {tuple(token.shape)}, expected "
                        f"{(batch_size, self.config.d_model)}"
                    )
                present = self._declared_presence(batch, group, present)
                token = token * present.to(dtype=token.dtype).unsqueeze(-1)
            tokens.append(token)
            presences.append(present)

        source_tokens = torch.stack(tokens, dim=1)
        source_present = torch.stack(presences, dim=1)
        return {
            "source_tokens": source_tokens,
            "source_present": source_present,
        }

    def fuse_encoded_sources(
        self,
        source_tokens: torch.Tensor,
        source_present: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply presence-aware fusion to already encoded native tokens.

        This is the canonical fusion boundary for both :meth:`forward` and
        frozen post-hoc importance.  Passing ``False`` for one source invokes
        the learned missing-token semantics in :class:`PresenceAwareFusion`;
        callers must not synthesize their own missing representation.
        """

        if source_tokens.ndim != 3:
            raise ValueError("source_tokens must have shape [B, S, D]")
        expected = (len(self.source_order), self.config.d_model)
        if tuple(source_tokens.shape[1:]) != expected:
            raise ValueError(
                "source_tokens trailing shape must be "
                f"{expected}, got {tuple(source_tokens.shape[1:])}"
            )
        if source_present.shape != source_tokens.shape[:2]:
            raise ValueError("source_present must have shape [B, S]")
        if not torch.isfinite(source_tokens).all():
            raise ValueError("source_tokens must be finite")
        embedding, fused_source_tokens = self.fusion(source_tokens, source_present)
        return {
            "embedding": embedding,
            "h_multi": embedding,
            "source_tokens": source_tokens,
            "fused_source_tokens": fused_source_tokens,
            "source_present": source_present,
        }

    def predict_from_source_tokens(
        self,
        source_tokens: torch.Tensor,
        source_present: torch.Tensor,
        *,
        task: str,
    ) -> Dict[str, Any]:
        """Run only fusion and a named task head on cached source tokens."""

        selected_task = self._resolve_task({}, task)
        encoded = self.fuse_encoded_sources(source_tokens, source_present)
        logits = self.heads[selected_task](encoded["embedding"])
        return {
            **encoded,
            "logits": logits,
            "y_logit": logits,
            "task": selected_task,
        }

    def encode(
        self,
        batch: Mapping[str, Any],
        *,
        disabled_groups: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode and fuse a batch, optionally removing complete sources."""

        sources = self.encode_sources(batch, disabled_groups=disabled_groups)
        return self.fuse_encoded_sources(
            sources["source_tokens"], sources["source_present"]
        )

    def _resolve_task(
        self, batch: Mapping[str, Any], task: Optional[str]
    ) -> str:
        if task is None:
            value = batch.get("task")
            if isinstance(value, str):
                task = value
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                unique = {str(item) for item in value}
                if len(unique) != 1:
                    raise ValueError("a batch cannot mix downstream task heads")
                task = next(iter(unique))
            elif len(self.heads) == 1:
                task = next(iter(self.heads))
            else:
                raise ValueError("task must be passed when multiple heads are registered")
        task = str(task)
        if task not in self.heads:
            raise KeyError(
                f"unknown task {task!r}; registered tasks are {list(self.heads)}"
            )
        return task

    def forward(
        self,
        batch: Mapping[str, Any],
        task: Optional[str] = None,
        *,
        disabled_groups: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        selected_task = self._resolve_task(batch, task)
        sources = self.encode_sources(batch, disabled_groups=disabled_groups)
        return self.predict_from_source_tokens(
            sources["source_tokens"],
            sources["source_present"],
            task=selected_task,
        )


# A descriptive alias for configuration files and external imports.
OT2LANISTRModelV2 = Task2V2Model


__all__ = [
    "BinaryTaskHead",
    "DEFAULT_BINARY_TASKS",
    "ECGEncoder",
    "HierarchicalRetinalEncoder",
    "ImageNet1KInputNormalization",
    "MaskedAttentionPool",
    "MaskedTemporalPatchEncoder",
    "MaskAwareVectorEncoder",
    "OT2LANISTRModelV2",
    "PRIMARY_MODALITY_GROUPS",
    "PresenceAwareFusion",
    "RETINAL_GROUPS",
    "ResNet18RetinalBackbone",
    "SharedRetinalCNN",
    "Task2V2Model",
    "Task2V2ModelConfig",
    "build_retinal_backbone",
]
