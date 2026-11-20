# Copyright (c) 2026 The Authors
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE in the bundle root and https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Mixture-of-Experts over pathology foundation-model embeddings.
# Vendored (with attribution) from https://github.com/eliem08/paper_code
# (`src/breastpath_copilot/models/moe.py`), the code accompanying the manuscript
# "Toward reliable early breast lesion detection: a state-of-the-art,
# uncertainty-aware diagnostic engine with agentic decision support in H&E
# histopathology" (under review). Only the bundle entry point at the bottom of
# this file (`BundleMoEClassifier`) is new: it adapts the MoE head to the
# single-tensor-in / logits-out interface MONAI bundles expect, while keeping
# the trained checkpoint's state-dict keys (`moe.*`, `classifier.*`) unchanged.
#
# This module exposes three routing strategies that share a common
# :class:`MoEHead` interface so they can be benchmarked head-to-head:
#
# * :class:`HardMagRoutingMoE` -- magnification-conditioned hard routing via a
#   fixed lookup table. Interpretable, no learning, easy to defend to reviewers.
# * :class:`SoftTopKRoutingMoE` -- learned soft routing with top-k=2 selection,
#   dropless (every input keeps exactly k experts), Switch-style auxiliary
#   load-balance loss. Design choices follow Li et al., "Slicing and Dicing"
#   (arXiv:2605.11689), which found top-1 routing under-performs and dropless
#   routing yields a consistent gain.
# * :class:`StackingMoE` -- always-on weighted ensemble baseline. Per the same
#   paper, always-active "generalists" alongside routed experts tend to hurt,
#   so we keep stacking as a *replacement* for routing rather than an add-on.
#
# Setting context: our experts are a small number (3) of large pretrained
# visual foundation models (UNI / Virchow2 / Phikon-v2), and routing operates
# at the *case* level (one routing decision per tile) rather than per-token
# inside a transformer layer. This means several Slicing-and-Dicing findings
# (expert-count scaling, granularity heterogeneity) do not transfer cleanly --
# they should be cited as a methods-section disclaimer rather than driving
# design.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

RoutingKind = Literal["hard_mag", "soft_topk", "stacking"]

# Canonical expert order and embedding dims for the shipped checkpoint
# (`models/model.pt`, trained by `scripts/train_agent_moe_ckpt.py` in
# paper_code). Virchow2 is 2560-d because its embeddings pool the CLS token
# AND the mean patch token (1280 + 1280); UNI and Phikon-v2 use the CLS token
# only (1024-d each).
EXPERT_ORDER: tuple[str, ...] = ("uni", "virchow2", "phikon_v2_ft")
EXPERT_DIMS: dict[str, int] = {"uni": 1024, "virchow2": 2560, "phikon_v2_ft": 1024}
# Class order of the shipped 3-class head (BRACS 3-class scheme).
CLASS_LABELS: tuple[str, ...] = ("benign", "atypical", "malignant")


@dataclass
class MoEOutput:
    """Return type for every MoE head.

    Attributes
    ----------
    routed_embedding
        ``(B, shared_dim)`` -- mixed expert representation, ready for a
        classifier head or for retrieval / agentic inspection.
    gate_weights
        ``(B, num_experts)`` -- per-expert weight after routing (zeros for
        experts not selected by top-k). Sums to 1 per row. Exposed for
        auditability and for the agent harness's expert-attribution evidence.
    aux_loss
        Scalar load-balance auxiliary loss. ``0`` for ``hard_mag`` and
        ``stacking`` (which have no routing parameters to balance).
    chosen_experts
        Indices of the selected experts per input. Shape ``(B, k)`` for
        ``soft_topk``, ``(B, 1)`` (argmax) for ``hard_mag`` and ``stacking``.
    """

    routed_embedding: torch.Tensor
    gate_weights: torch.Tensor
    aux_loss: torch.Tensor
    chosen_experts: torch.Tensor


# Default magnification routing table.
# Priors:
#   40X  -- global tissue architecture -> favor Virchow2 (ViT-H, larger context)
#   100X -- mixed tissue + cellular     -> Virchow2 / UNI balanced
#   200X -- cellular detail              -> UNI weighted slightly higher
#   400X -- nuclear detail               -> UNI / Phikon-v2 weighted higher
#   unknown / other -> uniform fallback
# Weights are normalised in HardMagRoutingMoE; raw values just need to be >= 0.
DEFAULT_MAG_ROUTING: dict[str, dict[str, float]] = {
    "40X": {"uni": 0.20, "virchow2": 0.60, "phikon_v2_ft": 0.20},
    "100X": {"uni": 0.35, "virchow2": 0.45, "phikon_v2_ft": 0.20},
    "200X": {"uni": 0.40, "virchow2": 0.30, "phikon_v2_ft": 0.30},
    "400X": {"uni": 0.50, "virchow2": 0.20, "phikon_v2_ft": 0.30},
}
UNKNOWN_MAGNIFICATION_KEY = "unknown"


@dataclass
class MoEConfig:
    """Configuration for :func:`build_moe`."""

    routing: RoutingKind
    shared_dim: int = 512
    layer_norm: bool = True
    top_k: int = 2
    gate_hidden_dim: int | None = None
    load_balance_weight: float = 0.01
    per_dim_weights: bool = False
    magnification_to_weights: dict[str, dict[str, float]] | None = None
    note: str = ""


class ExpertProjector(nn.Module):
    """Per-FM linear projection to a shared embedding dim, with optional LayerNorm."""

    def __init__(self, expert_dims: Mapping[str, int], shared_dim: int, layer_norm: bool = True) -> None:
        super().__init__()
        if not expert_dims:
            raise ValueError("expert_dims must contain at least one expert")
        self.expert_names: tuple[str, ...] = tuple(expert_dims.keys())
        self.shared_dim = int(shared_dim)
        self.projectors = nn.ModuleDict(
            {name: nn.Linear(int(dim), self.shared_dim) for name, dim in expert_dims.items()}
        )
        self.norm = nn.LayerNorm(self.shared_dim) if layer_norm else nn.Identity()

    def forward(self, expert_embeddings: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Returns a stacked tensor of shape ``(B, num_experts, shared_dim)``."""
        missing = [n for n in self.expert_names if n not in expert_embeddings]
        if missing:
            raise KeyError(f"expert_embeddings missing keys: {missing}")
        projected = [self.norm(self.projectors[name](expert_embeddings[name])) for name in self.expert_names]
        return torch.stack(projected, dim=1)


class MoEHead(nn.Module):
    """Common base class for all routing variants."""

    expert_names: tuple[str, ...]
    num_experts: int
    shared_dim: int
    projector: ExpertProjector

    def forward(
        self, expert_embeddings: Mapping[str, torch.Tensor], magnifications: list[str] | None = None
    ) -> MoEOutput:  # pragma: no cover - subclassed
        raise NotImplementedError


class HardMagRoutingMoE(MoEHead):
    """Magnification-conditioned hard routing.

    The routing weights are *not* learned; they come from a lookup table.
    For magnifications not in the table, we fall back to uniform weights so
    that BACH / BRACS (which lack explicit magnification metadata) still
    receive a valid forward pass.
    """

    def __init__(
        self,
        expert_dims: Mapping[str, int],
        shared_dim: int = 512,
        magnification_to_weights: Mapping[str, Mapping[str, float]] | None = None,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.expert_names = tuple(expert_dims.keys())
        self.num_experts = len(self.expert_names)
        self.shared_dim = int(shared_dim)
        self.projector = ExpertProjector(expert_dims, shared_dim, layer_norm=layer_norm)
        self.magnification_to_weights = dict(magnification_to_weights or DEFAULT_MAG_ROUTING)

    def _row_for_magnification(self, mag: str) -> torch.Tensor:
        if mag in self.magnification_to_weights:
            row = [float(self.magnification_to_weights[mag].get(name, 0.0)) for name in self.expert_names]
        else:
            row = [1.0 / self.num_experts] * self.num_experts
        tensor = torch.tensor(row, dtype=torch.float32)
        total = tensor.sum()
        if total <= 0:
            return torch.full((self.num_experts,), 1.0 / self.num_experts)
        return tensor / total

    def forward(
        self, expert_embeddings: Mapping[str, torch.Tensor], magnifications: list[str] | None = None
    ) -> MoEOutput:
        projected = self.projector(expert_embeddings)  # (B, E, D)
        batch_size = projected.shape[0]
        if magnifications is None:
            magnifications = [UNKNOWN_MAGNIFICATION_KEY] * batch_size
        if len(magnifications) != batch_size:
            raise ValueError(f"magnifications has {len(magnifications)} entries, expected {batch_size}")
        rows = torch.stack([self._row_for_magnification(m) for m in magnifications], dim=0)
        rows = rows.to(projected.device, dtype=projected.dtype)
        routed = (rows.unsqueeze(-1) * projected).sum(dim=1)
        return MoEOutput(
            routed_embedding=routed,
            gate_weights=rows,
            aux_loss=torch.zeros((), device=projected.device, dtype=projected.dtype),
            chosen_experts=rows.argmax(dim=-1, keepdim=True),
        )


class SoftTopKRoutingMoE(MoEHead):
    """Learned soft routing with top-k selection and a load-balance aux loss.

    Default ``top_k=2`` per Li et al. (Slicing and Dicing): top-1 routing
    consistently under-performs because gradients update only one router row
    per input. ``top_k=num_experts`` collapses to a full soft ensemble, which
    is useful as a sanity-check ablation.

    The auxiliary load-balance loss follows the Switch Transformer form,
    adapted for top-k routing::

        L_aux = N_E * sum_i f_i * P_i

    with ``f_i`` = fraction of inputs that include expert ``i`` in their top-k
    and ``P_i`` = mean softmax probability for expert ``i`` over the batch.
    """

    def __init__(
        self,
        expert_dims: Mapping[str, int],
        shared_dim: int = 512,
        top_k: int = 2,
        gate_hidden_dim: int | None = None,
        load_balance_weight: float = 0.01,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.expert_names = tuple(expert_dims.keys())
        self.num_experts = len(self.expert_names)
        if not (1 <= top_k <= self.num_experts):
            raise ValueError(f"top_k must be in [1, {self.num_experts}], got {top_k}")
        self.top_k = int(top_k)
        self.shared_dim = int(shared_dim)
        self.load_balance_weight = float(load_balance_weight)
        self.projector = ExpertProjector(expert_dims, shared_dim, layer_norm=layer_norm)

        gate_in = self.shared_dim * self.num_experts
        gate_h = int(gate_hidden_dim) if gate_hidden_dim else max(self.shared_dim // 2, 64)
        self.gate = nn.Sequential(nn.Linear(gate_in, gate_h), nn.GELU(), nn.Linear(gate_h, self.num_experts))

    def forward(
        self, expert_embeddings: Mapping[str, torch.Tensor], magnifications: list[str] | None = None
    ) -> MoEOutput:
        projected = self.projector(expert_embeddings)  # (B, E, D)
        B, E, D = projected.shape
        gate_logits = self.gate(projected.reshape(B, E * D))
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, E)

        topk_vals, topk_idx = gate_probs.topk(self.top_k, dim=-1)
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        gate_mask = torch.zeros_like(gate_probs).scatter(1, topk_idx, topk_weights)

        routed = (gate_mask.unsqueeze(-1) * projected).sum(dim=1)

        # Switch-style load-balance loss
        membership = torch.zeros_like(gate_probs).scatter(1, topk_idx, 1.0)
        f = membership.mean(dim=0)
        P = gate_probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum() * self.load_balance_weight

        return MoEOutput(routed_embedding=routed, gate_weights=gate_mask, aux_loss=aux_loss, chosen_experts=topk_idx)


class StackingMoE(MoEHead):
    """Always-on weighted-ensemble baseline.

    Reviewers will (correctly) note this is *not* a true MoE because all
    experts are always active. We include it as a deliberate baseline so the
    Q1 ablation table can show what the routing strategies buy *over* a
    learned weighted ensemble. Weights are softmax-normalised over experts so
    they sum to 1 (per-expert or per-dim per :attr:`per_dim_weights`).
    """

    def __init__(
        self,
        expert_dims: Mapping[str, int],
        shared_dim: int = 512,
        per_dim_weights: bool = False,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.expert_names = tuple(expert_dims.keys())
        self.num_experts = len(self.expert_names)
        self.shared_dim = int(shared_dim)
        self.per_dim_weights = bool(per_dim_weights)
        self.projector = ExpertProjector(expert_dims, shared_dim, layer_norm=layer_norm)
        if self.per_dim_weights:
            self.weight_logits = nn.Parameter(torch.zeros(self.num_experts, self.shared_dim))
        else:
            self.weight_logits = nn.Parameter(torch.zeros(self.num_experts))

    def forward(
        self, expert_embeddings: Mapping[str, torch.Tensor], magnifications: list[str] | None = None
    ) -> MoEOutput:
        projected = self.projector(expert_embeddings)  # (B, E, D)
        B = projected.shape[0]
        if self.per_dim_weights:
            w = F.softmax(self.weight_logits, dim=0)  # (E, D)
            routed = (w.unsqueeze(0) * projected).sum(dim=1)  # (B, D)
            gate = w.mean(dim=-1).unsqueeze(0).expand(B, -1)
        else:
            w = F.softmax(self.weight_logits, dim=0)  # (E,)
            routed = (w.view(1, self.num_experts, 1) * projected).sum(dim=1)
            gate = w.unsqueeze(0).expand(B, -1)
        return MoEOutput(
            routed_embedding=routed,
            gate_weights=gate,
            aux_loss=torch.zeros((), device=projected.device, dtype=projected.dtype),
            chosen_experts=gate.argmax(dim=-1, keepdim=True),
        )


def build_moe(expert_dims: Mapping[str, int], config: MoEConfig) -> MoEHead:
    """Factory: build an MoE head from an :class:`MoEConfig`.

    Parameters
    ----------
    expert_dims
        Mapping ``{fm_name: embedding_dim}``, e.g.
        ``{"uni": 1024, "virchow2": 2560, "phikon_v2_ft": 1024}``. The key order
        defines the canonical expert index.
    config
        Routing strategy + hyperparameters.
    """
    if config.routing == "hard_mag":
        return HardMagRoutingMoE(
            expert_dims=expert_dims,
            shared_dim=config.shared_dim,
            magnification_to_weights=config.magnification_to_weights,
            layer_norm=config.layer_norm,
        )
    if config.routing == "soft_topk":
        return SoftTopKRoutingMoE(
            expert_dims=expert_dims,
            shared_dim=config.shared_dim,
            top_k=config.top_k,
            gate_hidden_dim=config.gate_hidden_dim,
            load_balance_weight=config.load_balance_weight,
            layer_norm=config.layer_norm,
        )
    if config.routing == "stacking":
        return StackingMoE(
            expert_dims=expert_dims,
            shared_dim=config.shared_dim,
            per_dim_weights=config.per_dim_weights,
            layer_norm=config.layer_norm,
        )
    raise ValueError(f"Unknown routing strategy: {config.routing!r}")


class MoEClassifier(nn.Module):
    """Convenience wrapper: MoE head + linear classifier.

    Useful for the ablation study runner; downstream code that needs only the
    routed embedding (e.g., retrieval, gate-aware agentic tools) should call
    the underlying MoE head directly.
    """

    def __init__(
        self, expert_dims: Mapping[str, int], num_classes: int, config: MoEConfig, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.moe = build_moe(expert_dims, config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.moe.shared_dim, int(num_classes))

    def forward(
        self, expert_embeddings: Mapping[str, torch.Tensor], magnifications: list[str] | None = None
    ) -> tuple[torch.Tensor, MoEOutput]:
        out = self.moe(expert_embeddings, magnifications=magnifications)
        logits = self.classifier(self.dropout(out.routed_embedding))
        return logits, out


class BundleMoEClassifier(MoEClassifier):
    """MONAI-bundle entry point for the BRACS MoE head.

    Same weights as :class:`MoEClassifier` (the shipped ``models/model.pt``
    loads with identical state-dict keys), but with a MONAI-friendly forward:
    one input, one logits tensor out.

    Accepted inputs
    ---------------
    - A mapping ``{"uni": (B, 1024), "virchow2": (B, 2560),
      "phikon_v2_ft": (B, 1024)}`` -- the form the bundle dataset produces.
    - A single concatenated tensor ``(B, 4608)`` in :data:`EXPERT_ORDER`,
      i.e. ``[uni | virchow2 | phikon_v2_ft]``.

    Returns logits of shape ``(B, num_classes)`` over
    :data:`CLASS_LABELS` (benign, atypical, malignant). Softmax is applied by
    the bundle's postprocessing, not here.
    """

    def __init__(
        self,
        expert_dims: Mapping[str, int] | None = None,
        num_classes: int = 3,
        routing: RoutingKind = "soft_topk",
        shared_dim: int = 512,
        top_k: int = 2,
        gate_hidden_dim: int | None = None,
        load_balance_weight: float = 0.01,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        dims = dict(EXPERT_DIMS if expert_dims is None else expert_dims)
        config = MoEConfig(
            routing=routing,
            shared_dim=shared_dim,
            top_k=top_k,
            gate_hidden_dim=gate_hidden_dim,
            load_balance_weight=load_balance_weight,
            layer_norm=layer_norm,
        )
        super().__init__(expert_dims=dims, num_classes=num_classes, config=config, dropout=dropout)
        self.expert_dims = dims

    def forward(
        self, image: Mapping[str, torch.Tensor] | torch.Tensor, magnifications: list[str] | None = None
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.dim() != 2 or image.shape[1] != sum(self.expert_dims.values()):
                raise ValueError(
                    f"concatenated embedding must have shape (B, {sum(self.expert_dims.values())}), "
                    f"got {tuple(image.shape)}"
                )
            expert_embeddings: dict[str, torch.Tensor] = {}
            start = 0
            for name in EXPERT_ORDER:
                dim = self.expert_dims[name]
                expert_embeddings[name] = image[:, start : start + dim]
                start += dim
        else:
            expert_embeddings = dict(image)
        logits, _ = super().forward(expert_embeddings, magnifications=magnifications)
        return logits
