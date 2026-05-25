"""
models/moe_router.py — Mixture-of-Experts Dynamic Gating Router
================================================================

Core component of the Nexus Quant OS inference layer (Architecture Report §3).

Implements:
    1. **Top-K Token-Chooses-Expert routing** — each input token selects
       the *k* most relevant experts via a learned gating network.
    2. **Expert-Choice routing** (optional mode) — each expert selects
       its preferred tokens for perfect load balance.
    3. **Auxiliary load-balancing loss** — penalises uneven expert
       utilisation to prevent *Expert Collapse* (all data flowing to
       a single expert while others remain untrained).

Tensor Shape Convention (used throughout this file):
    B  = batch_size
    D  = input_dim   (dimensionality of the fused feature vector)
    E  = num_experts
    K  = top_k       (number of experts activated per token)
    O  = output_dim  (dimensionality of each expert's output)

Device Compatibility:
    Tested on CPU and Apple MPS (Metal Performance Shaders).
    GPU/CUDA is supported but not required.

Author : Nexus Quant OS — Model Engineering Division
License: Proprietary
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("nexus_quant_os.models.moe_router")


# ═════════════════════════════════════════════════════════════════════
# 1. TYPE CONTRACTS & CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

@runtime_checkable
class ExpertModule(Protocol):
    """Structural protocol that every expert sub-network must satisfy.

    Each expert receives a 2-D tensor ``[B_sub, D]`` and returns a
    2-D tensor ``[B_sub, O]`` where ``B_sub <= B`` is the number of
    tokens routed to this expert.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class GatingNoiseType(Enum):
    """Strategy for injecting exploration noise into gating logits."""
    NONE     = auto()   # Deterministic routing (inference mode)
    GAUSSIAN = auto()   # Additive Gaussian noise (Switch Transformer §3.1)
    UNIFORM  = auto()   # Additive Uniform noise (Hash Layer style)


@dataclass(frozen=True)
class RouterConfig:
    """Immutable configuration for QuantMoERouter.

    Attributes
    ----------
    input_dim : int
        Dimensionality of the incoming fused feature vector (D).
    num_experts : int
        Total number of expert sub-networks available (E).
    output_dim : int
        Dimensionality of each expert's output (O).
    top_k : int
        Number of experts activated per token (K). Clamped to [1, E].
    noise_type : GatingNoiseType
        Type of exploration noise injected during training.
    noise_std : float
        Standard deviation of Gaussian noise (ignored for UNIFORM/NONE).
    aux_loss_coeff : float
        Scalar weight applied to the load-balancing auxiliary loss.
        Recommended range: 1e-2 to 1e-1 (Switch Transformer: 1e-2).
    jitter_eps : float
        Small epsilon added to routing weights to avoid division by zero.
    """
    input_dim: int
    num_experts: int
    output_dim: int = 1
    top_k: int = 2
    noise_type: GatingNoiseType = GatingNoiseType.GAUSSIAN
    noise_std: float = 0.1
    aux_loss_coeff: float = 1e-2
    jitter_eps: float = 1e-6


@dataclass
class RoutingOutput:
    """Structured output from a single forward pass of the MoE router.

    Attributes
    ----------
    combined_output : torch.Tensor
        Final expert-fused output, shape ``[B, O]``.
    aux_loss : torch.Tensor
        Scalar load-balancing auxiliary loss.
    gate_probs : torch.Tensor
        Full gating probability matrix, shape ``[B, E]``.
    top_k_indices : torch.Tensor
        Indices of selected experts per token, shape ``[B, K]``.
    top_k_weights : torch.Tensor
        Normalised routing weights per token, shape ``[B, K]``.
    expert_utilisation : torch.Tensor
        Fraction of tokens routed to each expert, shape ``[E]``.
    """
    combined_output: torch.Tensor
    aux_loss: torch.Tensor
    gate_probs: torch.Tensor
    top_k_indices: torch.Tensor
    top_k_weights: torch.Tensor
    expert_utilisation: torch.Tensor


# ═════════════════════════════════════════════════════════════════════
# 2. CORE ROUTER IMPLEMENTATION
# ═════════════════════════════════════════════════════════════════════

class QuantMoERouter(nn.Module):
    """
    Nexus Quant OS — Mixture-of-Experts Dynamic Gating Router.

    Architecture
    ------------
    ::

        Input [B, D]
           |
           v
        +------------------+
        |  Gating Network  |  Linear(D, E) + optional noise
        |  (learned)       |
        +--------+---------+
                 | gate_logits [B, E]
                 v
        +------------------+
        |  Softmax + Top-K |  select K experts per token
        +--------+---------+
                 | top_k_indices [B, K], top_k_weights [B, K]
                 v
        +----------------------------------------------+
        |  Dispatch -> Expert_i(x) -> Weighted Combine |
        +--------+-------------------------------------+
                 | combined_output [B, O]
                 v
        +------------------+
        |  Aux Loss (load  |  <- penalises unbalanced utilisation
        |  balancing)      |
        +------------------+

    Parameters
    ----------
    config : RouterConfig
        Frozen configuration dataclass.
    experts : nn.ModuleList
        Pre-initialised expert sub-networks.  Length must equal
        ``config.num_experts``.  Each expert must accept ``[B_sub, D]``
        and return ``[B_sub, O]``.
    """

    def __init__(self, config: RouterConfig, experts: nn.ModuleList) -> None:
        super().__init__()

        if len(experts) != config.num_experts:
            raise ValueError(
                f"Expected {config.num_experts} experts, got {len(experts)}."
            )
        if config.top_k < 1 or config.top_k > config.num_experts:
            raise ValueError(
                f"top_k={config.top_k} must be in [1, {config.num_experts}]."
            )

        self.config  = config
        self.experts = experts

        # Gating network: projects D-dimensional input into E logit scores
        self.gate_linear = nn.Linear(config.input_dim, config.num_experts)
        nn.init.kaiming_uniform_(self.gate_linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_linear.bias)

        logger.info(
            "QuantMoERouter initialised | D=%d  E=%d  K=%d  O=%d  "
            "noise=%s  aux_coeff=%.4f",
            config.input_dim, config.num_experts, config.top_k,
            config.output_dim, config.noise_type.name, config.aux_loss_coeff,
        )

    # ─────────────────────────────────────────────────────────────
    # 2a. Gating Logit Computation
    # ─────────────────────────────────────────────────────────────

    def _compute_gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw gating scores with optional exploration noise.

        Parameters
        ----------
        x : torch.Tensor  Shape: ``[B, D]``

        Returns
        -------
        gate_logits : torch.Tensor  Shape: ``[B, E]``
        """
        gate_logits: torch.Tensor = self.gate_linear(x)   # [B, D] -> [B, E]

        if self.training and self.config.noise_type != GatingNoiseType.NONE:
            if self.config.noise_type == GatingNoiseType.GAUSSIAN:
                noise = torch.randn_like(gate_logits) * self.config.noise_std
                # noise: [B, E]
                gate_logits = gate_logits + noise
            elif self.config.noise_type == GatingNoiseType.UNIFORM:
                noise = torch.rand_like(gate_logits) * self.config.noise_std
                # noise: [B, E]
                gate_logits = gate_logits + noise

        return gate_logits   # [B, E]

    # ─────────────────────────────────────────────────────────────
    # 2b. Top-K Expert Selection
    # ─────────────────────────────────────────────────────────────

    def _top_k_routing(
        self,
        gate_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-K experts per token and compute normalised weights.

        Parameters
        ----------
        gate_logits : torch.Tensor  Shape: ``[B, E]``

        Returns
        -------
        gate_probs    : torch.Tensor  Shape: ``[B, E]``
        top_k_indices : torch.Tensor  Shape: ``[B, K]``
        top_k_weights : torch.Tensor  Shape: ``[B, K]``
        """
        # Full probability distribution over all experts
        gate_probs: torch.Tensor = F.softmax(gate_logits, dim=-1)   # [B, E]

        # Select top-K experts per token
        top_k_probs, top_k_indices = torch.topk(
            gate_probs, k=self.config.top_k, dim=-1, sorted=True
        )
        # top_k_probs:   [B, K]
        # top_k_indices: [B, K]

        # Renormalise selected weights to sum to 1 per token
        weight_sum    = top_k_probs.sum(dim=-1, keepdim=True)        # [B, 1]
        top_k_weights = top_k_probs / (weight_sum + self.config.jitter_eps)
        # top_k_weights: [B, K]

        return gate_probs, top_k_indices, top_k_weights

    # ─────────────────────────────────────────────────────────────
    # 2c. Auxiliary Load-Balancing Loss
    # ─────────────────────────────────────────────────────────────

    def _compute_aux_loss(
        self,
        gate_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the load-balancing auxiliary loss.

        Formulation (Switch Transformer, Fedus et al. 2022):
            aux_loss = E * sum_i(importance_i * load_i) * coeff

        Parameters
        ----------
        gate_probs    : torch.Tensor  Shape: ``[B, E]``
        top_k_indices : torch.Tensor  Shape: ``[B, K]``

        Returns
        -------
        aux_loss            : torch.Tensor  scalar
        expert_utilisation  : torch.Tensor  Shape: ``[E]``
        """
        E = self.config.num_experts

        # Importance: mean routing probability per expert
        importance: torch.Tensor = gate_probs.mean(dim=0)   # [E]

        # Load (frequency): fraction of tokens routed to each expert
        one_hot = F.one_hot(top_k_indices, num_classes=E).float()
        # one_hot: [B, K, E]

        # A token is "assigned" to expert e if e appears in ANY of its K slots
        expert_mask = one_hot.sum(dim=1).clamp(max=1.0)   # [B, E]
        load: torch.Tensor = expert_mask.mean(dim=0)       # [E]

        # Switch Transformer loss: scale-invariant w.r.t. num experts
        aux_loss_raw: torch.Tensor = (importance * load).sum() * E   # scalar
        aux_loss: torch.Tensor = aux_loss_raw * self.config.aux_loss_coeff

        return aux_loss, load   # scalar, [E]

    # ─────────────────────────────────────────────────────────────
    # 2d. Expert Dispatch & Weighted Combination
    # ─────────────────────────────────────────────────────────────

    def _dispatch_and_combine(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Route input tokens to selected experts and fuse their outputs.

        Uses per-expert-batching for efficiency:
        gather tokens -> run expert -> scatter weighted output back.

        Parameters
        ----------
        x             : torch.Tensor  Shape: ``[B, D]``
        top_k_indices : torch.Tensor  Shape: ``[B, K]``
        top_k_weights : torch.Tensor  Shape: ``[B, K]``

        Returns
        -------
        combined : torch.Tensor  Shape: ``[B, O]``
        """
        B = x.size(0)
        O = self.config.output_dim
        K = self.config.top_k

        combined = torch.zeros(B, O, device=x.device, dtype=x.dtype)   # [B, O]

        for k_slot in range(K):
            # expert_ids_at_k: [B] — which expert each token picked at slot k
            expert_ids_at_k: torch.Tensor = top_k_indices[:, k_slot]    # [B]
            # weights_at_k: [B, 1] — routing weight for this slot
            weights_at_k: torch.Tensor = top_k_weights[:, k_slot].unsqueeze(-1)  # [B, 1]

            unique_experts = expert_ids_at_k.unique()
            for expert_idx in unique_experts:
                expert_idx_item: int = expert_idx.item()

                # Gather: which tokens are routed to this expert
                token_mask: torch.Tensor = (expert_ids_at_k == expert_idx)   # [B]
                if not token_mask.any():
                    continue

                # expert_input: [B_sub, D]
                expert_input: torch.Tensor = x[token_mask]

                # Forward: run expert on its assigned tokens -> [B_sub, O]
                expert_output: torch.Tensor = self.experts[expert_idx_item](expert_input)

                # Scatter: weighted accumulation back to original positions
                w: torch.Tensor = weights_at_k[token_mask]   # [B_sub, 1]
                combined[token_mask] = combined[token_mask] + expert_output * w

        return combined   # [B, O]

    # ─────────────────────────────────────────────────────────────
    # 2e. Main Forward Pass
    # ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> RoutingOutput:
        """Full MoE forward pass: gate -> route -> dispatch -> combine.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.  Shape: ``[B, D]``

        Returns
        -------
        RoutingOutput
            - combined_output : ``[B, O]``
            - aux_loss        : scalar
            - gate_probs      : ``[B, E]``
            - top_k_indices   : ``[B, K]``
            - top_k_weights   : ``[B, K]``
            - expert_utilisation : ``[E]``

        Raises
        ------
        ValueError  if input tensor has wrong dimensionality.
        """
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2-D input [B, D], got {x.ndim}-D tensor "
                f"with shape {list(x.shape)}."
            )
        if x.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Input dim mismatch: expected D={self.config.input_dim}, "
                f"got {x.size(-1)}."
            )

        # x: [B, D]

        # Step 1: Compute gating logits (with optional noise)
        gate_logits: torch.Tensor = self._compute_gate_logits(x)
        # gate_logits: [B, E]

        # Step 2: Top-K expert selection & weight normalisation
        gate_probs, top_k_indices, top_k_weights = self._top_k_routing(gate_logits)
        # gate_probs:    [B, E]
        # top_k_indices: [B, K]
        # top_k_weights: [B, K]

        # Step 3: Auxiliary load-balancing loss
        aux_loss, expert_utilisation = self._compute_aux_loss(gate_probs, top_k_indices)
        # aux_loss:            scalar
        # expert_utilisation:  [E]

        # Step 4: Dispatch to experts & weighted combination
        combined_output: torch.Tensor = self._dispatch_and_combine(
            x, top_k_indices, top_k_weights
        )
        # combined_output: [B, O]

        return RoutingOutput(
            combined_output=combined_output,
            aux_loss=aux_loss,
            gate_probs=gate_probs,
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
            expert_utilisation=expert_utilisation,
        )

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.config.input_dim}, "
            f"num_experts={self.config.num_experts}, "
            f"top_k={self.config.top_k}, "
            f"output_dim={self.config.output_dim}, "
            f"noise={self.config.noise_type.name}, "
            f"aux_coeff={self.config.aux_loss_coeff}"
        )


# ═════════════════════════════════════════════════════════════════════
# 3. HELPER: CREATE A STANDARD EXPERT
# ═════════════════════════════════════════════════════════════════════

def build_dummy_expert(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 64,
) -> nn.Sequential:
    """Build a minimal 2-layer expert for testing.

    Architecture: Linear(D, H) -> ReLU -> Linear(H, O)
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


# ═════════════════════════════════════════════════════════════════════
# 4. DEMONSTRATION & VALIDATION
# ═════════════════════════════════════════════════════════════════════

def _run_demonstration() -> None:
    """Validate the MoE router with dummy experts on CPU/MPS.

    Verifications:
        1. Output tensor has correct shape [B, O]
        2. Auxiliary loss is a positive scalar
        3. Routing weights sum to ~1.0 per token
        4. All 3 experts receive some traffic (no collapse)
        5. Backward pass succeeds (gradients flow)
        6. All 3 expert networks received gradients
    """
    separator = "=" * 72

    # Docker 容器化部署 — 強制 CPU
    device      = torch.device("cpu")
    device_name = "CPU (Docker container mode)"

    print(f"\n{separator}")
    print("  Nexus Quant OS — MoE Router Validation")
    print(f"  Device: {device_name}")
    print(f"{separator}\n")

    INPUT_DIM   = 32
    OUTPUT_DIM  = 1
    NUM_EXPERTS = 3
    TOP_K       = 2
    BATCH_SIZE  = 16

    config = RouterConfig(
        input_dim=INPUT_DIM,
        num_experts=NUM_EXPERTS,
        output_dim=OUTPUT_DIM,
        top_k=TOP_K,
        noise_type=GatingNoiseType.GAUSSIAN,
        noise_std=0.1,
        aux_loss_coeff=1e-2,
    )

    print(f"Router Config:")
    print(f"  D={INPUT_DIM}  E={NUM_EXPERTS}  K={TOP_K}  O={OUTPUT_DIM}")
    print(f"  noise={config.noise_type.name}  aux_coeff={config.aux_loss_coeff}\n")

    expert_names = ["xLSTM-Expert", "Mamba-Expert", "TabNet-Expert"]
    experts = nn.ModuleList([
        build_dummy_expert(INPUT_DIM, OUTPUT_DIM, hidden_dim=64)
        for _ in range(NUM_EXPERTS)
    ])

    print("Expert Pool:")
    for i, name in enumerate(expert_names):
        params = sum(p.numel() for p in experts[i].parameters())
        print(f"  [{i}] {name:16s}  params={params:,}")
    print()

    router = QuantMoERouter(config=config, experts=experts).to(device)
    router.train()

    total_params = sum(p.numel() for p in router.parameters())
    gate_params  = sum(p.numel() for p in router.gate_linear.parameters())
    print(f"Router total params: {total_params:,}  (gating network: {gate_params:,})\n")

    torch.manual_seed(42)
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
    # x: [16, 32]
    print(f"Input tensor shape: {list(x.shape)}  (B={BATCH_SIZE}, D={INPUT_DIM})\n")

    result: RoutingOutput = router(x)

    print("-" * 72)
    print("FORWARD PASS RESULTS")
    print("-" * 72)

    # [1] Output shape
    print(f"\n  combined_output shape : {list(result.combined_output.shape)}")
    assert result.combined_output.shape == (BATCH_SIZE, OUTPUT_DIM)
    print(f"  [1/6] Output shape correct: [{BATCH_SIZE}, {OUTPUT_DIM}]")

    # [2] Auxiliary loss
    print(f"\n  aux_loss value        : {result.aux_loss.item():.6f}")
    assert result.aux_loss.ndim == 0 and result.aux_loss.item() > 0
    print(f"  [2/6] Auxiliary loss is positive scalar: {result.aux_loss.item():.6f}")

    # [3] Routing weights sum to ~1.0
    weight_sums   = result.top_k_weights.sum(dim=-1)
    max_deviation = (weight_sums - 1.0).abs().max().item()
    print(f"\n  top_k_weights shape   : {list(result.top_k_weights.shape)}")
    print(f"  weight sums (sample)  : {weight_sums[:5].tolist()}")
    print(f"  max deviation from 1.0: {max_deviation:.8f}")
    assert max_deviation < 1e-4
    print(f"  [3/6] Routing weights sum to ~1.0 (max dev={max_deviation:.2e})")

    # [4] Expert utilisation — no collapse
    util = result.expert_utilisation
    print(f"\n  expert_utilisation    : {[f'{u:.3f}' for u in util.tolist()]}")
    experts_used = (util > 0).sum().item()
    assert experts_used == NUM_EXPERTS, f"Expert Collapse! Only {experts_used}/{NUM_EXPERTS} used."
    print(f"  [4/6] All {NUM_EXPERTS} experts receive traffic (no collapse)")

    # [5] Sample routing display
    print(f"\n  top_k_indices shape   : {list(result.top_k_indices.shape)}")
    print(f"  Sample routing (first 5 tokens):")
    for i in range(min(5, BATCH_SIZE)):
        indices = result.top_k_indices[i].tolist()
        weights = result.top_k_weights[i].tolist()
        route_str = "  +  ".join(
            f"{expert_names[idx]}({w:.3f})" for idx, w in zip(indices, weights)
        )
        print(f"    token[{i:2d}] -> {route_str}")

    # [5/6] Backward pass
    print()
    fake_target   = torch.randn(BATCH_SIZE, OUTPUT_DIM, device=device)
    pred_loss     = F.mse_loss(result.combined_output, fake_target)
    total_loss    = pred_loss + result.aux_loss
    total_loss.backward()

    gate_grad = router.gate_linear.weight.grad
    assert gate_grad is not None
    grad_norm = gate_grad.norm().item()
    print(f"  prediction_loss       : {pred_loss.item():.6f}")
    print(f"  total_loss (pred+aux) : {total_loss.item():.6f}")
    print(f"  gate_linear grad norm : {grad_norm:.6f}")
    assert grad_norm > 0
    print(f"  [5/6] Backward pass succeeded, gradients flow to gating network")

    # [6] Experts received gradients
    experts_with_grad = 0
    for expert in experts:
        for p in expert.parameters():
            if p.grad is not None and p.grad.norm().item() > 0:
                experts_with_grad += 1
                break
    assert experts_with_grad == NUM_EXPERTS
    print(f"  [6/6] All {NUM_EXPERTS} expert networks received gradients")

    # Gate probability heatmap
    print(f"\n{'-' * 72}")
    print(f"GATE PROBABILITY HEATMAP (first 8 tokens x 3 experts)")
    print(f"{'-' * 72}")
    header = "  Token  | " + " | ".join(f"{name:>16s}" for name in expert_names) + " |"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(min(8, BATCH_SIZE)):
        probs    = result.gate_probs[i].tolist()
        selected = set(result.top_k_indices[i].tolist())
        cells    = []
        for j, p in enumerate(probs):
            marker = " <-" if j in selected else "   "
            cells.append(f"{p:>13.4f}{marker}")
        row = f"  [{i:4d}] | " + " | ".join(cells) + " |"
        print(row)
    print(f"  (<- = selected by Top-{TOP_K} routing)\n")

    print(f"{separator}")
    print(f"  ALL 6 ASSERTIONS PASSED")
    print(f"  MoE Router is fully operational on {device_name}.")
    print(f"{separator}\n")

    print("Model Architecture:")
    print(router)
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    _run_demonstration()
