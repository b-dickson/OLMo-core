"""
Identity Hyper Connections for multi-stream residual connections.

This implements the "Identity HC" variant where H_res = I (identity) and dynamic
(input-dependent) h_pre/h_post weights are generated per-token via a learned projection φ.

Standard residual: ``x_{l+1} = x_l + F(x_l)``

Identity HC with *n* streams (state ``H`` in ``R^{n x d}``):

For each sublayer ``F``:
  1. Compute dynamic weights from flattened state: ``(h_pre, h_post) = σ(α · φ(H) + bias)``
  2. ``h_input = Σ h_pre_i · H_i``   — weighted merge of streams
  3. ``h_output = F(h_input)``        — run the sublayer
  4. ``H_new = H + diag(h_post) · h_output``  — identity residual + distribute

Based on Identity HC findings showing Identity HC > mHC > mHC lite > mHC orthogonal,
and the key insight that H_res = I avoids cumulative rank-1 collapse of doubly-stochastic
matrices while the φ projection still provides cross-stream information mixing.

See :class:`IdentityHyperConnectionConfig` for the config and :class:`HyperConnectionStream` for
the drop-in replacement of :class:`~olmo_core.nn.residual_stream.ResidualStream`.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from olmo_core.config import Config


@dataclass
class IdentityHyperConnectionConfig(Config):
    """
    Configuration for Identity Hyper Connections.

    :param n_streams: Number of parallel residual streams (expansion rate).
    """

    n_streams: int = 4


class HyperConnectionStream(nn.Module):
    """
    Drop-in replacement for :class:`~olmo_core.nn.residual_stream.ResidualStream`
    that implements Identity Hyper Connections with dynamic (input-dependent) weights.

    Each instance uses a learned projection ``φ`` (without bias) to map the flattened
    multi-stream state to ``2n`` values, then applies sigmoid with learnable temperature
    scalars to produce per-token merge and distribute weights:

    - ``h_pre = sigmoid(α_pre · φ(H)[:n] + bias[:n])``  — merge weights ∈ [0, 1]
    - ``h_post = 2 · sigmoid(α_post · φ(H)[n:] + bias[n:])``  — distribute weights ∈ [0, 2]

    The bias is **not** scaled by α, so at init (α ≈ 0) the weights are purely
    determined by the bias, giving a round-robin one-hot pattern for ``h_pre``
    and uniform ``h_post ≈ 1``. As α grows during training, weights become
    input-dependent via the φ projection (which provides cross-stream mixing).

    :param n_streams: Number of parallel residual streams.
    :param d_model: Model dimensionality.
    :param sublayer_idx: Global sublayer index for round-robin bias initialization.
    """

    def __init__(self, n_streams: int, d_model: int, sublayer_idx: int = 0):
        super().__init__()
        self.n_streams = n_streams
        self.d_model = d_model
        self.sublayer_idx = sublayer_idx

        # φ projection: flattened multi-stream state → 2n values (no bias here).
        self.phi = nn.Linear(n_streams * d_model, 2 * n_streams, bias=False)

        # Separate bias (not scaled by alpha) so init is clean.
        self.bias = nn.Parameter(torch.empty(2 * n_streams))

        # Temperature scalars controlling how input-dependent the weights are.
        # Start small so weights are ~constant at init, gradually become dynamic.
        # Use 1D tensors (not scalars) for FSDP compatibility.
        self.alpha_pre = nn.Parameter(torch.empty(1))
        self.alpha_post = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        # Zero phi weights so projection starts as zero (output = bias only at init).
        nn.init.zeros_(self.phi.weight)

        stream = self.sublayer_idx % self.n_streams
        with torch.no_grad():
            # h_pre bias: round-robin one-hot via sigmoid.
            # sigmoid(5) ≈ 0.993, sigmoid(-5) ≈ 0.007
            self.bias[: self.n_streams].fill_(-5.0)
            self.bias[stream] = 5.0

            # h_post bias: 2 * sigmoid(0) = 1.0
            self.bias[self.n_streams :].fill_(0.0)

            # Start alpha small so weights are near-constant at init.
            self.alpha_pre.fill_(0.01)
            self.alpha_post.fill_(0.01)

    def _get_weights(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic h_pre and h_post from the multi-stream state.

        :param H: Multi-stream state of shape ``(..., n, d)``.
        :returns: Tuple of (h_pre, h_post), each of shape ``(..., n)``.
        """
        flat = H.flatten(-2, -1)  # (..., n*d)
        proj = self.phi(flat)  # (..., 2n)

        h_pre = torch.sigmoid(
            self.alpha_pre * proj[..., : self.n_streams] + self.bias[: self.n_streams]
        )
        h_post = 2.0 * torch.sigmoid(
            self.alpha_post * proj[..., self.n_streams :] + self.bias[self.n_streams :]
        )
        return h_pre, h_post

    def merge(self, H: torch.Tensor) -> torch.Tensor:
        """
        Merge multi-stream state into a single sublayer input using dynamic weights.

        :param H: Multi-stream residual state of shape ``(..., n, d)``.
        :returns: Merged tensor of shape ``(..., d)``.
        """
        h_pre, _ = self._get_weights(H)
        # Weighted sum across streams: h_pre (..., n) * H (..., n, d) → (..., d)
        return (H * h_pre.unsqueeze(-1)).sum(dim=-2)

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Distribute sublayer output back into all streams via identity residual.

        :param residual: Multi-stream state of shape ``(..., n, d)``.
        :param x: Sublayer output of shape ``(..., d)``.
        :returns: Updated multi-stream state of shape ``(..., n, d)``.
        """
        _, h_post = self._get_weights(residual)
        # h_post (..., n) → (..., n, 1), x (..., d) → (..., 1, d)
        return residual + h_post.unsqueeze(-1) * x.unsqueeze(-2)
