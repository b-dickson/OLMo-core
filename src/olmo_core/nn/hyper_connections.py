"""
Identity Hyper Connections for multi-stream residual connections.

This implements the "Identity HC" variant from the mHC paper (DeepSeek, 2512.24880),
which sets H_res = I (identity) and learns only the H_pre/H_post vectors per sublayer.

Standard residual: ``x_{l+1} = x_l + F(x_l)``

Identity HC with *n* streams (state ``H`` in ``R^{n x d}``):

For each sublayer ``F``:
  1. ``h_input = H_pre @ H``   — merge *n* streams into one vector
  2. ``h_output = F(h_input)``  — run the sublayer
  3. ``H_new = H + H_post^T * h_output``  — distribute output back to all streams (identity residual)

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
    :param init_strategy: Initialization strategy for h_pre/h_post vectors.
        ``"broadcast"`` recovers standard residual at init;
        ``"one_hot"`` keeps only the first stream active at init;
        ``"round_robin"`` uses the paper's round-robin one-hot init where each
        sublayer reads from a different stream (sublayer_idx % n_streams).
    """

    n_streams: int = 4
    init_strategy: str = "round_robin"


class HyperConnectionStream(nn.Module):
    """
    Drop-in replacement for :class:`~olmo_core.nn.residual_stream.ResidualStream`
    that implements Identity Hyper Connections.

    Each instance holds two learned vectors (``h_pre``, ``h_post``) of size ``n_streams``.
    The ``merge()`` method maps the multi-stream state ``(B, S, n, d)`` to a single
    sublayer input ``(B, S, d)``, and ``forward()`` distributes the sublayer output
    back into all streams via the identity residual.

    :param n_streams: Number of parallel residual streams.
    :param init_strategy: ``"broadcast"``, ``"one_hot"``, or ``"round_robin"``.
    :param sublayer_idx: Global sublayer index (used by ``"round_robin"`` init to assign
        each sublayer to a different stream via ``sublayer_idx % n_streams``).
    """

    def __init__(self, n_streams: int, init_strategy: str = "round_robin", sublayer_idx: int = 0):
        super().__init__()
        self.n_streams = n_streams
        self.init_strategy = init_strategy
        self.sublayer_idx = sublayer_idx
        self.h_pre = nn.Parameter(torch.empty(n_streams))
        self.h_post = nn.Parameter(torch.empty(n_streams))
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_strategy == "broadcast":
            nn.init.constant_(self.h_pre, 1.0 / self.n_streams)
            nn.init.ones_(self.h_post)
        elif self.init_strategy == "one_hot":
            nn.init.zeros_(self.h_pre)
            nn.init.zeros_(self.h_post)
            with torch.no_grad():
                self.h_pre[0] = 1.0
                self.h_post[0] = 1.0
        elif self.init_strategy == "round_robin":
            # Paper's init: each sublayer reads from one stream (round-robin),
            # writes back to all streams equally. This spreads sublayer assignments
            # across streams so each stream gets ~1/n share of sublayers.
            stream = self.sublayer_idx % self.n_streams
            nn.init.zeros_(self.h_pre)
            nn.init.ones_(self.h_post)
            with torch.no_grad():
                self.h_pre[stream] = 1.0
        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy!r}")

    def merge(self, H: torch.Tensor) -> torch.Tensor:
        """
        Merge multi-stream state into a single sublayer input.

        :param H: Multi-stream residual state of shape ``(..., n, d)``.
        :returns: Merged tensor of shape ``(..., d)``.
        """
        return torch.einsum("...nd,n->...d", H, self.h_pre)

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Distribute sublayer output back into all streams via identity residual.

        :param residual: Multi-stream state of shape ``(..., n, d)``.
        :param x: Sublayer output of shape ``(..., d)``.
        :returns: Updated multi-stream state of shape ``(..., n, d)``.
        """
        # h_post: (n,) -> (n, 1) for broadcasting with x: (..., d) -> (..., 1, d)
        return residual + self.h_post.unsqueeze(-1) * x.unsqueeze(-2)
