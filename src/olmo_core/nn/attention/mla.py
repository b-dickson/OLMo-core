"""
Multi-head Latent Attention (MLA) with DeepSeek Sparse Attention (DSA).

Based on the DeepSeek-V3.2 implementation:
https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py

MLA compresses the KV cache using low-rank projections, reducing memory by ~93%
while maintaining or improving performance. DSA (via the Lightning Indexer) further
improves efficiency by selecting only the top-k most relevant tokens for attention.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.config import Config

from ..buffer_cache import BufferCache
from ..layer_norm import RMSNorm
from ..rope import ComplexRotaryEmbedding, RoPEConfig
from .backend import AttentionBackend
from .ring import RingAttentionLoadBalancerType

__all__ = [
    "MLAConfig",
    "IndexerConfig",
    "LightningIndexer",
    "MLAAttentionBackend",
    "MLAKVCacheManager",
    "MLAttention",
]

log = logging.getLogger(__name__)

# Hadamard transform, Deepseek uses this, 
# should improve DSA top-k selection quality ?
try:
    from fast_hadamard_transform import hadamard_transform

    HAS_HADAMARD = True
except ImportError:
    HAS_HADAMARD = False
    hadamard_transform = None  # type: ignore


@dataclass
class MLAConfig(Config):
    """
    Configuration for Multi-head Latent Attention.

    MLA uses low-rank projections to compress queries and key-values,
    significantly reducing the KV cache size while maintaining performance.
    """

    q_lora_rank: int = 1536
    """Rank for low-rank query projection. Set to 0 to disable query compression."""

    kv_lora_rank: int = 512
    """Rank for low-rank key/value projection (the latent dimension)."""

    qk_nope_head_dim: int = 128
    """Dimensionality of non-positional (content) query/key projections."""

    qk_rope_head_dim: int = 64
    """Dimensionality of rotary-positional query/key projections."""

    v_head_dim: int = 128
    """Dimensionality of value projections per head."""

    @property
    def qk_head_dim(self) -> int:
        """Total query/key head dimension (nope + rope)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass
class IndexerConfig(Config):
    """
    Configuration for Lightning Indexer (sparse attention via top-k selection).

    The indexer computes lightweight relevance scores for each query against
    all keys, then selects the top-k most relevant tokens for attention.
    """

    enabled: bool = True
    """Whether to enable sparse attention via the indexer."""

    n_heads: int = 64
    """Number of indexer attention heads (can differ from MLA heads)."""

    head_dim: int = 128
    """Dimensionality of indexer head projections."""

    top_k: int = 2048
    """Number of tokens to select for sparse attention."""

    use_hadamard: bool = True
    """
    Use Hadamard transform before computing index scores.
    Requires fast-hadamard-transform library.
    """


class MLAAttentionBackend(AttentionBackend):
    """
    Attention backend for MLA with asymmetric head dimensions (qk_head_dim != v_head_dim).
    """

    def __init__(
        self,
        *,
        qk_head_dim: int,
        v_head_dim: int,
        n_heads: int,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            head_dim=qk_head_dim,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=(-1, -1),
            cache=cache,
        )
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

    @classmethod
    def assert_supported(cls):
        pass

    @classmethod
    def assert_supports_swa(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support sliding window attention")

    @classmethod
    def assert_supports_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support context parallelism")

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support KV caching")

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = True,
    ) -> Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

        return out.transpose(1, 2)


class LightningIndexer(nn.Module):
    """
    Lightning indexer for top-k token selection with optional FP8 quantization.

    The indexer computes relevance scores between queries and keys using a
    lightweight attention mechanism, then selects the top-k tokens for the
    main attention computation.

    This module can be used with either:
    - MLA: Pass `q_input_dim=q_lora_rank` to use compressed queries
    - Standard attention: Pass `q_input_dim=d_model` to project from input directly

    :param config: Indexer configuration.
    :param q_input_dim: Dimension of query input (q_lora_rank for MLA, d_model for standard).
    :param d_model: Model hidden dimension.
    :param rope_head_dim: Dimension for RoPE positional encoding (0 to disable).
    :param dtype: Data type for parameters.
    :param init_device: Device to initialize weights on.
    """

    def __init__(
        self,
        config: IndexerConfig,
        q_input_dim: int,
        d_model: int,
        rope_head_dim: int = 0,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.rope_head_dim = rope_head_dim

        # Check for optional dependencies
        self.use_hadamard = config.use_hadamard and HAS_HADAMARD
        if config.use_hadamard and not HAS_HADAMARD:
            warnings.warn(
                "use_hadamard=True but fast-hadamard-transform not installed. "
                "Install with: uv pip install fast-hadamard-transform. "
                "Falling back to identity transform."
            )

        # Query projection
        # For MLA: projects from q_lora_rank (compressed queries)
        # For standard attention: projects from d_model (input directly)
        self.w_q = nn.Linear(
            q_input_dim,
            config.n_heads * config.head_dim,
            bias=False,
            dtype=dtype,
            device=init_device,
        )

        # Key projection
        self.w_k = nn.Linear(
            d_model,
            config.head_dim,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.k_norm = nn.LayerNorm(
            config.head_dim,
            elementwise_affine=True,
            dtype=dtype,
            device=init_device,
        )

        # Head weights (computed in fp32 like DeepSeek reference)
        self.w_weights = nn.Linear(
            d_model,
            config.n_heads,
            bias=False,
            dtype=torch.float32,
            device=init_device,
        )

        self.softmax_scale = config.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
        q_input: Tensor,
        freqs_cis: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute top-k indices for sparse attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            q_input: Query input tensor. For MLA, this is the compressed query
                of shape (batch_size, seq_len, q_lora_rank). For standard attention,
                this should be the input x of shape (batch_size, seq_len, d_model).
            freqs_cis: Precomputed rotary embedding frequencies
            mask: Optional causal mask

        Returns:
            Tensor of shape (batch_size, seq_len, top_k) containing selected indices.
        """
        bsz, seq_len, _ = x.shape

        # Compute index queries
        q = self.w_q(q_input)  # (bsz, seq, n_heads * head_dim)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)

        # Split into RoPE and non-RoPE components (like DeepSeek reference)
        q_rope = q[..., : self.rope_head_dim]
        q_nope = q[..., self.rope_head_dim :]

        # Apply RoPE to positional component
        if freqs_cis is not None:
            q_rope = self._apply_rotary_emb(q_rope, freqs_cis)

        q = torch.cat([q_rope, q_nope], dim=-1)

        # Compute index keys
        k = self.w_k(x)  # (bsz, seq, head_dim)
        k = self.k_norm(k)

        # Split and apply RoPE to keys
        k_rope = k[..., : self.rope_head_dim]
        k_nope = k[..., self.rope_head_dim :]

        if freqs_cis is not None:
            k_rope = self._apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis).squeeze(2)

        k = torch.cat([k_rope, k_nope], dim=-1)

        # Apply Hadamard transform if enabled
        if self.use_hadamard:
            assert hadamard_transform is not None
            q = hadamard_transform(q.contiguous(), scale=self.head_dim**-0.5)
            k = hadamard_transform(k.contiguous(), scale=self.head_dim**-0.5)

        # Compute scores: (bsz, seq_q, n_heads, seq_k)
        scores = torch.einsum("bqhd,bkd->bqhk", q, k) * self.softmax_scale

        # Apply head weights and sum across heads
        weights = self.w_weights(x.float()) * (self.n_heads**-0.5)  # (bsz, seq, n_heads)
        scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # (bsz, seq_q, seq_k)

        # Apply causal mask if provided
        if mask is not None:
            scores = scores + mask

        # Select top-k indices
        top_k_indices = scores.topk(min(self.top_k, seq_len), dim=-1).indices

        return top_k_indices

    def _apply_rotary_emb(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        """Apply rotary positional embeddings."""
        # This is a simplified version - the actual implementation should match
        # the RoPE implementation used in the main attention module
        dtype = x.dtype
        x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        y = torch.view_as_real(x * freqs_cis).flatten(-2)
        return y.to(dtype)


class MLAKVCacheManager(nn.Module):
    """
    KV cache manager for MLA - stores compressed latents instead of full K, V.

    This provides ~93% memory reduction compared to standard KV caching by
    storing only the low-rank latent representation and RoPE components.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        rope_head_dim: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim

        # Compressed KV cache (much smaller than full K, V)
        self.register_buffer(
            "kv_cache",
            torch.zeros(batch_size, max_seq_len, kv_lora_rank, dtype=dtype, device=device),
            persistent=False,
        )
        # Separate RoPE cache (needed for position encoding)
        self.register_buffer(
            "rope_cache",
            torch.zeros(batch_size, max_seq_len, rope_head_dim, dtype=dtype, device=device),
            persistent=False,
        )

    def update(self, kv_latent: Tensor, k_rope: Tensor, start_pos: int) -> None:
        """Update cache with new latent and RoPE values."""
        bsz, seq_len, _ = kv_latent.shape
        end_pos = start_pos + seq_len

        self.kv_cache[:bsz, start_pos:end_pos] = kv_latent
        self.rope_cache[:bsz, start_pos:end_pos] = k_rope

    def get_cached(self, bsz: int, end_pos: int) -> Tuple[Tensor, Tensor]:
        """Retrieve cached latent and RoPE values."""
        return self.kv_cache[:bsz, :end_pos], self.rope_cache[:bsz, :end_pos]

    def reset(self, batch_size: Optional[int] = None, max_seq_len: Optional[int] = None) -> None:
        """Reset cache, optionally reallocating with new dimensions."""
        if batch_size is not None and max_seq_len is not None:
            if batch_size != self.batch_size or max_seq_len != self.max_seq_len:
                # Reallocate
                device = self.kv_cache.device
                dtype = self.kv_cache.dtype
                self.kv_cache = torch.zeros(
                    batch_size, max_seq_len, self.kv_lora_rank, dtype=dtype, device=device
                )
                self.rope_cache = torch.zeros(
                    batch_size, max_seq_len, self.rope_head_dim, dtype=dtype, device=device
                )
                self.batch_size = batch_size
                self.max_seq_len = max_seq_len

        self.kv_cache.zero_()
        self.rope_cache.zero_()


class MLAttention(nn.Module):
    """
    Multi-head Latent Attention with optional sparse indexing (DSA).

    MLA uses low-rank projections to compress queries and key-values:
    - Query: x -> q_compressed (low-rank) -> Q (full)
    - KV: x -> kv_compressed (low-rank, cached) -> K, V (decompressed on-the-fly)

    This provides ~93% KV cache reduction while maintaining performance.

    Optionally, the Lightning Indexer can be used for sparse attention,
    selecting only the top-k most relevant tokens.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param mla_config: Configuration for MLA compression.
    :param indexer_config: Optional configuration for sparse attention.
    :param rope: The config for RoPE, if RoPE should be used.
    :param cache: Buffer cache for RoPE.
    :param bias: Include biases with linear layers.
    :param dropout: Dropout probability.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        mla_config: MLAConfig,
        indexer_config: Optional[IndexerConfig] = None,
        rope: Optional[RoPEConfig] = None,
        cache: Optional[BufferCache] = None,
        bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mla_config = mla_config
        self.indexer_config = indexer_config
        self.dropout = dropout

        # Derived dimensions
        self.q_lora_rank = mla_config.q_lora_rank
        self.kv_lora_rank = mla_config.kv_lora_rank
        self.qk_nope_head_dim = mla_config.qk_nope_head_dim
        self.qk_rope_head_dim = mla_config.qk_rope_head_dim
        self.qk_head_dim = mla_config.qk_head_dim
        self.v_head_dim = mla_config.v_head_dim

        # Softmax scale
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.qk_head_dim**-0.5

        # Query low-rank projection
        if self.q_lora_rank > 0:
            self.w_q_down = nn.Linear(d_model, self.q_lora_rank, bias=bias, dtype=dtype, device=init_device)
            self.q_norm = RMSNorm(size=self.q_lora_rank, init_device=init_device)
            self.w_q_up = nn.Linear(
                self.q_lora_rank, n_heads * self.qk_head_dim, bias=bias, dtype=dtype, device=init_device
            )
        else:
            # No query compression
            self.w_q_down = None
            self.q_norm = None
            self.w_q_up = nn.Linear(d_model, n_heads * self.qk_head_dim, bias=bias, dtype=dtype, device=init_device)

        # KV low-rank projection
        # Projects to: kv_lora_rank (for latent) + qk_rope_head_dim (for RoPE)
        self.w_kv_down = nn.Linear(
            d_model,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
            dtype=dtype,
            device=init_device,
        )
        self.kv_norm = RMSNorm(size=self.kv_lora_rank, init_device=init_device)

        # KV decompression: latent -> K_nope + V
        self.w_kv_up = nn.Linear(
            self.kv_lora_rank,
            n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            dtype=dtype,
            device=init_device,
        )

        # Output projection
        self.w_out = nn.Linear(n_heads * self.v_head_dim, d_model, bias=bias, dtype=dtype, device=init_device)

        # Optional indexer for sparse attention
        if indexer_config is not None and indexer_config.enabled:
            q_input_dim = self.q_lora_rank if self.q_lora_rank > 0 else d_model
            self.indexer = LightningIndexer(
                indexer_config,
                q_input_dim=q_input_dim,
                d_model=d_model,
                rope_head_dim=self.qk_rope_head_dim,
                dtype=dtype,
                init_device=init_device,
            )
        else:
            self.indexer = None

        # KV cache manager (initialized lazily)
        self.kv_cache_manager: Optional[MLAKVCacheManager] = None

        # RoPE module for positional embeddings
        # MLA uses ComplexRotaryEmbedding for the rope portion of Q and K
        self.rope: Optional[ComplexRotaryEmbedding] = None
        if rope is not None:
            rope_module = rope.build(self.qk_rope_head_dim, cache=cache)
            assert isinstance(rope_module, ComplexRotaryEmbedding), (
                f"MLA requires ComplexRotaryEmbedding (rope.name='complex'), got {type(rope_module)}"
            )
            self.rope = rope_module

        # Control whether to use sparse attention (can be toggled for DSA training)
        # If False, always uses dense attention even if indexer is present
        self.use_sparse_attention: bool = True

        # Attention backend for dense attention
        self.backend = MLAAttentionBackend(
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            n_heads=n_heads,
            scale=self.softmax_scale,
            dropout_p=dropout,
            cache=cache,
        )

    def init_kv_cache_manager(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the KV cache manager."""
        if dtype is None:
            dtype = self.w_kv_down.weight.dtype
        if device is None:
            device = self.w_kv_down.weight.device

        self.kv_cache_manager = MLAKVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kv_lora_rank=self.kv_lora_rank,
            rope_head_dim=self.qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        start_pos: int = 0,
    ) -> Tensor:
        """
        Forward pass for Multi-head Latent Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            start_pos: Starting position for caching (used in inference)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        bsz, seq_len, _ = x.shape
        end_pos = start_pos + seq_len

        # Get RoPE frequencies from the rope module
        freqs_cis: Optional[Tensor] = None
        if self.rope is not None:
            rope_buffers = self.rope.get_buffers(end_pos, x.device)
            freqs_cis = rope_buffers.freqs_cis
            if freqs_cis is not None:
                freqs_cis = freqs_cis[start_pos:end_pos]

        # Query projection (low-rank)
        if self.w_q_down is not None:
            q_compressed = self.w_q_down(x)
            q_compressed = self.q_norm(q_compressed)
            q = self.w_q_up(q_compressed)
        else:
            q_compressed = x  # For indexer, use x directly
            q = self.w_q_up(x)

        q = q.view(bsz, seq_len, self.n_heads, self.qk_head_dim)

        # Split query into non-positional and positional components
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE to positional component
        if freqs_cis is not None:
            q_rope = self._apply_rotary_emb(q_rope, freqs_cis)

        q = torch.cat([q_nope, q_rope], dim=-1)

        # KV projection (low-rank)
        kv = self.w_kv_down(x)
        kv_latent, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)

        # Apply RoPE to key positional component
        if freqs_cis is not None:
            k_rope = self._apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis).squeeze(2)

        # Update cache with compressed representation
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.update(kv_latent, k_rope, start_pos)
            kv_latent, k_rope = self.kv_cache_manager.get_cached(bsz, end_pos)

        # Decompress KV
        kv_decompressed = self.w_kv_up(kv_latent)
        kv_decompressed = kv_decompressed.view(bsz, -1, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_decompressed.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Combine key components: k_rope is broadcast across heads
        k = torch.cat([k_nope, k_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1)

        # Get sparse indices from indexer (only if sparse attention is enabled)
        top_k_indices = None
        if self.indexer is not None and self.use_sparse_attention:
            top_k_indices = self.indexer(x, q_compressed, freqs_cis, mask)

        # Compute attention (sparse or dense)
        if top_k_indices is not None:
            att = self._sparse_attention(q, k, v, top_k_indices)
        else:
            # Use causal attention for autoregressive decoding
            is_causal = mask is None  # If no explicit mask, assume causal
            att = self._dense_attention(q, k, v, is_causal=is_causal)

        # Output projection
        return self.w_out(att.flatten(2))

    def _dense_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        return self.backend(q, k, v, is_causal=is_causal)

    def _sparse_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        top_k_indices: Tensor,
    ) -> Tensor:
        """Sparse attention using top-k selected indices."""
        bsz, seq_q, n_heads, head_dim = q.shape
        _, seq_k, _, v_head_dim = v.shape

        # Gather selected keys and values
        # top_k_indices: (bsz, seq_q, top_k)
        # Expand indices for gathering: (bsz, seq_q, top_k, n_heads, dim)
        k_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_heads, head_dim)
        v_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_heads, v_head_dim)

        # Gather: k_selected (bsz, seq_q, top_k, n_heads, head_dim)
        k_expanded = k.unsqueeze(1).expand(-1, seq_q, -1, -1, -1)
        v_expanded = v.unsqueeze(1).expand(-1, seq_q, -1, -1, -1)

        k_selected = torch.gather(k_expanded, 2, k_indices)
        v_selected = torch.gather(v_expanded, 2, v_indices)

        # Compute attention scores
        # q: (bsz, seq_q, n_heads, head_dim) -> (bsz, seq_q, n_heads, 1, head_dim)
        # k_selected: (bsz, seq_q, top_k, n_heads, head_dim) -> (bsz, seq_q, n_heads, top_k, head_dim)
        q = q.unsqueeze(-2)  # (bsz, seq_q, n_heads, 1, head_dim)
        k_selected = k_selected.transpose(2, 3)  # (bsz, seq_q, n_heads, top_k, head_dim)
        v_selected = v_selected.transpose(2, 3)  # (bsz, seq_q, n_heads, top_k, v_head_dim)

        scores = torch.einsum("bqhod,bqhkd->bqhok", q, k_selected).squeeze(-2) * self.softmax_scale
        # scores: (bsz, seq_q, n_heads, top_k)

        # Softmax over selected tokens
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Compute output
        out = torch.einsum("bqhk,bqhkd->bqhd", attn_weights, v_selected)
        return out

    def _apply_rotary_emb(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        """Apply rotary positional embeddings."""
        dtype = x.dtype
        x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        y = torch.view_as_real(x * freqs_cis).flatten(-2)
        return y.to(dtype)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        """Apply tensor parallelism to the attention module."""
        # TODO: Implement tensor parallelism for MLA
        raise NotImplementedError("Tensor parallelism not yet implemented for MLAttention")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        """Apply context parallelism to the attention module."""
        # TODO: Implement context parallelism for MLA
        raise NotImplementedError("Context parallelism not yet implemented for MLAttention")

    def num_flops_per_token(self, seq_len: int) -> int:
        # TODO: make sure this is accurate
        """Calculate FLOPs per token for this attention layer."""
        # Query projection
        if self.q_lora_rank > 0:
            q_flops = 2 * self.d_model * self.q_lora_rank  # w_q_down
            q_flops += 2 * self.q_lora_rank * self.n_heads * self.qk_head_dim  # w_q_up
        else:
            q_flops = 2 * self.d_model * self.n_heads * self.qk_head_dim

        # KV projection
        kv_down_flops = 2 * self.d_model * (self.kv_lora_rank + self.qk_rope_head_dim)
        kv_up_flops = 2 * self.kv_lora_rank * self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)

        # Attention (approximate for dense, less for sparse)
        attn_flops = 2 * seq_len * self.n_heads * self.qk_head_dim  # QK^T
        attn_flops += 2 * seq_len * self.n_heads * self.v_head_dim  # AV

        # Output projection
        out_flops = 2 * self.n_heads * self.v_head_dim * self.d_model

        return q_flops + kv_down_flops + kv_up_flops + attn_flops + out_flops
