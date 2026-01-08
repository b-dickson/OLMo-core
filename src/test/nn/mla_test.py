"""
Tests for Multi-head Latent Attention (MLA) with DeepSeek Sparse Attention (DSA).
"""

from typing import Optional

import pytest
import torch

from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    IndexerConfig,
    LightningIndexer,
    MLAConfig,
    MLAKVCacheManager,
    MLAttention,
)
from olmo_core.testing import DEVICES, GPU_MARKS
from olmo_core.utils import seed_all

# Tolerances for numerical comparisons
BF16_RTOL = 1e-4
BF16_ATOL = 5e-3


class TestMLAKVCacheManager:
    """Tests for MLAKVCacheManager."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device: torch.device):
        """Test cache manager initialization."""
        batch_size = 2
        max_seq_len = 1024
        kv_lora_rank = 512
        rope_head_dim = 64

        cache = MLAKVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kv_lora_rank=kv_lora_rank,
            rope_head_dim=rope_head_dim,
            device=device,
        )

        assert cache.kv_cache.shape == (batch_size, max_seq_len, kv_lora_rank)
        assert cache.rope_cache.shape == (batch_size, max_seq_len, rope_head_dim)

    @pytest.mark.parametrize("device", DEVICES)
    def test_update_and_get(self, device: torch.device):
        """Test cache update and retrieval."""
        batch_size = 2
        max_seq_len = 1024
        kv_lora_rank = 512
        rope_head_dim = 64
        seq_len = 32

        cache = MLAKVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kv_lora_rank=kv_lora_rank,
            rope_head_dim=rope_head_dim,
            device=device,
        )

        # Create test data with same dtype as cache (bfloat16)
        dtype = cache.kv_cache.dtype
        kv_latent = torch.randn(batch_size, seq_len, kv_lora_rank, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, seq_len, rope_head_dim, device=device, dtype=dtype)

        # Update cache
        cache.update(kv_latent, k_rope, start_pos=0)

        # Retrieve and verify
        cached_kv, cached_rope = cache.get_cached(batch_size, seq_len)
        assert torch.allclose(cached_kv, kv_latent)
        assert torch.allclose(cached_rope, k_rope)

    @pytest.mark.parametrize("device", DEVICES)
    def test_reset(self, device: torch.device):
        """Test cache reset."""
        cache = MLAKVCacheManager(
            batch_size=2,
            max_seq_len=1024,
            kv_lora_rank=512,
            rope_head_dim=64,
            device=device,
        )

        # Fill with data
        cache.kv_cache.fill_(1.0)

        # Reset
        cache.reset()

        assert cache.kv_cache.sum() == 0


class TestMLAttention:
    """Tests for MLAttention."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.bfloat16, id="bf16", marks=GPU_MARKS),
            pytest.param(torch.float32, id="fp32"),
        ],
    )
    def test_forward_basic(self, device: torch.device, dtype: torch.dtype):
        """Test basic forward pass."""
        seed_all(42)

        d_model = 512
        n_heads = 8
        batch_size = 2
        seq_len = 64

        mla_config = MLAConfig(
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            v_head_dim=64,
        )

        mla = MLAttention(
            d_model=d_model,
            n_heads=n_heads,
            mla_config=mla_config,
            dtype=dtype,
            init_device=str(device),
        ).to(device)

        x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)

        # Forward pass
        y = mla(x)

        # Check output shape
        assert y.shape == x.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_forward_with_mask(self, device: torch.device):
        """Test forward pass with attention mask."""
        seed_all(42)

        d_model = 256
        n_heads = 4
        batch_size = 2
        seq_len = 32

        mla_config = MLAConfig(
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
        )

        mla = MLAttention(
            d_model=d_model,
            n_heads=n_heads,
            mla_config=mla_config,
            init_device=str(device),
        ).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Create causal mask
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

        # Forward pass with mask
        y = mla(x, mask=mask)

        assert y.shape == x.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward(self, device: torch.device):
        """Test gradient flow through MLA."""
        seed_all(42)

        d_model = 256
        n_heads = 4
        batch_size = 2
        seq_len = 32

        mla_config = MLAConfig(
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
        )

        mla = MLAttention(
            d_model=d_model,
            n_heads=n_heads,
            mla_config=mla_config,
            init_device=str(device),
        ).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

        # Forward and backward
        y = mla(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check model gradients
        for name, param in mla.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_no_query_compression(self, device: torch.device):
        """Test MLA without query compression (q_lora_rank=0)."""
        seed_all(42)

        d_model = 256
        n_heads = 4
        batch_size = 2
        seq_len = 32

        mla_config = MLAConfig(
            q_lora_rank=0,  # No query compression
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
        )

        mla = MLAttention(
            d_model=d_model,
            n_heads=n_heads,
            mla_config=mla_config,
            init_device=str(device),
        ).to(device)

        # Check that w_q_down is None
        assert mla.w_q_down is None
        assert mla.q_norm is None

        x = torch.randn(batch_size, seq_len, d_model, device=device)
        y = mla(x)

        assert y.shape == x.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_kv_cache_manager_integration(self, device: torch.device):
        """Test MLA with KV cache manager."""
        seed_all(42)

        d_model = 256
        n_heads = 4
        batch_size = 2
        max_seq_len = 128
        prompt_len = 32

        mla_config = MLAConfig(
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
        )

        mla = MLAttention(
            d_model=d_model,
            n_heads=n_heads,
            mla_config=mla_config,
            init_device=str(device),
        ).to(device)

        # Initialize KV cache
        mla.init_kv_cache_manager(batch_size, max_seq_len, device=device)
        assert mla.kv_cache_manager is not None

        # Process prompt
        x = torch.randn(batch_size, prompt_len, d_model, device=device)
        y = mla(x, start_pos=0)

        assert y.shape == x.shape

        # Generate single token
        x_next = torch.randn(batch_size, 1, d_model, device=device)
        y_next = mla(x_next, start_pos=prompt_len)

        assert y_next.shape == (batch_size, 1, d_model)


class TestLightningIndexer:
    """Tests for LightningIndexer."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device: torch.device):
        """Test indexer initialization."""
        config = IndexerConfig(
            n_heads=32,
            head_dim=64,
            top_k=64,
            use_hadamard=False,
        )

        indexer = LightningIndexer(
            config=config,
            q_input_dim=128,
            d_model=256,
            rope_head_dim=16,
            init_device=str(device),
        ).to(device)

        assert indexer.top_k == 64
        assert indexer.n_heads == 32
        assert indexer.head_dim == 64

    @pytest.mark.parametrize("device", DEVICES)
    def test_top_k_selection(self, device: torch.device):
        """Test top-k index selection."""
        seed_all(42)

        config = IndexerConfig(
            n_heads=8,
            head_dim=32,
            top_k=16,
            use_hadamard=False,
        )

        indexer = LightningIndexer(
            config=config,
            q_input_dim=64,
            d_model=128,
            rope_head_dim=16,
            init_device=str(device),
        ).to(device)

        batch_size = 2
        seq_len = 32

        x = torch.randn(batch_size, seq_len, 128, device=device)
        q_input = torch.randn(batch_size, seq_len, 64, device=device)

        indices = indexer(x, q_input)

        # Should return top-k indices
        assert indices is not None
        assert indices.shape == (batch_size, seq_len, config.top_k)

        # Indices should be in valid range
        assert indices.min() >= 0
        assert indices.max() < seq_len


class TestAttentionConfigMLA:
    """Tests for AttentionConfig with MLA."""

    def test_build_mla_basic(self):
        """Test building MLA from AttentionConfig."""
        config = AttentionConfig(
            name=AttentionType.mla,
            n_heads=8,
            mla=MLAConfig(
                q_lora_rank=256,
                kv_lora_rank=128,
            ),
        )

        mla = config.build(d_model=512, layer_idx=0, n_layers=12)

        assert isinstance(mla, MLAttention)
        assert mla.n_heads == 8
        assert mla.q_lora_rank == 256
        assert mla.kv_lora_rank == 128

    def test_build_mla_with_indexer(self):
        """Test building MLA with indexer from AttentionConfig."""
        config = AttentionConfig(
            name=AttentionType.mla,
            n_heads=8,
            mla=MLAConfig(
                q_lora_rank=256,
                kv_lora_rank=128,
            ),
            indexer=IndexerConfig(
                enabled=True,
                top_k=64,
                use_hadamard=False,
            ),
        )

        mla = config.build(d_model=512, layer_idx=0, n_layers=12)

        assert isinstance(mla, MLAttention)
        assert mla.indexer is not None
        assert mla.indexer.top_k == 64

    def test_build_mla_without_config_raises(self):
        """Test that building MLA without mla config raises error."""
        config = AttentionConfig(
            name=AttentionType.mla,
            n_heads=8,
            # mla config missing
        )

        with pytest.raises(Exception):  # OLMoConfigurationError
            config.build(d_model=512, layer_idx=0, n_layers=12)


class TestMLAWithSparseAttention:
    """Tests for MLA with sparse attention via indexer."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_sparse_attention_forward(self, device: torch.device):
        """Test MLA forward with sparse attention enabled."""
        seed_all(42)

        d_model = 256
        n_heads = 4
        batch_size = 2
        seq_len = 128

        mla_config = MLAConfig(
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
        )

        indexer_config = IndexerConfig(
            enabled=True,
            n_heads=16,
            head_dim=32,
            top_k=32,
            use_hadamard=False,
        )

        mla = MLAttention(
            d_model=d_model,
            n_heads=n_heads,
            mla_config=mla_config,
            indexer_config=indexer_config,
            init_device=str(device),
        ).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Forward pass with sparse attention
        y = mla(x)

        assert y.shape == x.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_toggle_sparse_attention(self, device: torch.device):
        """Test toggling sparse attention on/off at runtime."""
        seed_all(42)

        d_model = 256
        n_heads = 4
        batch_size = 2

        mla_config = MLAConfig(
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
        )

        indexer_config = IndexerConfig(
            enabled=True,
            n_heads=16,
            head_dim=32,
            top_k=32,
            use_hadamard=False,
        )

        mla = MLAttention(
            d_model=d_model,
            n_heads=n_heads,
            mla_config=mla_config,
            indexer_config=indexer_config,
            init_device=str(device),
        ).to(device)

        x = torch.randn(batch_size, 64, d_model, device=device)

        # With sparse attention enabled (default)
        assert mla.use_sparse_attention is True
        y_sparse = mla(x)

        # Disable sparse attention
        mla.use_sparse_attention = False
        y_dense = mla(x)

        # Both should produce valid outputs
        assert y_sparse.shape == x.shape
        assert y_dense.shape == x.shape


class TestMLANumFlops:
    """Tests for FLOP counting."""

    def test_num_flops_per_token(self):
        """Test FLOP counting for MLA."""
        d_model = 512
        n_heads = 8
        seq_len = 1024

        mla_config = MLAConfig(
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            v_head_dim=64,
        )

        mla = MLAttention(
            d_model=d_model,
            n_heads=n_heads,
            mla_config=mla_config,
        )

        flops = mla.num_flops_per_token(seq_len)

        # Should be a reasonable positive number
        assert flops > 0
        assert isinstance(flops, int)


class TestStandardAttentionWithDSA:
    """
    Tests for standard Attention class with DSA (Lightning Indexer).

    This tests the decoupled DSA implementation where sparse attention
    can be used with standard attention (not just MLA).
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_attention_with_indexer_instantiation(self, device: torch.device):
        """Test that standard Attention can be instantiated with IndexerConfig."""
        from olmo_core.nn.attention import Attention

        indexer_config = IndexerConfig(
            enabled=True,
            n_heads=16,
            head_dim=32,
            top_k=32,
            use_hadamard=False,
        )

        attn = Attention(
            d_model=256,
            n_heads=8,
            indexer_config=indexer_config,
            init_device=str(device),
        ).to(device)

        # Check indexer was created
        assert attn.indexer is not None
        assert attn.indexer.top_k == 32
        assert attn.use_sparse_attention is True

    @pytest.mark.parametrize("device", DEVICES)
    def test_attention_sparse_forward(self, device: torch.device):
        """Test standard Attention forward with sparse attention."""
        from olmo_core.nn.attention import Attention

        seed_all(42)

        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 128

        indexer_config = IndexerConfig(
            enabled=True,
            n_heads=16,
            head_dim=32,
            top_k=32,
            use_hadamard=False,
        )

        attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            indexer_config=indexer_config,
            init_device=str(device),
        ).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device)
        y = attn(x)

        assert y.shape == x.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_attention_indexer_backward(self, device: torch.device):
        """Test gradients flow through standard Attention with indexer.

        Note: The indexer itself does NOT receive gradients through the normal
        backward pass because top-k selection is non-differentiable. The indexer
        is trained via an auxiliary alignment loss in DSATrainingCallback.
        This test verifies that the main attention parameters and input receive
        gradients correctly.
        """
        from olmo_core.nn.attention import Attention

        seed_all(42)

        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 128

        indexer_config = IndexerConfig(
            enabled=True,
            n_heads=16,
            head_dim=32,
            top_k=32,
            use_hadamard=False,
        )

        attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            indexer_config=indexer_config,
            init_device=str(device),
        ).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

        # Forward and backward
        y = attn(x)
        loss = y.sum()
        loss.backward()

        # Check input gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check main attention parameter gradients
        for name, param in attn.named_parameters():
            if param.requires_grad and not name.startswith("indexer."):
                assert param.grad is not None, f"No gradient for {name}"

        # Note: Indexer gradients are None because top-k selection is non-differentiable.
        # The indexer is trained via auxiliary alignment loss in DSATrainingCallback.

    def test_attention_config_with_indexer(self):
        """Test AttentionConfig builds standard Attention with indexer."""
        config = AttentionConfig(
            name=AttentionType.default,
            n_heads=8,
            indexer=IndexerConfig(
                enabled=True,
                top_k=32,
                use_hadamard=False,
            ),
        )

        from olmo_core.nn.attention import Attention

        attn = config.build(d_model=256, layer_idx=0, n_layers=12)

        assert isinstance(attn, Attention)
        assert attn.indexer is not None
        assert attn.indexer.top_k == 32

    @pytest.mark.parametrize("device", DEVICES)
    def test_toggle_sparse_attention(self, device: torch.device):
        """Test toggling sparse attention on/off at runtime."""
        from olmo_core.nn.attention import Attention

        seed_all(42)

        indexer_config = IndexerConfig(
            enabled=True,
            n_heads=16,
            head_dim=32,
            top_k=32,
            use_hadamard=False,
        )

        attn = Attention(
            d_model=256,
            n_heads=8,
            indexer_config=indexer_config,
            init_device=str(device),
        ).to(device)

        x = torch.randn(2, 128, 256, device=device)

        # With sparse attention enabled (default)
        assert attn.use_sparse_attention is True
        y_sparse = attn(x)

        # Disable sparse attention
        attn.use_sparse_attention = False
        y_dense = attn(x)

        # Both should produce valid outputs
        assert y_sparse.shape == x.shape
        assert y_dense.shape == x.shape

        # Outputs may differ since sparse uses top-k selection
        # (This is expected behavior, not a bug)
