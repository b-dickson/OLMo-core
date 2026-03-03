"""
Integration tests for Identity Hyper Connections with the full transformer stack.
"""

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.hyper_connections import (
    HyperConnectionStream,
    IdentityHyperConnectionConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)


def _make_config(
    block_type: TransformerBlockType = TransformerBlockType.default,
    n_streams: int = 4,
    n_layers: int = 2,
    d_model: int = 64,
    vocab_size: int = 128,
    **block_kwargs,
) -> TransformerConfig:
    block = TransformerBlockConfig(
        name=block_type,
        sequence_mixer=AttentionConfig(n_heads=4),
        feed_forward=FeedForwardConfig(hidden_size=d_model * 4),
        layer_norm=LayerNormConfig(),
        **block_kwargs,
    )
    return TransformerConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        block=block,
        lm_head=LMHeadConfig(),
        hyper_connections=IdentityHyperConnectionConfig(n_streams=n_streams),
    )


class TestHCTransformerBlock:
    def test_forward_produces_valid_output(self):
        config = _make_config()
        model = config.build(init_device="cpu")
        model.init_weights()

        B, S = 2, 16
        input_ids = torch.randint(0, 128, (B, S))
        labels = torch.randint(0, 128, (B, S))

        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.isfinite()
        assert output.loss.requires_grad

    @pytest.mark.parametrize(
        "block_type",
        [
            TransformerBlockType.default,
            TransformerBlockType.reordered_norm,
        ],
    )
    def test_block_types(self, block_type: TransformerBlockType):
        config = _make_config(block_type=block_type)
        model = config.build(init_device="cpu")
        model.init_weights()

        input_ids = torch.randint(0, 128, (2, 16))
        logits = model(input_ids)
        assert logits.shape == (2, 16, 128)
        assert logits.isfinite().all()


class TestHCFLABlock:
    @pytest.mark.gpu
    def test_forward(self):
        try:
            from olmo_core.nn.fla import FLAConfig  # noqa: F401
        except ImportError:
            pytest.skip("fla package not available")

        device = "cuda"
        block = TransformerBlockConfig(
            name=TransformerBlockType.fla,
            sequence_mixer=AttentionConfig(n_heads=4),
            feed_forward=FeedForwardConfig(hidden_size=256),
            layer_norm=LayerNormConfig(),
            fla=FLAConfig(name="GatedDeltaNet"),
        )
        config = TransformerConfig(
            d_model=64,
            vocab_size=128,
            n_layers=2,
            block=block,
            lm_head=LMHeadConfig(),
            hyper_connections=IdentityHyperConnectionConfig(n_streams=4),
        )
        model = config.build(init_device=device)
        model.init_weights(device=torch.device(device))

        input_ids = torch.randint(0, 128, (2, 16), device=device)
        logits = model(input_ids)
        assert logits.shape == (2, 16, 128)
        assert logits.isfinite().all()

    def test_build_and_init(self):
        """Verify FLA block with HC can be built and initialized (CPU-safe)."""
        try:
            from olmo_core.nn.fla import FLAConfig  # noqa: F401
        except ImportError:
            pytest.skip("fla package not available")

        block = TransformerBlockConfig(
            name=TransformerBlockType.fla,
            sequence_mixer=AttentionConfig(n_heads=4),
            feed_forward=FeedForwardConfig(hidden_size=256),
            layer_norm=LayerNormConfig(),
            fla=FLAConfig(name="GatedDeltaNet"),
        )
        config = TransformerConfig(
            d_model=64,
            vocab_size=128,
            n_layers=2,
            block=block,
            lm_head=LMHeadConfig(),
            hyper_connections=IdentityHyperConnectionConfig(n_streams=4),
        )
        model = config.build(init_device="cpu")
        model.init_weights()

        # Verify HC streams were applied to FLA blocks.
        for blk in model.blocks.values():
            assert isinstance(blk.fla_residual_stream, HyperConnectionStream)
            assert isinstance(blk.ffn_residual_stream, HyperConnectionStream)


class TestHCFLAHybrid:
    @pytest.mark.gpu
    def test_forward(self):
        try:
            from olmo_core.nn.fla import FLAConfig  # noqa: F401
        except ImportError:
            pytest.skip("fla package not available")

        device = "cuda"
        block = TransformerBlockConfig(
            name=TransformerBlockType.fla_hybrid,
            sequence_mixer=AttentionConfig(n_heads=4),
            feed_forward=FeedForwardConfig(hidden_size=256),
            layer_norm=LayerNormConfig(),
            fla=FLAConfig(name="GatedDeltaNet"),
            fla_hybrid_attention_indices=[1, 3],
        )
        config = TransformerConfig(
            d_model=64,
            vocab_size=128,
            n_layers=4,
            block=block,
            lm_head=LMHeadConfig(),
            hyper_connections=IdentityHyperConnectionConfig(n_streams=4),
        )
        model = config.build(init_device=device)
        model.init_weights(device=torch.device(device))

        input_ids = torch.randint(0, 128, (2, 16), device=device)
        logits = model(input_ids)
        assert logits.shape == (2, 16, 128)
        assert logits.isfinite().all()

    def test_build_and_init(self):
        """Verify hybrid FLA block with HC can be built and initialized (CPU-safe)."""
        try:
            from olmo_core.nn.fla import FLAConfig  # noqa: F401
        except ImportError:
            pytest.skip("fla package not available")

        block = TransformerBlockConfig(
            name=TransformerBlockType.fla_hybrid,
            sequence_mixer=AttentionConfig(n_heads=4),
            feed_forward=FeedForwardConfig(hidden_size=256),
            layer_norm=LayerNormConfig(),
            fla=FLAConfig(name="GatedDeltaNet"),
            fla_hybrid_attention_indices=[1, 3],
        )
        config = TransformerConfig(
            d_model=64,
            vocab_size=128,
            n_layers=4,
            block=block,
            lm_head=LMHeadConfig(),
            hyper_connections=IdentityHyperConnectionConfig(n_streams=4),
        )
        model = config.build(init_device="cpu")
        model.init_weights()

        # Verify HC applied: attention blocks get TransformerBlock RS, FLA blocks get HCS.
        from olmo_core.nn.transformer.block import FLABlock as FLABlockCls

        for blk in model.blocks.values():
            if isinstance(blk, FLABlockCls):
                assert isinstance(blk.fla_residual_stream, HyperConnectionStream)
            else:
                assert isinstance(blk.attention_residual_stream, HyperConnectionStream)


class TestHCNumParams:
    def test_config_matches_model(self):
        config = _make_config()
        model = config.build(init_device="cpu")
        assert config.num_params == model.num_params

    def test_hc_adds_params(self):
        config_no_hc = TransformerConfig(
            d_model=64,
            vocab_size=128,
            n_layers=2,
            block=TransformerBlockConfig(
                name=TransformerBlockType.default,
                sequence_mixer=AttentionConfig(n_heads=4),
                feed_forward=FeedForwardConfig(hidden_size=256),
                layer_norm=LayerNormConfig(),
            ),
            lm_head=LMHeadConfig(),
        )
        config_hc = TransformerConfig(
            d_model=64,
            vocab_size=128,
            n_layers=2,
            block=TransformerBlockConfig(
                name=TransformerBlockType.default,
                sequence_mixer=AttentionConfig(n_heads=4),
                feed_forward=FeedForwardConfig(hidden_size=256),
                layer_norm=LayerNormConfig(),
            ),
            lm_head=LMHeadConfig(),
            hyper_connections=IdentityHyperConnectionConfig(n_streams=4),
        )
        # 2 layers * 2 sublayers * 2 vectors * 4 streams = 32 extra params
        assert config_hc.num_params == config_no_hc.num_params + 32


class TestHCInitWeights:
    def test_init_weights_completes(self):
        config = _make_config()
        model = config.build(init_device="cpu")
        model.init_weights()

        # Verify HC parameters are properly initialized
        for block in model.blocks.values():
            for attr in ["attention_residual_stream", "feed_forward_residual_stream"]:
                stream = getattr(block, attr, None)
                if stream is not None and isinstance(stream, HyperConnectionStream):
                    assert stream.h_pre.isfinite().all()
                    assert stream.h_post.isfinite().all()


class TestHCDisabled:
    def test_no_hc_backward_compatible(self):
        """Model without HC should work identically to before."""
        config = TransformerConfig(
            d_model=64,
            vocab_size=128,
            n_layers=2,
            block=TransformerBlockConfig(
                name=TransformerBlockType.default,
                sequence_mixer=AttentionConfig(n_heads=4),
                feed_forward=FeedForwardConfig(hidden_size=256),
                layer_norm=LayerNormConfig(),
            ),
            lm_head=LMHeadConfig(),
        )
        model = config.build(init_device="cpu")
        model.init_weights()

        input_ids = torch.randint(0, 128, (2, 16))
        logits = model(input_ids)
        assert logits.shape == (2, 16, 128)
