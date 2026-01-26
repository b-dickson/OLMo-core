"""
Tests for the attention scaling ladder experiment script.

Tests the AttentionScalingModelConfigurator and verifies that different attention
types produce valid configurations.
"""

import pytest

from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.fla import FLAConfig
from olmo_core.nn.transformer import TransformerBlockType


def _fla_available():
    """Check if the FLA library is available."""
    try:
        import fla  # noqa: F401

        return True
    except ImportError:
        return False


class TestAttentionScalingModelConfigurator:
    """Tests for AttentionScalingModelConfigurator."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return TokenizerConfig.dolma2()

    @pytest.fixture
    def configurator_kwargs(self):
        """Common kwargs for configurator."""
        return {
            "rank_microbatch_size": None,
        }

    def test_full_attention_config(self, tokenizer, configurator_kwargs):
        """Test that full attention disables sliding window."""
        from scripts.train.ladder.attention_scaling_ladder import (
            AttentionScalingModelConfigurator,
        )

        configurator = AttentionScalingModelConfigurator(
            attention_type="full",
            **configurator_kwargs,
        )

        config = configurator.configure_model(
            size_spec="60M",
            sequence_length=2048,
            tokenizer=tokenizer,
            device_type="NVIDIA H100 80GB HBM3",
        )

        # Full attention should have no sliding window
        assert config.block.attention.sliding_window is None
        # Should still be a default block type
        assert config.block.name == TransformerBlockType.default

    def test_sliding_attention_config(self, tokenizer, configurator_kwargs):
        """Test that sliding attention configures the window correctly."""
        from scripts.train.ladder.attention_scaling_ladder import (
            AttentionScalingModelConfigurator,
        )

        window_size = 2048
        configurator = AttentionScalingModelConfigurator(
            attention_type="sliding",
            window_size=window_size,
            **configurator_kwargs,
        )

        config = configurator.configure_model(
            size_spec="60M",
            sequence_length=2048,
            tokenizer=tokenizer,
            device_type="NVIDIA H100 80GB HBM3",
        )

        # Should have sliding window configured
        assert config.block.attention.sliding_window is not None
        sw = config.block.attention.sliding_window
        assert isinstance(sw, SlidingWindowAttentionConfig)
        # Pattern should be [window_size, window_size, window_size, -1]
        assert sw.pattern == [window_size, window_size, window_size, -1]
        assert sw.force_full_attention_on_last_layer is True

    @pytest.mark.skipif(not _fla_available(), reason="FLA library not available")
    def test_gated_deltanet_config(self, tokenizer, configurator_kwargs):
        """Test that gated_deltanet configures FLA correctly."""
        from scripts.train.ladder.attention_scaling_ladder import (
            AttentionScalingModelConfigurator,
        )

        configurator = AttentionScalingModelConfigurator(
            attention_type="gated_deltanet",
            **configurator_kwargs,
        )

        config = configurator.configure_model(
            size_spec="60M",
            sequence_length=2048,
            tokenizer=tokenizer,
            device_type="NVIDIA H100 80GB HBM3",
        )

        # Should be an FLA block
        assert config.block.name == TransformerBlockType.fla
        assert config.block.fla is not None
        assert isinstance(config.block.fla, FLAConfig)
        assert config.block.fla.name == "GatedDeltaNet"
        # GatedDeltaNet should have use_gate=True
        assert config.block.fla.fla_layer_kwargs.get("use_gate") is True

    @pytest.mark.skipif(not _fla_available(), reason="FLA library not available")
    def test_deltanet_config(self, tokenizer, configurator_kwargs):
        """Test that deltanet configures FLA without gating."""
        from scripts.train.ladder.attention_scaling_ladder import (
            AttentionScalingModelConfigurator,
        )

        configurator = AttentionScalingModelConfigurator(
            attention_type="deltanet",
            **configurator_kwargs,
        )

        config = configurator.configure_model(
            size_spec="60M",
            sequence_length=2048,
            tokenizer=tokenizer,
            device_type="NVIDIA H100 80GB HBM3",
        )

        # Should be an FLA block
        assert config.block.name == TransformerBlockType.fla
        assert config.block.fla is not None
        assert config.block.fla.name == "GatedDeltaNet"
        # DeltaNet should have use_gate=False
        assert config.block.fla.fla_layer_kwargs.get("use_gate") is False

    @pytest.mark.skipif(not _fla_available(), reason="FLA library not available")
    def test_mamba2_config(self, tokenizer, configurator_kwargs):
        """Test that mamba2 configures FLA with Mamba2."""
        from scripts.train.ladder.attention_scaling_ladder import (
            AttentionScalingModelConfigurator,
        )

        configurator = AttentionScalingModelConfigurator(
            attention_type="mamba2",
            **configurator_kwargs,
        )

        config = configurator.configure_model(
            size_spec="60M",
            sequence_length=2048,
            tokenizer=tokenizer,
            device_type="NVIDIA H100 80GB HBM3",
        )

        # Should be an FLA block with Mamba2
        assert config.block.name == TransformerBlockType.fla
        assert config.block.fla is not None
        assert config.block.fla.name == "Mamba2"

    @pytest.mark.skipif(not _fla_available(), reason="FLA library not available")
    def test_rwkv7_config(self, tokenizer, configurator_kwargs):
        """Test that rwkv7 configures FLA with RWKV7Attention."""
        from scripts.train.ladder.attention_scaling_ladder import (
            AttentionScalingModelConfigurator,
        )

        configurator = AttentionScalingModelConfigurator(
            attention_type="rwkv7",
            **configurator_kwargs,
        )

        config = configurator.configure_model(
            size_spec="60M",
            sequence_length=2048,
            tokenizer=tokenizer,
            device_type="NVIDIA H100 80GB HBM3",
        )

        # Should be an FLA block with RWKV7
        assert config.block.name == TransformerBlockType.fla
        assert config.block.fla is not None
        assert config.block.fla.name == "RWKV7Attention"

    def test_invalid_attention_type(self, tokenizer, configurator_kwargs):
        """Test that invalid attention type raises ValueError."""
        from scripts.train.ladder.attention_scaling_ladder import (
            AttentionScalingModelConfigurator,
        )

        configurator = AttentionScalingModelConfigurator(
            attention_type="invalid_type",
            **configurator_kwargs,
        )

        with pytest.raises(ValueError, match="Unknown attention type"):
            configurator.configure_model(
                size_spec="60M",
                sequence_length=2048,
                tokenizer=tokenizer,
                device_type="NVIDIA H100 80GB HBM3",
            )


class TestScalingLawFitting:
    """Tests for the scaling law fitting functions."""

    def test_chinchilla_loss_function(self):
        """Test the Chinchilla loss function computation."""
        import numpy as np

        from scripts.analysis.fit_scaling_laws import chinchilla_loss

        # Test with known values
        N = np.array([1e8, 1e9])  # 100M, 1B params
        D = np.array([2e9, 2e10])  # 2B, 20B tokens
        E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28

        L = chinchilla_loss((N, D), E, A, alpha, B, beta)

        # Should return array of same shape
        assert L.shape == N.shape
        # Loss should be positive
        assert np.all(L > 0)
        # Loss should decrease with more params/data
        assert L[0] > L[1]

    def test_chinchilla_tokens_calculation(self):
        """Test the Chinchilla-optimal tokens calculation."""
        from scripts.analysis.fit_scaling_laws import chinchilla_tokens

        # 1x Chinchilla: 20 * N tokens
        assert chinchilla_tokens(100_000_000, 1.0) == 2_000_000_000
        # 0.5x Chinchilla
        assert chinchilla_tokens(100_000_000, 0.5) == 1_000_000_000
        # 2x Chinchilla
        assert chinchilla_tokens(100_000_000, 2.0) == 4_000_000_000

    @pytest.mark.skipif(
        not _scipy_available(),
        reason="scipy not available for curve fitting",
    )
    def test_fit_scaling_law_synthetic(self):
        """Test scaling law fitting with synthetic data."""
        import numpy as np
        import pandas as pd

        from scripts.analysis.fit_scaling_laws import chinchilla_loss, fit_scaling_law

        # Generate synthetic data from known parameters
        true_params = {"E": 1.7, "A": 400.0, "alpha": 0.35, "B": 400.0, "beta": 0.28}

        np.random.seed(42)
        N_values = np.array([60e6, 100e6, 190e6, 370e6, 760e6])
        D_values = 20 * N_values  # 1x Chinchilla

        L_values = chinchilla_loss(
            (N_values, D_values),
            true_params["E"],
            true_params["A"],
            true_params["alpha"],
            true_params["B"],
            true_params["beta"],
        )
        # Add small noise
        L_values += np.random.normal(0, 0.01, size=L_values.shape)

        df = pd.DataFrame(
            {
                "num_params": N_values,
                "tokens": D_values,
                "loss": L_values,
            }
        )

        # Fit and check we recover approximately the true parameters
        fitted = fit_scaling_law(df)

        # Should be close to true values (within tolerance due to noise)
        assert abs(fitted["E"] - true_params["E"]) < 0.1
        assert abs(fitted["alpha"] - true_params["alpha"]) < 0.1
        assert abs(fitted["beta"] - true_params["beta"]) < 0.1

    def test_evaluate_predictions(self):
        """Test prediction evaluation metrics."""
        import numpy as np
        import pandas as pd

        from scripts.analysis.fit_scaling_laws import evaluate_predictions

        # Create test data
        params = {"E": 1.7, "A": 400.0, "alpha": 0.35, "B": 400.0, "beta": 0.28}
        test_df = pd.DataFrame(
            {
                "num_params": [600e6, 1e9],
                "tokens": [12e9, 20e9],
                "loss": [2.5, 2.3],  # Approximate expected losses
            }
        )

        result = evaluate_predictions(params, test_df)

        # Should have required keys
        assert "mse" in result
        assert "mae" in result
        assert "mape" in result
        assert "predicted" in result
        assert "observed" in result
        assert "residuals" in result

        # Metrics should be non-negative
        assert result["mse"] >= 0
        assert result["mae"] >= 0
        assert result["mape"] >= 0


def _scipy_available():
    """Check if scipy is available."""
    try:
        from scipy.optimize import curve_fit  # noqa: F401

        return True
    except ImportError:
        return False
