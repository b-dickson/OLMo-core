"""
Attention Mechanism Scaling Law Experiment

Evaluate how alternative attention mechanisms affect language model scaling behavior.
Fit Chinchilla-style scaling laws for each variant.

Supported Attention Types:
- full: Full softmax attention (disables OLMo3 sliding window)
- sliding: Sliding window attention with configurable window size
- gated_deltanet: GatedDeltaNet linear RNN (with gating)
- deltanet: DeltaNet linear RNN (without gating)
- mamba2: Mamba2 SSM architecture
- rwkv7: RWKV-7 attention

Experiment Grid:
- Model sizes: 60M, 100M, 190M, 370M, 760M (training) + 600M, 1B (test)
- Token budgets: 0.5x, 1x, 2x, 4x Chinchilla-optimal
- Attention types: full, sliding, gated_deltanet, deltanet, mamba2, rwkv7

Usage:
    # Dry run
    python attention_scaling_ladder.py dry-run --size=60M --attention-type=full

    # Launch single run with GatedDeltaNet
    python attention_scaling_ladder.py launch --size=60M --attention-type=gated_deltanet --chinchilla-multiple=1

    # Launch single run with Mamba2
    python attention_scaling_ladder.py launch --size=60M --attention-type=mamba2 --chinchilla-multiple=1

    # Launch all training runs for a given attention type
    python attention_scaling_ladder.py launch-all --attention-type=full --name=attn-scaling-full

    # Get metrics
    python attention_scaling_ladder.py metrics-all --output-dir=./results
"""

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

from olmo_core.config import DType
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import Olmo3ModelConfigurator, TransformerModelConfigurator
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.fla import FLAConfig
from olmo_core.nn.transformer import TransformerBlockType

log = logging.getLogger(__name__)


# Linear attention type configurations
# Each entry maps attention_type name to (FLA layer name, fla_layer_kwargs factory)
LINEAR_ATTENTION_CONFIGS = {
    "gated_deltanet": {
        "name": "GatedDeltaNet",
        "kwargs_factory": lambda d_model, n_heads: {
            # FLA repo: num_heads * head_dim = 0.75 * hidden_size for GatedDeltaNet
            "head_dim": int(0.75 * d_model / n_heads),
            "use_gate": True,
            "allow_neg_eigval": False,
        },
    },
    "deltanet": {
        "name": "GatedDeltaNet",
        "kwargs_factory": lambda d_model, n_heads: {
            # Without gating: num_heads * head_dim = hidden_size
            "head_dim": int(d_model / n_heads),
            "use_gate": False,
            "allow_neg_eigval": False,
        },
    },
    "mamba2": {
        "name": "Mamba2",
        "kwargs_factory": lambda d_model, n_heads: {
            # Mamba2 uses default head_dim from the library
        },
    },
    "rwkv7": {
        "name": "RWKV7Attention",
        "kwargs_factory": lambda d_model, n_heads: {
            # RWKV7 uses default configuration
        },
    },
}


@dataclass(kw_only=True, eq=True)
class AttentionScalingModelConfigurator(Olmo3ModelConfigurator):
    """
    Model configurator that supports different attention types for scaling law experiments.

    Supports attention mechanisms:
    - full: Full softmax attention (disables OLMo3 sliding window)
    - sliding: Sliding window attention with configurable pattern
    - gated_deltanet: GatedDeltaNet linear RNN (with gating)
    - deltanet: DeltaNet linear RNN (without gating)
    - mamba2: Mamba2 SSM architecture
    - rwkv7: RWKV-7 attention
    """

    attention_type: str = "full"  # full, sliding, gated_deltanet, deltanet, mamba2, rwkv7
    window_size: int = 4096

    model_construction_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def configure_model(self, *, size_spec, sequence_length, tokenizer, device_type):
        # Get base config from parent
        config = super().configure_model(
            size_spec=size_spec,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            device_type=device_type,
        )

        if self.attention_type == "full":
            # Disable sliding window for true full attention
            config.block.attention.sliding_window = None
            log.info("Configured full softmax attention (no sliding window)")

        elif self.attention_type == "sliding":
            # Use sliding window pattern: [window_size, window_size, window_size, -1]
            # This means 3 layers with sliding window, then 1 layer with full attention
            config.block.attention.sliding_window = SlidingWindowAttentionConfig(
                pattern=[self.window_size] * 3 + [-1],
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=True,
            )
            log.info(f"Configured sliding window attention with window_size={self.window_size}")

        elif self.attention_type in LINEAR_ATTENTION_CONFIGS:
            # Linear attention variant (GatedDeltaNet, DeltaNet, Mamba2, RWKV7)
            linear_config = LINEAR_ATTENTION_CONFIGS[self.attention_type]
            n_heads = config.block.attention.n_heads

            # Get layer kwargs from the factory
            fla_kwargs = linear_config["kwargs_factory"](config.d_model, n_heads)

            config.block.name = TransformerBlockType.fla
            config.block.fla = FLAConfig(
                name=linear_config["name"],
                dtype=config.dtype,
                fla_layer_kwargs=fla_kwargs,
            )
            log.info(
                f"Configured {linear_config['name']} linear attention "
                f"(type={self.attention_type}, kwargs={fla_kwargs})"
            )

        else:
            valid_types = ["full", "sliding"] + list(LINEAR_ATTENTION_CONFIGS.keys())
            raise ValueError(
                f"Unknown attention type: {self.attention_type}. "
                f"Valid types: {valid_types}"
            )

        return config


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    """Add attention-type specific arguments to the parser."""
    valid_attention_types = ["full", "sliding"] + list(LINEAR_ATTENTION_CONFIGS.keys())
    parser.add_argument(
        "--attention-type",
        choices=valid_attention_types,
        default="full",
        help=(
            "Type of attention mechanism: "
            "full (softmax), sliding (window), "
            "gated_deltanet (GatedDeltaNet with gating), "
            "deltanet (DeltaNet without gating), "
            "mamba2 (Mamba2 SSM), "
            "rwkv7 (RWKV-7 attention)"
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4096,
        help="Sliding window size (only used for sliding attention type)",
    )


def configure_model(args: argparse.Namespace) -> TransformerModelConfigurator:
    """Create the model configurator from command line arguments."""
    return AttentionScalingModelConfigurator(
        rank_microbatch_size=None if args.rank_mbz is None else args.rank_mbz * args.sequence_length,
        attention_type=args.attention_type,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main(configure_model=configure_model, add_additional_args=add_additional_args)
