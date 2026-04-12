import argparse
import logging
import os
import re
import typing
from dataclasses import dataclass, field

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
import olmo_core.train.callbacks as callbacks
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.nn.attention import (
    GateConfig,
    GateGranularity,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.fla import FLAConfig
from olmo_core.nn.hyper_connections import IdentityHyperConnectionConfig
from olmo_core.nn.transformer import TransformerBlockType
from olmo_core.optim.muon import MuonAdjustLRStrategy, MuonConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment

# This ladder has been run under the names:
# "attn-scaling-ladder-sliding_gated", "attn-scaling-ladder-hybrid_gated_deltanet"

log = logging.getLogger(__name__)

# Keep cluster mappings local so Slurm/local runs do not depend on Beaker APIs.
CLUSTER_TO_GPU_TYPE = {
    "ai2/augusta": "NVIDIA H100 80GB HBM3",
    "ai2/jupiter": "NVIDIA H100 80GB HBM3",
    "ai2/titan": "NVIDIA B200",
}


def get_cluster_gpu_type(cluster: str) -> str:
    return CLUSTER_TO_GPU_TYPE.get(cluster, "NVIDIA H100 80GB HBM3")


def get_cluster_root_dir(cluster: str) -> str:
    del cluster
    # Override this in Slurm/local with e.g. LADDER_ROOT_DIR=/data/user/dicksonb/checkpoints
    return os.environ.get("LADDER_ROOT_DIR", "gs://ai2-llm")


def _ensure_attention_type_in_name(name: str, attention_type: str) -> str:
    if attention_type in name:
        return name
    return f"{name}-{attention_type}"


def _name_has_size_token(name: str, size_spec: str) -> bool:
    return re.search(rf"(^|[-_]){re.escape(size_spec)}($|[-_])", name) is not None


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    del cmd
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu:gloo,cuda:nccl",
        help="Distributed backend for `prepare_training_environment` (set `none` for CPU local run).",
    )
    parser.add_argument(
        "--attention-type",
        type=str,
        default="sliding_gated",
        choices=["sliding_gated", "hybrid_gated_deltanet", "vanilla_gated"],
        help="Attention intervention to apply.",
    )
    parser.add_argument(
        "--batch-size-multiplier",
        type=float,
        default=1.0,
        help="Multiplier to apply to the target WSDS batch size.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Optional training data glob (e.g. /data/user/dicksonb/data/nanochat/tokenized/*.npy).",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default=None,
        help="Optional base directory for v3_small_ppl_validation mix.",
    )
    parser.add_argument(
        "--microbatch-discount",
        type=float,
        default=1.0,
        help=(
            "Divide the auto-configured rank microbatch by this factor (OLMo3.1-hybrid style). "
            "Use >1.0 for lower-memory GPUs."
        ),
    )
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=4096,
        help="Sliding-window size for the local layers when attention-type=sliding_gated.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="iu-cogai",
        help="Explicit W&B entity for logging.",
    )
    parser.add_argument(
        "--no-grad-accum",
        action="store_true",
        default=False,
        help="Set rank microbatch size equal to the global batch size (no gradient accumulation).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="muon",
        choices=["muon", "skipstep_adamw"],
        help="Optimizer family to use for WSDS ladder runs.",
    )
    parser.add_argument(
        "--hyper-connections",
        type=int,
        default=0,
        help="Number of Identity HC streams (0=disabled, 4=typical).",
    )


@dataclass(kw_only=True)
class AttentionLadder(ModelLadder):
    wandb_entity: str | None = None
    eval_data_base_dir: str | None = None
    _num_params_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def _get_mix_base_dir(self):
        if self.eval_data_base_dir is not None:
            return self.eval_data_base_dir
        return super()._get_mix_base_dir()

    def get_num_params(self, size_spec: str):
        """
        Get non-embedding parameter count from an instantiated model, not config estimates.
        """
        if size_spec in self._num_params_cache:
            return self._num_params_cache[size_spec]

        model_config = self.get_model_config(size_spec)
        model = model_config.build(init_device="meta")
        num_params = model.num_non_embedding_params
        self._num_params_cache[size_spec] = num_params
        log.info(
            "Using built-model non-embedding param count for %s: %s",
            size_spec,
            f"{num_params:,d}",
        )
        return num_params

    def run(self, size_spec: str, for_benchmarking: bool = False):
        """
        Execute a particular model run of the experiment locally and store the results.

        Overrides base ladder behavior to ensure scheduling/optimizer math uses built-model
        parameter counts (important for FLA hybrid blocks where config-only counts undercount).
        """
        if size_spec not in self.sizes:
            raise ValueError(f"Invalid size_spec '{size_spec}', must be one of {self.sizes}")
        prepare_training_environment(seed=self.seed, backend=self.backend)
        set_composable_seed(self.seed)

        # Configure model.
        model_config = self.get_model_config(size_spec)
        num_params = self.get_num_params(size_spec)

        # Configure global batch size, make sure requested number of devices matches the number
        # of devices available.
        (
            global_batch_size,
            rank_microbatch_size,
            requested_devices,
            dp_world_size,
        ) = self._configure_batch_size_and_num_devices(size_spec, num_params)

        # With no_grad_accum, set rank microbatch so there's only 1 accum step.
        # Each rank processes global_batch_size / dp_world_size tokens per step.
        if (
            isinstance(self.model_configurator, AttentionModelConfigurator)
            and self.model_configurator.no_grad_accum
        ):
            rank_microbatch_size = global_batch_size // dp_world_size

        if requested_devices != dist_utils.get_world_size():
            raise OLMoConfigurationError(
                f"Requested {requested_devices} devices for model of size '{size_spec}', "
                f"but {dist_utils.get_world_size()} are available."
            )

        # Configure optimizer and scheduler.
        optim_config = self.run_configurator.configure_optimizer(num_params, global_batch_size)
        scheduler = self.run_configurator.configure_lr_scheduler(num_params, global_batch_size)

        # Configure trainer.
        trainer_config = self._configure_trainer(size_spec, for_benchmarking=for_benchmarking)
        # Build instance sources and data loader.
        instance_sources = [
            source.build(work_dir=self.work_dir) for source in self.instance_sources
        ]
        data_loader = self.data_loader.build(
            *instance_sources,
            work_dir=self.work_dir,
            global_batch_size=global_batch_size,
            tokenizer=self.tokenizer,
        )
        if data_loader.sequence_length != self.sequence_length:
            raise OLMoConfigurationError(
                f"Data loader sequence of {data_loader.sequence_length} does not match "
                f"configured sequence length of {self.sequence_length}."
            )

        # Build train module.
        train_module = self.model_configurator.build_train_module(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            model_config=model_config,
            optim_config=optim_config,
            scheduler=scheduler,
            device_type=self.device_type,
        )

        # Build trainer.
        trainer = trainer_config.build(train_module, data_loader)

        # Record all configs.
        config_dict = {
            "seed": self.seed,
            "size": str(size_spec),
            "model": model_config.as_config_dict(),
            "model_num_non_embedding_params": num_params,
            "optim": optim_config.as_config_dict(),
            "scheduler": scheduler.as_config_dict(),
            "data_loader": self.data_loader.as_config_dict(),
            "instance_sources": [s.as_config_dict() for s in self.instance_sources],
        }
        typing.cast(
            callbacks.ConfigSaverCallback, trainer.callbacks["config_saver"]
        ).config = config_dict

        # Train.
        trainer.fit()

        teardown_training_environment()

    def _configure_trainer(
        self,
        size_spec: str,
        for_benchmarking: bool = False,
    ):
        trainer_config = super()._configure_trainer(
            size_spec=size_spec,
            for_benchmarking=for_benchmarking,
        )
        if self.wandb_entity:
            trainer_config.callbacks["wandb"].entity = self.wandb_entity
        # If caller passed a fully-qualified run name containing size already (common for Slurm),
        # avoid appending size a second time in the W&B run name.
        if _name_has_size_token(self.name, str(size_spec)):
            trainer_config.callbacks["wandb"].name = self.name
        # Always pin the project to a fixed ladder-level project unless explicitly overridden.
        if trainer_config.callbacks["wandb"].project is None:
            trainer_config.callbacks["wandb"].project = self.project or "attn-scaling-ladder"
        wandb_callback = trainer_config.callbacks["wandb"]
        if dist_utils.get_rank() == 0 and wandb_callback.enabled:
            log.info(
                "W&B enabled for %s (project=%s, entity=%s)",
                size_spec,
                wandb_callback.project or self.name,
                wandb_callback.entity,
            )
        return trainer_config


@dataclass(kw_only=True)
class AttentionModelConfigurator(Olmo3ModelConfigurator):
    attention_type: str = "sliding_gated"
    microbatch_discount: float = 1.0
    no_grad_accum: bool = False
    force_min_world_size: int | None = None
    hyper_connections_n_streams: int = 0

    def configure_model(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        tokenizer: TokenizerConfig,
        device_type: str,
    ):
        model = super().configure_model(
            size_spec=size_spec,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            device_type=device_type,
        )

        if self.attention_type == "hybrid_gated_deltanet":
            head_dim = int(0.75 * model.d_model / model.block.sequence_mixer.n_heads)
            model.block.name = TransformerBlockType.fla_hybrid
            model.block.fla = FLAConfig(
                name="GatedDeltaNet",
                dtype=model.dtype,
                fla_layer_kwargs={"head_dim": head_dim, "use_gate": True, "allow_neg_eigval": True},
            )
            model.block.fla_hybrid_attention_indices = [
                i for i in range(model.n_layers) if i % 4 == 3
            ]

        if self.hyper_connections_n_streams > 0:
            model.hyper_connections = IdentityHyperConnectionConfig(
                n_streams=self.hyper_connections_n_streams,
            )

        return model

    def configure_rank_microbatch_size(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> int:
        if self.rank_microbatch_size is not None:
            return super().configure_rank_microbatch_size(
                size_spec=size_spec,
                sequence_length=sequence_length,
                device_type=device_type,
            )

        mbz = super().configure_rank_microbatch_size(
            size_spec=size_spec,
            sequence_length=sequence_length,
            device_type=device_type,
        )
        if self.microbatch_discount <= 0:
            raise ValueError("'microbatch_discount' must be > 0")

        # Mirror OLMo3.1-hybrid style: discount per-rank microbatch to save memory.
        discounted = int(mbz // self.microbatch_discount)
        discounted = (discounted // sequence_length) * sequence_length
        return max(sequence_length, discounted)

    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> DeviceMeshSpec:
        spec = super().configure_minimal_device_mesh_spec(
            size_spec=size_spec,
            sequence_length=sequence_length,
            device_type=device_type,
        )
        if self.force_min_world_size is None:
            return spec
        forced = max(1, int(self.force_min_world_size))
        return DeviceMeshSpec(world_size=forced, dp_world_size=forced)


@dataclass(kw_only=True)
class WSDSAttentionLadderRunConfigurator(WSDSChinchillaRunConfigurator):
    batch_size_multiplier: float = 1.0

    def configure_target_batch_size(self, num_params: int) -> int:
        bs = super().configure_target_batch_size(num_params)
        return int(bs * self.batch_size_multiplier)


class MuonWSDSAttentionLadderRunConfigurator(WSDSAttentionLadderRunConfigurator):
    def configure_optimizer(self, num_params: int, batch_size: int) -> MuonConfig:
        del batch_size  # unused
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        # but divide by 2 for WSD schedule (seems to work empirically).
        lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
        lr /= 2.0
        return MuonConfig(
            lr=lr, weight_decay=0.1, adjust_lr=MuonAdjustLRStrategy.rms_norm, use_triton=True
        )


def _make_model_construction_kwargs(
    attention_type: str,
    sliding_window_size: int,
) -> dict[str, object]:
    if attention_type == "sliding_gated":
        return dict(
            sliding_window=SlidingWindowAttentionConfig(
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=True,
                pattern=[sliding_window_size, sliding_window_size, sliding_window_size, -1],
            ),
            gate=GateConfig(granularity=GateGranularity.headwise),
        )
    elif attention_type == "hybrid_gated_deltanet":
        return dict(
            sliding_window=None,
            gate=GateConfig(granularity=GateGranularity.headwise),
        )
    elif attention_type == "vanilla_gated":
        return dict(
            sliding_window=None,
            gate=GateConfig(granularity=GateGranularity.headwise),
        )
    raise ValueError(f"Unknown attention type: {attention_type}")


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    if args.train_data is not None:
        train_paths = os.path.expanduser(args.train_data)
        if os.path.isdir(train_paths):
            train_paths = os.path.join(train_paths, "*.npy")
        train_sources = [
            NumpyDocumentSourceConfig(
                source_paths=[train_paths],
                tokenizer=tokenizer,
                expand_glob=True,
                source_group_size=1,
            )
        ]
    else:
        train_sources = [
            NumpyDocumentSourceMixConfig(
                tokenizer=tokenizer,
                mix=DataMix.OLMo_mix_0925,
                mix_base_dir="gs://ai2-llm/",
            )
        ]

    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=train_sources,
            sequence_length=args.sequence_length,
        )
    ]

    ladder_name = _ensure_attention_type_in_name(args.name, args.attention_type)
    ladder = AttentionLadder(
        name=ladder_name,
        project=args.project or "attn-scaling-ladder",
        dir=str(
            io.join_path(
                get_cluster_root_dir(args.cluster),
                "model-ladders",
                ladder_name,
            )
        ),
        wandb_entity=args.wandb_entity,
        sizes=[args.size],
        max_devices=args.max_gpus,
        device_type=get_cluster_gpu_type(args.cluster),
        model_configurator=AttentionModelConfigurator(
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            attention_type=args.attention_type,
            microbatch_discount=args.microbatch_discount,
            no_grad_accum=args.no_grad_accum,
            force_min_world_size=int(os.environ.get("FORCE_MIN_WORLD_SIZE", "0")) or None,
            model_construction_kwargs=_make_model_construction_kwargs(
                args.attention_type,
                sliding_window_size=args.sliding_window_size,
            ),
            hyper_connections_n_streams=args.hyper_connections,
        ),
        run_configurator=(
            MuonWSDSAttentionLadderRunConfigurator(
                chinchilla_multiple=args.chinchilla_multiple,
                lr_multiplier=args.lr_multiplier,
                stepped_schedule=args.stepped_schedule,
                batch_size_multiplier=args.batch_size_multiplier,
            )
            if args.optimizer == "muon"
            else WSDSAttentionLadderRunConfigurator(
                chinchilla_multiple=args.chinchilla_multiple,
                lr_multiplier=args.lr_multiplier,
                stepped_schedule=args.stepped_schedule,
                batch_size_multiplier=args.batch_size_multiplier,
            )
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8,
            instance_filter_config=InstanceFilterConfig(),
        ),
        eval_data_base_dir=args.eval_data_dir,
        backend=None if args.backend == "none" else args.backend,
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder=configure_ladder, add_additional_args=add_additional_args)
