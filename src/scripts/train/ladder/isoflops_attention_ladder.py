import argparse
import logging
import os
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
    RunConfigurator,
    TransformerSize,
)
from olmo_core.nn.attention import (
    GateConfig,
    GateGranularity,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.fla import FLAConfig
from olmo_core.nn.hyper_connections import IdentityHyperConnectionConfig
from olmo_core.nn.transformer import TransformerBlockType
from olmo_core.optim import WSD, MuonConfig, Scheduler, SchedulerUnits
from olmo_core.optim.muon import MuonAdjustLRStrategy
from olmo_core.train import (
    Duration,
    prepare_training_environment,
    teardown_training_environment,
)

log = logging.getLogger(__name__)


CLUSTER_TO_GPU_TYPE = {
    "ai2/augusta": "NVIDIA H100 80GB HBM3",
    "ai2/jupiter": "NVIDIA H100 80GB HBM3",
    "ai2/titan": "NVIDIA B200",
}


def get_cluster_gpu_type(cluster: str) -> str:
    return CLUSTER_TO_GPU_TYPE.get(cluster, "NVIDIA H100 80GB HBM3")


def get_cluster_root_dir(cluster: str) -> str:
    del cluster
    return os.environ.get("LADDER_ROOT_DIR", "gs://ai2-llm")


def _get_model_flops_per_token_strict(model, sequence_length: int) -> int:
    try:
        return int(model.num_flops_per_token(sequence_length))
    except NotImplementedError as exc:
        raise OLMoConfigurationError(
            "FLOPs/token is not implemented for this model architecture at the requested sequence "
            "length. IsoFLOPs runs require explicit FLOPs accounting; fallback estimation is disabled."
        ) from exc


@dataclass
class KarpathyMuonScheduleCallback(callbacks.Callback):
    """
    Match nanochat-style Muon schedules:
    - momentum ramps from 0.85 -> 0.95 over first 300 steps
    - Muon weight decay decays linearly to 0 over training
    """

    enabled: bool = True
    momentum_start: float = 0.85
    momentum_end: float = 0.95
    momentum_warmup_steps: int = 300
    _muon_group_indices: list[int] = field(default_factory=list, init=False, repr=False)
    _initial_group_weight_decay: dict[int, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def pre_train(self):
        if not self.enabled:
            return

        optim = self.trainer.train_module.optim
        for idx, group in enumerate(optim.param_groups):
            if group.get("algorithm", "muon") == "muon":
                self._muon_group_indices.append(idx)
                self._initial_group_weight_decay[idx] = float(group.get("weight_decay", 0.0))

        if not self._muon_group_indices:
            self.enabled = False

    def pre_optim_step(self):
        if not self.enabled:
            return

        max_steps = self.trainer.max_steps
        if max_steps is None or max_steps <= 0:
            return

        # Trainer step starts at 1; nanochat schedule starts at 0.
        step0 = max(self.step - 1, 0)
        momentum_frac = min(step0 / self.momentum_warmup_steps, 1.0)
        muon_momentum = (
            self.momentum_start + (self.momentum_end - self.momentum_start) * momentum_frac
        )
        wd_scale = max(0.0, 1.0 - (step0 / max_steps))

        optim = self.trainer.train_module.optim
        for idx in self._muon_group_indices:
            group = optim.param_groups[idx]
            group["mu"] = muon_momentum
            group["weight_decay"] = self._initial_group_weight_decay[idx] * wd_scale

        self.trainer.record_metric("muon momentum", muon_momentum, namespace="optim")
        self.trainer.record_metric(
            "muon weight decay",
            optim.param_groups[self._muon_group_indices[0]]["weight_decay"],
            namespace="optim",
        )


@dataclass(kw_only=True)
class KarpathyIsoFlopsRunConfigurator(RunConfigurator):
    """
    Karpathy/nanochat-style fixed-compute run configurator:
    - fixed global batch size
    - training length chosen by target FLOPs
    - constant LR then 50% linear warmdown
    - Muon+Adam split parameter groups
    """

    target_flops: float
    target_batch_size: int = 524_288
    reference_batch_size: int = 524_288

    embedding_lr: float = 0.3
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    weight_decay: float = 0.2
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95

    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5

    _runtime_flops_per_token: int | None = field(default=None, init=False, repr=False)
    _runtime_d_model: int | None = field(default=None, init=False, repr=False)
    _runtime_n_layers: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.target_flops <= 0:
            raise OLMoConfigurationError("'target_flops' must be > 0")
        if self.target_batch_size <= 0:
            raise OLMoConfigurationError("'target_batch_size' must be > 0")
        if self.reference_batch_size <= 0:
            raise OLMoConfigurationError("'reference_batch_size' must be > 0")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise OLMoConfigurationError("'warmup_ratio' must be between 0 and 1")
        if not (0.0 <= self.warmdown_ratio <= 1.0):
            raise OLMoConfigurationError("'warmdown_ratio' must be between 0 and 1")

    def set_runtime_model_stats(self, *, flops_per_token: int, d_model: int, n_layers: int) -> None:
        self._runtime_flops_per_token = flops_per_token
        self._runtime_d_model = d_model
        self._runtime_n_layers = n_layers

    def _require_runtime_stats(self) -> tuple[int, int, int]:
        if (
            self._runtime_flops_per_token is None
            or self._runtime_d_model is None
            or self._runtime_n_layers is None
        ):
            raise RuntimeError(
                "Runtime model stats are unset. Call set_runtime_model_stats() before optimizer/scheduler config."
            )
        return self._runtime_flops_per_token, self._runtime_d_model, self._runtime_n_layers

    def configure_target_batch_size(self, num_params: int) -> int:
        del num_params
        return int(self.target_batch_size)

    def _configure_num_steps(self, batch_size: int) -> int:
        flops_per_token, _, _ = self._require_runtime_stats()
        return max(1, round(self.target_flops / (flops_per_token * batch_size)))

    def configure_duration(self, num_params: int, batch_size: int) -> Duration:
        del num_params
        num_steps = self._configure_num_steps(batch_size)
        return Duration.tokens(num_steps * batch_size)

    def configure_optimizer(self, num_params: int, batch_size: int) -> MuonConfig:
        del num_params
        _, d_model, n_layers = self._require_runtime_stats()

        batch_lr_scale = (batch_size / self.reference_batch_size) ** 0.5
        dmodel_lr_scale = (d_model / 768) ** -0.5
        matrix_lr = self.matrix_lr * batch_lr_scale
        embed_lr = self.embedding_lr * batch_lr_scale * dmodel_lr_scale
        lm_head_lr = self.unembedding_lr * batch_lr_scale * dmodel_lr_scale
        wd_scaled = self.weight_decay * (12 / n_layers) ** 2

        return MuonConfig(
            lr=matrix_lr,
            embed_lr=embed_lr,
            lm_head_lr=lm_head_lr,
            mu=0.95,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=wd_scaled,
            cautious_wd=True,
            nesterov=True,
            adjust_lr=MuonAdjustLRStrategy.rms_norm,
            flatten=True,
            use_triton=True,
        )

    def configure_lr_scheduler(self, num_params: int, batch_size: int) -> Scheduler:
        del num_params, batch_size
        return WSD(
            units=SchedulerUnits.tokens,
            warmup_fraction=self.warmup_ratio,
            decay_fraction=self.warmdown_ratio,
            decay_min_lr=0.0,
        )

    def configure_checkpoint_intervals(
        self, num_params: int, batch_size: int
    ) -> list[tuple[Duration, str]]:
        duration = self.configure_duration(num_params, batch_size)
        return [(duration, f"{self.target_flops:.2e} FLOPs")]

    def plot_lr_schedule(
        self,
        num_params: int,
        batch_size: int,
        *,
        show: bool = True,
        save_path: str | None = None,
    ) -> str | None:
        del num_params
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import pandas as pd  # type: ignore
        except ImportError:
            log.warning("matplotlib and pandas are required to plot LR schedule")
            return None

        optim = self.configure_optimizer(0, batch_size)
        scheduler = self.configure_lr_scheduler(0, batch_size)
        t_max = self.configure_duration(0, batch_size).value
        tokens_seen = 0
        tokens = []
        lrs = []
        while tokens_seen <= t_max:
            tokens_seen += batch_size
            lr = float(scheduler.get_lr(optim.lr, tokens_seen, t_max))
            tokens.append(tokens_seen)
            lrs.append(lr)

        df = pd.DataFrame({"tokens": tokens, "LR": lrs})
        df.plot(x="tokens", y="LR", legend=False, figsize=(12, 6))
        plt.grid(True)
        plt.title("Karpathy-style fixed-compute LR schedule")
        plt.tight_layout()

        if save_path is not None:
            save_path_obj = os.path.expanduser(save_path)
            save_dir = os.path.dirname(save_path_obj)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path_obj)

        if show:
            plt.show()

        return save_path


@dataclass(kw_only=True)
class IsoFlopsAttentionLadder(ModelLadder):
    wandb_entity: str | None = None
    eval_data_base_dir: str | None = None
    _num_params_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _model_stats_cache: dict[str, tuple[int, int, int, int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _get_mix_base_dir(self):
        if self.eval_data_base_dir is not None:
            return self.eval_data_base_dir
        return super()._get_mix_base_dir()

    def _get_model_runtime_stats(self, size_spec: str) -> tuple[int, int, int, int]:
        # Returns (num_non_embedding_params, flops_per_token, d_model, n_layers)
        if size_spec in self._model_stats_cache:
            return self._model_stats_cache[size_spec]

        model_config = self.get_model_config(size_spec)
        model = model_config.build(init_device="meta")
        num_params = model.num_non_embedding_params
        flops_per_token = _get_model_flops_per_token_strict(model, self.sequence_length)
        d_model = int(model.d_model)
        n_layers = int(model.n_layers)
        self._model_stats_cache[size_spec] = (
            int(num_params),
            int(flops_per_token),
            d_model,
            n_layers,
        )
        self._num_params_cache[size_spec] = num_params
        log.info(
            "Model stats for %s: non-embedding params=%s, FLOPs/token=%s, d_model=%d, n_layers=%d",
            size_spec,
            f"{num_params:,d}",
            f"{flops_per_token:,d}",
            d_model,
            n_layers,
        )
        return num_params, flops_per_token, d_model, n_layers

    def get_num_params(self, size_spec: str):
        if size_spec in self._num_params_cache:
            return self._num_params_cache[size_spec]
        num_params, _, _, _ = self._get_model_runtime_stats(size_spec)
        return num_params

    def _prepare_run_configurator(self, size_spec: str):
        num_params, flops_per_token, d_model, n_layers = self._get_model_runtime_stats(size_spec)
        run_configurator = typing.cast(KarpathyIsoFlopsRunConfigurator, self.run_configurator)
        run_configurator.set_runtime_model_stats(
            flops_per_token=flops_per_token,
            d_model=d_model,
            n_layers=n_layers,
        )
        return num_params

    def dry_run(self, size_spec: str, show_plot: bool = True, save_plot: str | None = None):
        self._prepare_run_configurator(size_spec)
        return super().dry_run(size_spec=size_spec, show_plot=show_plot, save_plot=save_plot)

    def run(self, size_spec: str, for_benchmarking: bool = False):
        if size_spec not in self.sizes:
            raise ValueError(f"Invalid size_spec '{size_spec}', must be one of {self.sizes}")

        prepare_training_environment(seed=self.seed, backend=self.backend)
        set_composable_seed(self.seed)

        model_config = self.get_model_config(size_spec)
        num_params = self._prepare_run_configurator(size_spec)

        (
            global_batch_size,
            rank_microbatch_size,
            requested_devices,
            _,
        ) = self._configure_batch_size_and_num_devices(size_spec, num_params)
        if requested_devices != dist_utils.get_world_size():
            raise OLMoConfigurationError(
                f"Requested {requested_devices} devices for model of size '{size_spec}', "
                f"but {dist_utils.get_world_size()} are available."
            )

        optim_config = self.run_configurator.configure_optimizer(num_params, global_batch_size)
        scheduler = self.run_configurator.configure_lr_scheduler(num_params, global_batch_size)

        trainer_config = self._configure_trainer(size_spec, for_benchmarking=for_benchmarking)

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

        train_module = self.model_configurator.build_train_module(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            model_config=model_config,
            optim_config=optim_config,
            scheduler=scheduler,
            device_type=self.device_type,
        )

        trainer = trainer_config.build(train_module, data_loader)

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

        trainer_config.callbacks["karpathy_muon_schedule"] = KarpathyMuonScheduleCallback(
            enabled=not for_benchmarking
        )

        if self.wandb_entity:
            trainer_config.callbacks["wandb"].entity = self.wandb_entity
        if trainer_config.callbacks["wandb"].project is None:
            trainer_config.callbacks["wandb"].project = self.project or "attn-scaling-ladder"

        return trainer_config


@dataclass(kw_only=True)
class AttentionModelConfigurator(Olmo3ModelConfigurator):
    attention_type: str = "sliding_gated"
    microbatch_discount: float = 1.0
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
    if attention_type == "hybrid_gated_deltanet":
        return dict(
            sliding_window=None,
            gate=GateConfig(granularity=GateGranularity.headwise),
        )
    raise ValueError(f"Unknown attention type: {attention_type}")


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    del cmd
    parser.add_argument(
        "--attention-type",
        type=str,
        default="sliding_gated",
        choices=["sliding_gated", "hybrid_gated_deltanet"],
        help="Attention intervention to apply.",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=1e19,
        help="Target training FLOPs budget for each run.",
    )
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=524_288,
        help="Fixed global batch size in tokens.",
    )
    parser.add_argument(
        "--reference-batch-size",
        type=int,
        default=524_288,
        help="Reference batch size used for sqrt LR scaling.",
    )
    parser.add_argument(
        "--embedding-lr",
        type=float,
        default=0.3,
        help="Embedding LR (Adam sub-groups).",
    )
    parser.add_argument(
        "--unembedding-lr",
        type=float,
        default=0.004,
        help="LM head LR (Adam sub-groups).",
    )
    parser.add_argument(
        "--matrix-lr",
        type=float,
        default=0.02,
        help="Matrix LR (Muon sub-groups).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.2,
        help="Base Muon weight decay before depth scaling.",
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.8,
        help="Adam beta1 for Adam sub-groups.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.95,
        help="Adam beta2 for Adam sub-groups.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup ratio for LR schedule.",
    )
    parser.add_argument(
        "--warmdown-ratio",
        type=float,
        default=0.5,
        help="Warmdown ratio for linear LR decay.",
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
        default=1024,
        help="Sliding-window size for the local layers when attention-type=sliding_gated.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="iu-cogai",
        help="Explicit W&B entity for logging.",
    )
    parser.add_argument(
        "--hyper-connections",
        type=int,
        default=0,
        help="Number of Identity HC streams (0=disabled, 4=typical).",
    )


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

    ladder = IsoFlopsAttentionLadder(
        name=f"{args.name}-{args.attention_type}",
        project=args.project or "attn-scaling-ladder",
        dir=str(
            io.join_path(
                get_cluster_root_dir(args.cluster),
                "model-ladders",
                f"{args.name}-{args.attention_type}",
            )
        ),
        wandb_entity=args.wandb_entity,
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_cluster_gpu_type(args.cluster),
        model_configurator=AttentionModelConfigurator(
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            attention_type=args.attention_type,
            microbatch_discount=args.microbatch_discount,
            force_min_world_size=int(os.environ.get("FORCE_MIN_WORLD_SIZE", "0")) or None,
            model_construction_kwargs=_make_model_construction_kwargs(
                args.attention_type,
                sliding_window_size=args.sliding_window_size,
            ),
            hyper_connections_n_streams=args.hyper_connections,
        ),
        run_configurator=KarpathyIsoFlopsRunConfigurator(
            target_flops=args.target_flops,
            target_batch_size=args.target_batch_size,
            reference_batch_size=args.reference_batch_size,
            embedding_lr=args.embedding_lr,
            unembedding_lr=args.unembedding_lr,
            matrix_lr=args.matrix_lr,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            warmup_ratio=args.warmup_ratio,
            warmdown_ratio=args.warmdown_ratio,
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8,
            instance_filter_config=InstanceFilterConfig(),
        ),
        eval_data_base_dir=args.eval_data_dir,
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder=configure_ladder, add_additional_args=add_additional_args)
