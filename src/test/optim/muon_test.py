from collections import Counter

import pytest
import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh
from olmo_core.nn.hyper_connections import IdentityHyperConnectionConfig
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim.muon import MuonConfig, _patch_dion_adamw_update_foreach_hc_split
from olmo_core.testing import DEVICES, requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import requires_dion
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all


def build_transformer_model() -> Transformer:
    config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    model = config.build()
    return model


@requires_dion
def test_muon_config_to_optim():
    from dion import Muon  # type: ignore[reportMissingImports]

    config = MuonConfig()

    model = build_transformer_model()
    optim = config.build(model)

    assert isinstance(optim, Muon)
    assert len(optim.param_groups) == 4  # emb, matrix, vector, lm_head

    assert config.merge(["lr=1e-1"]).lr == 0.1


@requires_dion
def test_muon_patch_only_enabled_for_hc(monkeypatch):
    patch_calls: list[set[int]] = []

    def _count_patch_calls(hc_vector_sizes: set[int]) -> None:
        patch_calls.append(set(hc_vector_sizes))

    monkeypatch.setattr(
        "olmo_core.optim.muon._patch_dion_adamw_update_foreach_hc_split",
        _count_patch_calls,
    )

    # No HC => no Dion patch call.
    model = build_transformer_model()
    _ = MuonConfig().build(model)
    assert patch_calls == []

    # HC enabled => patch is installed with model n_streams size.
    hc_model_config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    hc_model_config.hyper_connections = IdentityHyperConnectionConfig(n_streams=4)
    hc_model = hc_model_config.build()
    _ = MuonConfig().build(hc_model)
    assert patch_calls == [{4}]

    hc8_model_config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    hc8_model_config.hyper_connections = IdentityHyperConnectionConfig(n_streams=8)
    hc8_model = hc8_model_config.build()
    _ = MuonConfig().build(hc8_model)
    assert patch_calls == [{4}, {8}]


@requires_dion
def test_muon_dion_adamw_foreach_hc_split_dispatch(monkeypatch):
    from dion import muon as dion_muon  # type: ignore[reportMissingImports]
    from dion import scalar_opts  # type: ignore[reportMissingImports]

    calls = {"compiled": [], "eager": []}

    def _compiled_foreach(
        X,
        G,
        M,
        V,
        lr,
        beta1,
        beta2,
        weight_decay,
        step,
        epsilon,
        cautious_wd=False,
    ):
        del G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd
        calls["compiled"].append([tuple(t.shape) for t in X])

    def _eager_foreach(
        X,
        G,
        M,
        V,
        lr,
        beta1,
        beta2,
        weight_decay,
        step,
        epsilon,
        cautious_wd=False,
    ):
        del G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd
        calls["eager"].append([tuple(t.shape) for t in X])

    _compiled_foreach._torchdynamo_orig_callable = _eager_foreach  # type: ignore[attr-defined]
    monkeypatch.setattr(scalar_opts, "adamw_update_foreach", _compiled_foreach)
    monkeypatch.delattr(scalar_opts, "_olmo_core_hc_foreach_split_installed", raising=False)

    _patch_dion_adamw_update_foreach_hc_split({4})
    wrapped_foreach_once = scalar_opts.adamw_update_foreach
    wrapped_async_once = dion_muon.adamw_update_foreach_async
    _patch_dion_adamw_update_foreach_hc_split({4})
    assert scalar_opts.adamw_update_foreach is wrapped_foreach_once
    assert scalar_opts.adamw_update_foreach_async is wrapped_async_once
    assert dion_muon.adamw_update_foreach_async is wrapped_async_once

    params = [
        torch.randn(4),  # HC-like: should go eager
        torch.randn(8),  # non-HC: should stay compiled
        torch.randn(2, 2),  # numel=4 but not vector, should stay compiled
        torch.randn(4),  # HC-like: should go eager
    ]
    grads = [torch.randn_like(p) for p in params]
    momentums = [torch.zeros_like(p) for p in params]
    variances = [torch.zeros_like(p) for p in params]

    list(
        dion_muon.adamw_update_foreach_async(
            params,
            grads,
            momentums,
            variances,
            torch.tensor(1e-3),
            torch.tensor(0.9),
            torch.tensor(0.95),
            torch.tensor(0.1),
            torch.tensor(1),
            torch.tensor(1e-8),
            False,
        )
    )

    compiled_shapes = [shape for call_shapes in calls["compiled"] for shape in call_shapes]
    eager_shapes = [shape for call_shapes in calls["eager"] for shape in call_shapes]
    assert Counter(compiled_shapes) == Counter({(8,): 1, (2, 2): 1})
    assert Counter(eager_shapes) == Counter({(4,): 2})


@requires_dion
def test_muon_dion_adamw_foreach_hc_split_fallback_only_for_failing_signature(monkeypatch):
    from dion import muon as dion_muon  # type: ignore[reportMissingImports]
    from dion import scalar_opts  # type: ignore[reportMissingImports]

    calls = {"compiled": [], "eager": []}

    def _compiled_foreach(
        X,
        G,
        M,
        V,
        lr,
        beta1,
        beta2,
        weight_decay,
        step,
        epsilon,
        cautious_wd=False,
    ):
        del G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd
        shapes = [tuple(t.shape) for t in X]
        calls["compiled"].append(shapes)
        if shapes and shapes[0] == (8,):
            raise RuntimeError("InductorError: AssertionError in fuse_nodes SchedulerNode")

    def _eager_foreach(
        X,
        G,
        M,
        V,
        lr,
        beta1,
        beta2,
        weight_decay,
        step,
        epsilon,
        cautious_wd=False,
    ):
        del G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd
        calls["eager"].append([tuple(t.shape) for t in X])

    _compiled_foreach._torchdynamo_orig_callable = _eager_foreach  # type: ignore[attr-defined]
    monkeypatch.setattr(scalar_opts, "adamw_update_foreach", _compiled_foreach)
    monkeypatch.delattr(scalar_opts, "_olmo_core_hc_foreach_split_installed", raising=False)

    _patch_dion_adamw_update_foreach_hc_split({4})

    def _step_once() -> None:
        params = [torch.randn(8), torch.randn(16)]
        grads = [torch.randn_like(p) for p in params]
        momentums = [torch.zeros_like(p) for p in params]
        variances = [torch.zeros_like(p) for p in params]
        list(
            dion_muon.adamw_update_foreach_async(
                params,
                grads,
                momentums,
                variances,
                torch.tensor(1e-3),
                torch.tensor(0.9),
                torch.tensor(0.95),
                torch.tensor(0.1),
                torch.tensor(1),
                torch.tensor(1e-8),
                False,
            )
        )

    _step_once()
    _step_once()

    compiled_shapes = [shape for call_shapes in calls["compiled"] for shape in call_shapes]
    eager_shapes = [shape for call_shapes in calls["eager"] for shape in call_shapes]
    assert Counter(compiled_shapes) == Counter({(8,): 1, (16,): 2})
    assert Counter(eager_shapes) == Counter({(8,): 2})


@requires_dion
@pytest.mark.parametrize("device", DEVICES)
def test_muon(device: torch.device, tmp_path):
    config = MuonConfig()

    model = build_transformer_model().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()

    # Test that initial_lr is a "fixed field" that gets reset on checkpoint load.
    # Corrupt initial_lr, save, then load—initial_lr should be restored to original, not loaded from checkpoint.
    original_initial_lrs = [group["initial_lr"] for group in optim.param_groups]
    for group in optim.param_groups:
        group["initial_lr"] = 1e-8
    save_model_and_optim_state(tmp_path, model, optim)
    load_model_and_optim_state(tmp_path, model, optim)
    for group, original_lr in zip(optim.param_groups, original_initial_lrs):
        assert group["initial_lr"] == original_lr


def _run_hsdp_muon(shard_degree: int, num_replicas: int):
    device = get_default_device()

    # HSDP Transformer
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp, shard_degree=shard_degree, num_replicas=num_replicas
    )
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)
    config = TransformerConfig.olmo2_30M(vocab_size=1024)
    model = config.build(init_device=device.type)
    model.train()
    model = parallelize_model(model, world_mesh=world_mesh, device=device, dp_config=dp_config)

    # Create the Muon optimizer
    optim_config = MuonConfig()
    optim = optim_config.create_optimizer(model)

    # Fwd-bwd
    bs, seq_len = 2, 8
    input_ids = torch.randint(0, 1024, (bs, seq_len), device=device)
    logits = model(input_ids)
    logits.sum().backward()

    # Take optimizer step to test Muon with HSDP
    optim.step()


@requires_dion
@requires_multi_gpu
@pytest.mark.parametrize(
    "shard_degree,num_replicas",
    [
        pytest.param(2, 1, id="shard2_replica1"),
        pytest.param(1, 2, id="shard1_replica2"),
    ],
)
def test_hsdp_muon(shard_degree: int, num_replicas: int):
    seed_all(0)
    run_distributed_test(
        _run_hsdp_muon,
        backend="nccl",
        start_method="spawn",
        world_size=2,
        func_args=(shard_degree, num_replicas),
    )


def _run_fsdp_muon():
    device = get_default_device()

    # FSDP Transformer
    dp_config = TransformerDataParallelConfig(name=DataParallelType.fsdp)
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)
    config = TransformerConfig.olmo2_30M(vocab_size=1024)
    model = config.build(init_device=device.type)
    model.train()
    model = parallelize_model(model, world_mesh=world_mesh, device=device, dp_config=dp_config)

    # Create the Muon optimizer
    optim_config = MuonConfig()
    optim = optim_config.create_optimizer(model)

    # Fwd-bwd
    bs, seq_len = 2, 8
    input_ids = torch.randint(0, 1024, (bs, seq_len), device=device)
    logits = model(input_ids)
    logits.sum().backward()

    # Take optimizer step to test Muon with FSDP
    optim.step()


@requires_dion
@requires_multi_gpu
def test_fsdp_muon():
    seed_all(0)
    run_distributed_test(
        _run_fsdp_muon,
        backend="nccl",
        start_method="spawn",
        world_size=2,
    )
