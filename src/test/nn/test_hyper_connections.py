import pytest
import torch

from olmo_core.nn.hyper_connections import (
    HyperConnectionStream,
    IdentityHyperConnectionConfig,
)
from olmo_core.nn.residual_stream import ResidualStream


class TestHyperConnectionStream:
    @pytest.mark.parametrize("n_streams", [2, 4, 8])
    def test_merge_shape(self, n_streams: int):
        hcs = HyperConnectionStream(n_streams=n_streams)
        B, S, d = 2, 16, 64
        H = torch.randn(B, S, n_streams, d)
        out = hcs.merge(H)
        assert out.shape == (B, S, d)

    @pytest.mark.parametrize("n_streams", [2, 4, 8])
    def test_distribute_shape(self, n_streams: int):
        hcs = HyperConnectionStream(n_streams=n_streams)
        B, S, d = 2, 16, 64
        H = torch.randn(B, S, n_streams, d)
        x = torch.randn(B, S, d)
        out = hcs(H, x)
        assert out.shape == (B, S, n_streams, d)

    def test_identity_residual(self):
        """distribute(H, zeros) == H (identity residual)."""
        hcs = HyperConnectionStream(n_streams=4)
        B, S, d = 2, 16, 64
        H = torch.randn(B, S, 4, d)
        zeros = torch.zeros(B, S, d)
        out = hcs(H, zeros)
        torch.testing.assert_close(out, H)

    def test_broadcast_initialization(self):
        n = 4
        hcs = HyperConnectionStream(n_streams=n, init_strategy="broadcast")
        torch.testing.assert_close(hcs.h_pre, torch.full((n,), 1.0 / n))
        torch.testing.assert_close(hcs.h_post, torch.ones(n))

    def test_one_hot_initialization(self):
        n = 4
        hcs = HyperConnectionStream(n_streams=n, init_strategy="one_hot")
        expected_pre = torch.zeros(n)
        expected_pre[0] = 1.0
        expected_post = torch.zeros(n)
        expected_post[0] = 1.0
        torch.testing.assert_close(hcs.h_pre, expected_pre)
        torch.testing.assert_close(hcs.h_post, expected_post)

    def test_gradient_flow(self):
        """Gradients should reach h_pre and h_post."""
        hcs = HyperConnectionStream(n_streams=4)
        B, S, d = 2, 16, 64
        H = torch.randn(B, S, 4, d, requires_grad=True)

        merged = hcs.merge(H)
        out = hcs(H, merged)
        loss = out.sum()
        loss.backward()

        assert hcs.h_pre.grad is not None
        assert hcs.h_post.grad is not None
        assert hcs.h_pre.grad.abs().sum() > 0
        assert hcs.h_post.grad.abs().sum() > 0

    def test_broadcast_init_recovers_standard_residual(self):
        """With broadcast init and identical streams, merge(expand(x)) ≈ x."""
        n = 4
        hcs = HyperConnectionStream(n_streams=n, init_strategy="broadcast")
        B, S, d = 2, 16, 64
        x = torch.randn(B, S, d)

        # Expand: same vector in all n streams
        H = x.unsqueeze(-2).expand(B, S, n, d).contiguous()

        # Merge should recover x (since h_pre = 1/n and all streams are identical)
        merged = hcs.merge(H)
        torch.testing.assert_close(merged, x)


class TestResidualStreamMerge:
    def test_merge_identity(self):
        """ResidualStream.merge(x) == x (no-op)."""
        rs = ResidualStream()
        x = torch.randn(2, 16, 64)
        out = rs.merge(x)
        assert out is x


class TestIdentityHyperConnectionConfig:
    def test_defaults(self):
        cfg = IdentityHyperConnectionConfig()
        assert cfg.n_streams == 4
        assert cfg.init_strategy == "broadcast"

    def test_custom(self):
        cfg = IdentityHyperConnectionConfig(n_streams=8, init_strategy="one_hot")
        assert cfg.n_streams == 8
        assert cfg.init_strategy == "one_hot"
