import pytest
import torch

from olmo_core.nn.hyper_connections import (
    HyperConnectionStream,
    IdentityHyperConnectionConfig,
)
from olmo_core.nn.residual_stream import ResidualStream


D_MODEL = 64


class TestHyperConnectionStream:
    @pytest.mark.parametrize("n_streams", [2, 4, 8])
    def test_merge_shape(self, n_streams: int):
        hcs = HyperConnectionStream(n_streams=n_streams, d_model=D_MODEL)
        B, S = 2, 16
        H = torch.randn(B, S, n_streams, D_MODEL)
        out = hcs.merge(H)
        assert out.shape == (B, S, D_MODEL)

    @pytest.mark.parametrize("n_streams", [2, 4, 8])
    def test_distribute_shape(self, n_streams: int):
        hcs = HyperConnectionStream(n_streams=n_streams, d_model=D_MODEL)
        B, S = 2, 16
        H = torch.randn(B, S, n_streams, D_MODEL)
        x = torch.randn(B, S, D_MODEL)
        out = hcs(H, x)
        assert out.shape == (B, S, n_streams, D_MODEL)

    def test_identity_residual(self):
        """distribute(H, zeros) ≈ H (identity residual) at init."""
        hcs = HyperConnectionStream(n_streams=4, d_model=D_MODEL)
        B, S = 2, 16
        H = torch.randn(B, S, 4, D_MODEL)
        zeros = torch.zeros(B, S, D_MODEL)
        out = hcs(H, zeros)
        torch.testing.assert_close(out, H)

    def test_gradient_flow(self):
        """Gradients should reach phi, bias, and alpha parameters."""
        hcs = HyperConnectionStream(n_streams=4, d_model=D_MODEL)
        B, S = 2, 16
        H = torch.randn(B, S, 4, D_MODEL, requires_grad=True)

        merged = hcs.merge(H)
        out = hcs(H, merged)
        loss = out.sum()
        loss.backward()

        assert hcs.phi.weight.grad is not None
        assert hcs.bias.grad is not None
        assert hcs.alpha_pre.grad is not None
        assert hcs.alpha_post.grad is not None

    def test_round_robin_bias_init(self):
        """At init, h_pre bias gives round-robin one-hot pattern via sigmoid."""
        n = 4
        for sublayer_idx in range(8):
            hcs = HyperConnectionStream(n_streams=n, d_model=D_MODEL, sublayer_idx=sublayer_idx)
            stream = sublayer_idx % n

            # h_pre bias: active stream has +5, others have -5
            assert hcs.bias[stream].item() == pytest.approx(5.0)
            for i in range(n):
                if i != stream:
                    assert hcs.bias[i].item() == pytest.approx(-5.0)

            # h_post bias: all zeros → 2*sigmoid(0) = 1.0
            for i in range(n, 2 * n):
                assert hcs.bias[i].item() == pytest.approx(0.0)

    def test_round_robin_merge_at_init(self):
        """At init (alpha ≈ 0, phi.weight = 0), merge approximately selects one stream."""
        n = 4
        B, S = 2, 16
        H = torch.randn(B, S, n, D_MODEL)

        for sublayer_idx in range(n):
            hcs = HyperConnectionStream(n_streams=n, d_model=D_MODEL, sublayer_idx=sublayer_idx)
            merged = hcs.merge(H)
            expected = H[..., sublayer_idx, :]
            # sigmoid(5) ≈ 0.993, sigmoid(-5) ≈ 0.007, so approximately selects one stream
            torch.testing.assert_close(merged, expected, atol=0.05, rtol=0.05)

    def test_h_post_bounded(self):
        """h_post should be bounded in [0, 2] due to 2*sigmoid."""
        hcs = HyperConnectionStream(n_streams=4, d_model=D_MODEL)
        H = torch.randn(2, 16, 4, D_MODEL)
        _, h_post = hcs._get_weights(H)
        assert (h_post >= 0).all()
        assert (h_post <= 2).all()

    def test_h_pre_bounded(self):
        """h_pre should be bounded in [0, 1] due to sigmoid."""
        hcs = HyperConnectionStream(n_streams=4, d_model=D_MODEL)
        H = torch.randn(2, 16, 4, D_MODEL)
        h_pre, _ = hcs._get_weights(H)
        assert (h_pre >= 0).all()
        assert (h_pre <= 1).all()

    def test_phi_weight_zero_init(self):
        """phi.weight should be initialized to zero."""
        hcs = HyperConnectionStream(n_streams=4, d_model=D_MODEL)
        assert (hcs.phi.weight == 0).all()

    def test_alpha_small_init(self):
        """alpha_pre and alpha_post should start at 0.01."""
        hcs = HyperConnectionStream(n_streams=4, d_model=D_MODEL)
        assert hcs.alpha_pre.item() == pytest.approx(0.01)
        assert hcs.alpha_post.item() == pytest.approx(0.01)

    def test_init_recovers_standard_residual(self):
        """At init with identical streams, merge+distribute ≈ standard residual."""
        n = 4
        hcs = HyperConnectionStream(n_streams=n, d_model=D_MODEL, sublayer_idx=0)
        B, S = 2, 16
        x = torch.randn(B, S, D_MODEL)

        # Expand: same vector in all n streams
        H = x.unsqueeze(-2).expand(B, S, n, D_MODEL).contiguous()

        # At init, merge selects stream 0 ≈ x
        merged = hcs.merge(H)
        torch.testing.assert_close(merged, x, atol=0.05, rtol=0.05)

        # sublayer output
        F_x = torch.randn(B, S, D_MODEL)
        out = hcs(H, F_x)

        # Each stream should be ≈ x + 1.0 * F_x (h_post ≈ 1 at init)
        for i in range(n):
            torch.testing.assert_close(out[..., i, :], x + F_x, atol=0.05, rtol=0.05)


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

    def test_custom(self):
        cfg = IdentityHyperConnectionConfig(n_streams=8)
        assert cfg.n_streams == 8
