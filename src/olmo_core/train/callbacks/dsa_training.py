"""
DSA (DeepSeek Sparse Attention) Training Callback.

Implements the two-stage training process for the Lightning Indexer:
1. Dense Warmup: Train only the indexer with dense attention patterns
2. Sparse Training: Train the full model with sparse attention

Based on the DeepSeek-V3.2 paper: https://arxiv.org/pdf/2512.02556
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .callback import Callback

log = logging.getLogger(__name__)


class DSATrainingStage(str, Enum):
    """Stages of DSA training."""

    dense_warmup = "dense_warmup"
    """Train only indexer with dense attention patterns."""

    sparse_training = "sparse_training"
    """Train full model with sparse attention."""


@dataclass
class DSATrainingConfig(Config):
    """
    Configuration for DSA two-stage training.

    The Lightning Indexer requires a two-stage training process:
    1. Dense Warmup: Initialize indexer to mimic dense attention patterns
    2. Sparse Training: Fine-tune full model with sparse attention

    All hyperparameters are configurable with DeepSeek defaults.
    """

    # Stage 1: Dense Warmup
    dense_warmup_steps: int = 1000
    """Number of steps for dense warmup phase (default: 1,000)."""

    dense_warmup_lr: float = 1e-3
    """Learning rate for dense warmup phase (default: 1e-3)."""

    # Stage 2: Sparse Training
    sparse_training_steps: int = 15000
    """Number of steps for sparse training phase (default: 15,000)."""

    sparse_training_lr: float = 7.3e-6
    """Learning rate for sparse training phase (default: 7.3e-6)."""

    # Alignment loss
    alignment_loss_weight: float = 1.0
    """Weight for the alignment loss term."""

    # Indexer configuration
    detach_indexer: bool = True
    """Whether to detach indexer from main model graph in sparse stage."""

    enabled: bool = True
    """Whether DSA training is enabled."""


@dataclass
class DSATrainingCallback(Callback):
    """
    Callback for two-stage DSA training.

    Stage 1 (Dense Warmup):
    - Freeze all parameters except Lightning Indexer
    - Use dense attention for training
    - Train indexer to predict dense attention patterns

    Stage 2 (Sparse Training):
    - Unfreeze all parameters
    - Use sparse attention with top-k selection
    - Continue training with alignment loss

    Example usage::

        config = DSATrainingConfig(
            dense_warmup_steps=1000,
            dense_warmup_lr=1e-3,
            sparse_training_steps=15000,
            sparse_training_lr=7.3e-6,
        )
        callback = DSATrainingCallback(config=config)
        trainer = Trainer(..., callbacks=[callback])
    """

    config: DSATrainingConfig

    _current_stage: DSATrainingStage = DSATrainingStage.dense_warmup
    _frozen_param_names: List[str] = None  # type: ignore
    _original_lrs: Dict[int, float] = None  # type: ignore

    def __post_init__(self):
        self._frozen_param_names = []
        self._original_lrs = {}

    @property
    def current_stage(self) -> DSATrainingStage:
        """Current training stage."""
        return self._current_stage

    def state_dict(self) -> Dict[str, Any]:
        """Save callback state for checkpointing."""
        return {
            "current_stage": self._current_stage.value,
            "frozen_param_names": self._frozen_param_names,
            "original_lrs": self._original_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load callback state from checkpoint."""
        self._current_stage = DSATrainingStage(state_dict["current_stage"])
        self._frozen_param_names = state_dict.get("frozen_param_names", [])
        self._original_lrs = state_dict.get("original_lrs", {})

    def pre_train(self):
        """Initialize DSA training - freeze non-indexer params and set dense mode."""
        if not self.config.enabled:
            return

        train_module = self.trainer.train_module

        # Validate that we have attention with indexer
        if not self._has_attention_with_indexer():
            raise OLMoConfigurationError(
                "DSATrainingCallback requires a model with attention modules that have a Lightning Indexer. "
                "Ensure your model uses AttentionConfig with an IndexerConfig."
            )

        if self._current_stage == DSATrainingStage.dense_warmup:
            log.info("DSA Training: Starting Dense Warmup stage")
            self._freeze_non_indexer_params()
            self._set_attention_mode(sparse=False)
            self._set_learning_rate(self.config.dense_warmup_lr)
        else:
            # Resuming from sparse training stage
            log.info("DSA Training: Resuming Sparse Training stage")
            self._set_attention_mode(sparse=True)

    def post_step(self):
        """Check for stage transitions after each step."""
        if not self.config.enabled:
            return

        step = self.step

        # Transition from dense warmup to sparse training
        if (
            self._current_stage == DSATrainingStage.dense_warmup
            and step >= self.config.dense_warmup_steps
        ):
            log.info(
                f"DSA Training: Transitioning to Sparse Training at step {step} "
                f"(after {self.config.dense_warmup_steps} warmup steps)"
            )
            self._transition_to_sparse_training()

    def _has_attention_with_indexer(self) -> bool:
        """Check if the model has attention modules with Lightning Indexer."""
        model = self._get_model()
        if model is None:
            return False

        # Look for MLAttention or Attention modules with indexer
        for module in model.modules():
            module_name = type(module).__name__
            if module_name in ("MLAttention", "Attention"):
                if hasattr(module, "indexer") and module.indexer is not None:
                    return True
        return False

    def _get_model(self) -> Optional[nn.Module]:
        """Get the underlying model from the train module."""
        train_module = self.trainer.train_module
        if hasattr(train_module, "model"):
            return train_module.model
        return None

    def _get_optimizer(self):
        """Get the optimizer from the train module."""
        train_module = self.trainer.train_module
        if hasattr(train_module, "optim"):
            return train_module.optim
        return None

    def _freeze_non_indexer_params(self):
        """Freeze all parameters except those in Lightning Indexer modules."""
        model = self._get_model()
        if model is None:
            return

        self._frozen_param_names = []

        for name, param in model.named_parameters():
            # Keep indexer parameters trainable
            if "indexer" in name:
                param.requires_grad = True
                log.debug(f"DSA Training: Keeping {name} trainable (indexer)")
            else:
                param.requires_grad = False
                self._frozen_param_names.append(name)

        frozen_count = len(self._frozen_param_names)
        total_count = sum(1 for _ in model.parameters())
        log.info(
            f"DSA Training: Froze {frozen_count}/{total_count} parameters "
            f"(keeping {total_count - frozen_count} indexer parameters trainable)"
        )

    def _unfreeze_all_params(self):
        """Unfreeze all parameters."""
        model = self._get_model()
        if model is None:
            return

        for name, param in model.named_parameters():
            param.requires_grad = True

        log.info(f"DSA Training: Unfroze all {sum(1 for _ in model.parameters())} parameters")
        self._frozen_param_names = []

    def _set_attention_mode(self, sparse: bool):
        """Set attention mode on all attention modules with indexer."""
        model = self._get_model()
        if model is None:
            return

        count = 0
        for module in model.modules():
            module_name = type(module).__name__
            if module_name in ("MLAttention", "Attention"):
                if hasattr(module, "use_sparse_attention"):
                    module.use_sparse_attention = sparse
                    count += 1
                    log.debug(
                        f"DSA Training: Set use_sparse_attention={sparse} on {module_name}"
                    )

        mode_str = "sparse" if sparse else "dense"
        log.info(f"DSA Training: Set attention mode to {mode_str} on {count} modules")

    def _set_learning_rate(self, lr: float):
        """Set learning rate for all parameter groups."""
        optim = self._get_optimizer()
        if optim is None:
            return

        # Store original LRs if not already stored
        if not self._original_lrs:
            for i, group in enumerate(optim.param_groups):
                self._original_lrs[i] = group["lr"]

        # Set new LR
        for group in optim.param_groups:
            group["lr"] = lr

        log.info(f"DSA Training: Set learning rate to {lr}")

    def _transition_to_sparse_training(self):
        """Transition from dense warmup to sparse training."""
        self._current_stage = DSATrainingStage.sparse_training

        # Unfreeze all parameters
        self._unfreeze_all_params()

        # Enable sparse attention
        self._set_attention_mode(sparse=True)

        # Update learning rate
        self._set_learning_rate(self.config.sparse_training_lr)

        log.info("DSA Training: Transition to Sparse Training complete")


def compute_dense_alignment_loss(
    indexer_scores: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """
    Compute alignment loss for dense warmup stage.

    The indexer should learn to predict the aggregate attention pattern
    across all heads.

    Args:
        indexer_scores: Raw indexer output, shape (batch, seq_q, seq_k)
        attention_weights: Softmax attention weights, shape (batch, n_heads, seq_q, seq_k)

    Returns:
        KL divergence loss between indexer distribution and target distribution
    """
    # Aggregate attention across heads and normalize (L1 normalization)
    target_dist = attention_weights.sum(dim=1)  # (batch, seq_q, seq_k)
    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)  # L1 normalize

    # Compute indexer distribution
    indexer_dist = F.softmax(indexer_scores, dim=-1)

    # KL divergence: KL(target || indexer)
    # Using log_target=True for numerical stability
    loss = F.kl_div(
        indexer_dist.log(),
        target_dist,
        reduction="batchmean",
        log_target=False,
    )
    return loss


def compute_sparse_alignment_loss(
    indexer_scores: Tensor,
    attention_weights: Tensor,
    top_k_indices: Tensor,
) -> Tensor:
    """
    Compute alignment loss for sparse training stage.

    Same as dense loss but restricted to selected top-k indices.

    Args:
        indexer_scores: Raw indexer output, shape (batch, seq_q, seq_k)
        attention_weights: Softmax attention weights, shape (batch, n_heads, seq_q, seq_k)
        top_k_indices: Selected token indices, shape (batch, seq_q, top_k)

    Returns:
        KL divergence loss over selected tokens only
    """
    # Aggregate and normalize target
    target_dist = attention_weights.sum(dim=1)  # (batch, seq_q, seq_k)
    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)

    # Gather only selected indices
    # target_selected: (batch, seq_q, top_k)
    target_selected = torch.gather(target_dist, -1, top_k_indices)
    target_selected = target_selected / target_selected.sum(dim=-1, keepdim=True)  # Re-normalize

    indexer_selected = torch.gather(indexer_scores, -1, top_k_indices)
    indexer_dist = F.softmax(indexer_selected, dim=-1)

    # KL divergence over selected tokens only
    loss = F.kl_div(
        indexer_dist.log(),
        target_selected,
        reduction="batchmean",
        log_target=False,
    )
    return loss
