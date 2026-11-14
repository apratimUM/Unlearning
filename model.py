
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """
    GPT-2 model configuration.

    The GPT-2 tokenizer and vocabulary are reused.
    All model weights are initialized randomly.
    """

    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


class Block(nn.Module):
    """
    One GPT-2 transformer block.

    Architecture:
        LayerNorm -> Causal Self-Attention -> Residual
        LayerNorm -> MLP -> Residual
    """

    def __init__(self, cfg):
        super().__init__()

        d = cfg.n_embd

        self.ln1 = nn.LayerNorm(d)

        # Combined query/key/value projection.
        self.qkv = nn.Linear(d, 3 * d)

        # Attention output projection.
        self.proj = nn.Linear(d, d)

        # Scale residual-output initialization by layer count.
        self.proj.RESIDUAL_SCALE = True

        self.ln2 = nn.LayerNorm(d)

        # GPT-2 MLP expands to 4x the embedding dimension.
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

        self.fc2.RESIDUAL_SCALE = True

        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape

        # ----------------------------------------------------
        # CAUSAL SELF-ATTENTION
        # ----------------------------------------------------

        q, k, v = self.qkv(
            self.ln1(x)
        ).split(C, dim=-1)

        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (
            a.view(
                B,
                T,
                self.n_head,
                C // self.n_head,
            ).transpose(1, 2)
            for a in (q, k, v)
        )

        # PyTorch SDPA uses available CUDA attention kernels.
        # Explicitly disable attention dropout in eval mode.
        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=(
                self.dropout if self.training else 0.0
            ),
        )

        a = (
            a.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        # Attention residual connection.
        x = x + F.dropout(
            self.proj(a),
            p=self.dropout,
            training=self.training,
        )

        # ----------------------------------------------------
        # FEED-FORWARD NETWORK
        # ----------------------------------------------------

        mlp = self.fc(
            self.ln2(x)
        )

        mlp = F.gelu(
            mlp,
            approximate="tanh",
        )

        mlp = self.fc2(mlp)

        x = x + F.dropout(
            mlp,
            p=self.dropout,
            training=self.training,
        )

        return x


class GPT(nn.Module):
    """
    GPT-2 causal language model trained from scratch.

    No pretrained model weights are loaded.

    Input:
        ids: (B, T)

    Optional targets:
        targets: (B, T), already shifted by one token.
    """

    def __init__(self, cfg):
        super().__init__()

        if min(
            cfg.vocab_size,
            cfg.block_size,
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd,
        ) <= 0:
            raise ValueError(
                "Model dimensions must be positive"
            )

        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError(
                "Embedding dimension must be divisible "
                "by the number of attention heads"
            )

        if not 0 <= cfg.dropout < 1:
            raise ValueError(
                "Dropout must be in [0, 1)"
            )

        self.config = cfg

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        # Token embeddings.
        self.wte = nn.Embedding(
            cfg.vocab_size,
            cfg.n_embd,
        )

        # Learned absolute positional embeddings.
        self.wpe = nn.Embedding(
            cfg.block_size,
            cfg.n_embd,
        )

        self.drop = nn.Dropout(
            cfg.dropout
        )

        # ----------------------------------------------------
        # TRANSFORMER BLOCKS
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([
            Block(cfg)
            for _ in range(cfg.n_layer)
        ])

        self.ln_f = nn.LayerNorm(
            cfg.n_embd
        )

        # ----------------------------------------------------
        # LANGUAGE MODEL HEAD
        # ----------------------------------------------------

        self.head = nn.Linear(
            cfg.n_embd,
            cfg.vocab_size,
            bias=False,
        )

        # Initialize all weights from scratch.
        self.apply(self._init)

        # Tie output projection to input token embeddings.
        self.head.weight = self.wte.weight

    def _init(self, module):
        """GPT-2-style random weight initialization."""

        if isinstance(
            module,
            (nn.Embedding, nn.Linear),
        ):
            std = 0.02

            if getattr(
                module,
                "RESIDUAL_SCALE",
                False,
            ):
                std /= math.sqrt(
                    2 * self.config.n_layer
                )

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )

            if (
                isinstance(module, nn.Linear)
                and module.bias is not None
            ):
                nn.init.zeros_(
                    module.bias
                )

    def forward(self, ids, targets=None):
        """
        Forward pass for causal next-token prediction.

        ids:
            (B, T) integer token IDs.

        targets:
            (B, T) integer token IDs,
            already shifted by one token.

        Returns:
            logits: (B, T, vocab_size)
            loss: cross-entropy or None
        """

        if ids.ndim != 2:
            raise ValueError(
                "Expected ids with shape (B, T)"
            )

        B, T = ids.shape

        if T < 1 or T > self.config.block_size:
            raise ValueError(
                "Input length must be between 1 "
                "and the configured context length"
            )

        if targets is not None:
            if targets.shape != ids.shape:
                raise ValueError(
                    "Targets must match input shape"
                )

        # ----------------------------------------------------
        # TOKEN + POSITION EMBEDDINGS
        # ----------------------------------------------------

        positions = torch.arange(
            T,
            device=ids.device,
        )

        x = self.drop(
            self.wte(ids)
            + self.wpe(positions)
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # ----------------------------------------------------
        # VOCABULARY LOGITS
        # ----------------------------------------------------

        logits = self.head(x)

        # ----------------------------------------------------
        # NEXT-TOKEN LOSS
        # ----------------------------------------------------

        loss = None

        if targets is not None:
            # Targets are already shifted by data.py.
            # No additional shift must be applied here.
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.inference_mode()
    def generate_greedy(
        self,
        prefix,
        new_tokens,
    ):
        """
        Greedy autoregressive generation.

        Uses argmax for every next-token prediction.

        No temperature, sampling, or top-k.

        Preserves the model's original train/eval mode.

        Rejects context overflow instead of truncating
        or resetting learned positional embeddings.

        Continues through EOS when extracting fixed-length
        memorized continuations.
        """

        if prefix.ndim != 2:
            raise ValueError(
                "Prefix must have shape (B, T)"
            )

        if prefix.shape[1] < 1:
            raise ValueError(
                "Prefix cannot be empty"
            )

        if new_tokens < 0:
            raise ValueError(
                "new_tokens must be nonnegative"
            )

        if (
            prefix.shape[1] + new_tokens
            > self.config.block_size
        ):
            raise ValueError(
                "Prefix plus continuation exceeds "
                "the model context length"
            )

        # Preserve training mode, including dropout state.
        original_mode = self.training

        try:
            self.eval()

            output = prefix.clone()

            for _ in range(new_tokens):

                # Predict next-token distribution.
                logits, _ = self(output)

                next_token = torch.argmax(
                    logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                )

                output = torch.cat(
                    (output, next_token),
                    dim=1,
                )

            return output

        finally:
            self.train(original_mode)


# ============================================================
# MODEL SIZE PRESETS
# ============================================================

SIZES = {
    "micro": (2, 2, 64),
    "tiny": (6, 6, 384),
    "small": (12, 12, 768),
}


def make_model(
    size="small",
    block_size=1024,
):
    """
    Construct a randomly initialized GPT model.

    micro:
        2 layers, 2 heads, 64 embedding dimensions.

    tiny:
        6 layers, 6 heads, 384 embedding dimensions.

    small:
        GPT-2 Small:
        12 layers, 12 heads, 768 dimensions.

    The training pipeline uses "small" with
    block_size=1024.
    """

    if size not in SIZES:
        raise ValueError(
            f"Unknown model size: {size!r}. "
            f"Available: {sorted(SIZES)}"
        )

    n_layer, n_head, n_embd = SIZES[size]

    cfg = GPTConfig(
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    return GPT(cfg)
