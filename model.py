from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    # GPT-2's fixed vocabulary is reused, while every model weight is random.
    # block_size controls the number of learned absolute position embeddings.
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


class Block(nn.Module):
    """
    One GPT-2 transformer block.

    Architecture:
        LayerNorm -> Causal Attention -> Residual
        LayerNorm -> MLP -> Residual
    """

    def __init__(self, cfg):
        super().__init__()

        d = cfg.n_embd

        self.ln1 = nn.LayerNorm(d)

        # Combined query, key, value projection.
        self.qkv = nn.Linear(d, 3 * d)

        self.proj = nn.Linear(d, d)
        # Mark residual-branch output projections for smaller initialization;
        # their contributions accumulate across all transformer layers.
        self.proj.RESIDUAL_SCALE = True

        self.ln2 = nn.LayerNorm(d)

        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.fc2.RESIDUAL_SCALE = True

        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape

        # Pre-normalization before attention.
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=-1)

        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (
            a.view(B, T, self.n_head, C // self.n_head)
             .transpose(1, 2)
            for a in (q, k, v)
        )

        # Causal attention: token t cannot see future tokens. This is what makes
        # next-token training valid even though all positions run in parallel.
        # PyTorch SDPA applies dropout whenever dropout_p > 0, so explicitly use
        # zero in evaluation mode instead of relying only on model.eval().
        a = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        a = a.transpose(1, 2).contiguous().view(B, T, C)

        x = x + F.dropout(
            self.proj(a),
            p=self.dropout,
            training=self.training,
        )

        # GPT-2 feed-forward network expands width fourfold, applies GELU, and
        # projects back. Each position has the same MLP, with no token mixing.
        x = x + F.dropout(
            self.fc2(
                F.gelu(self.fc(self.ln2(x)), approximate="tanh")
            ),
            p=self.dropout,
            training=self.training,
        )

        return x


class GPT(nn.Module):
    """
    GPT-2 causal language model.

    No pretrained model weights are loaded.
    All parameters are initialized from scratch.
    """

    def __init__(self, cfg):
        super().__init__()

        if min(cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd) <= 0:
            raise ValueError("Model dimensions must be positive")
        if cfg.n_embd % cfg.n_head or not 0 <= cfg.dropout < 1:
            raise ValueError("Invalid heads or dropout")

        self.config = cfg

        # Token and learned absolute position embeddings.
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            Block(cfg) for _ in range(cfg.n_layer)
        ])

        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # Output distribution over the vocabulary.
        self.head = nn.Linear(
            cfg.n_embd,
            cfg.vocab_size,
            bias=False,
        )

        self.apply(self._init)

        # Tie input embeddings and output projection: the same parameter table
        # both represents incoming tokens and scores outgoing vocabulary IDs.
        self.head.weight = self.wte.weight

    def _init(self, m):
        if isinstance(m, (nn.Embedding, nn.Linear)):
            std = 0.02

            if getattr(m, "RESIDUAL_SCALE", False):
                std /= math.sqrt(2 * self.config.n_layer)

            nn.init.normal_(m.weight, mean=0.0, std=std)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, targets=None):
        """
        ids:     (B, T)
        targets: (B, T), already shifted by one token.

        Returns:
            logits: (B, T, vocab_size)
            loss: next-token cross-entropy, if targets provided
        """

        B, T = ids.shape

        if T > self.config.block_size:
            raise ValueError("Input exceeds model context length")

        # A prompt always begins at position zero. Matched-context extraction
        # restores a passage's position by prepending its ORIGINAL block context.
        pos = torch.arange(T, device=ids.device)

        x = self.drop(
            self.wte(ids) + self.wpe(pos)
        )

        for block in self.blocks:
            x = block(x)

        logits = self.head(self.ln_f(x))

        loss = None

        if targets is not None:
            # targets already contains the next token at every input position.
            # Example: ids=[a,b,c] and targets=[b,c,d]. There is no second shift.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.inference_mode()
    def generate_greedy(self, prefix, new_tokens):
        """
        Generate tokens using greedy decoding.

        At each step:
            next_token = argmax P(token | prefix)

        No sampling, temperature, or top-k filtering.

        This is used for verbatim extraction evaluation.
        """

        self.eval()
        # Strict full-context evaluation: reject overflow instead of truncating
        # the prompt and silently resetting learned absolute positions.
        if prefix.ndim != 2 or prefix.shape[1] < 1 or new_tokens < 0:
            raise ValueError("Invalid prefix or generation length")
        if prefix.shape[1] + new_tokens > self.config.block_size:
            raise ValueError("Prefix plus continuation exceeds context; sliding evaluation is disabled")

        output = prefix.clone()

        for _ in range(new_tokens):
            context = output

            logits, _ = self(context)

            # Distribution for the next token.
            next_logits = logits[:, -1, :]

            # Choose the most probable token. Keep going even if it is EOS:
            # exact-length extraction must compare the entire requested suffix.
            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True,
            )

            output = torch.cat(
                (output, next_token),
                dim=1,
            )

        return output


SIZES = {
    "micro": (2, 2, 64),
    "tiny": (6, 6, 384),
    "small": (12, 12, 768),
}


def make_model(size="small", block_size=512):
    """
    tiny:  6 layers, 6 heads, embedding dimension 384
    small: GPT-2 small architecture, 12 layers,
           12 heads, embedding dimension 768
    """

    n_layer, n_head, n_embd = SIZES[size]

    cfg = GPTConfig(
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    return GPT(cfg)
