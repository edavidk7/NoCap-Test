import glob
import math
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch._inductor import config
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def inv_rms(x, eps=1e-6):
    return torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).type_as(x)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.learn_temperature = config.learn_temperature
        self.head_gate = config.head_gate
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        if self.learn_temperature:
            self.log_temperature = nn.Parameter(torch.zeros(self.n_head))  # default temperature to 1.0
        self.log_temperature_scale = config.temperature_scale
        self.gqa = config.gqa
        self.temperature_ones = torch.ones(2 * self.n_embd)
        if self.head_gate:
            self.gate_proj = nn.Linear(self.n_embd, self.n_head)

    def _get_learned_temperatures(self) -> torch.Tensor:
        return torch.exp(self.log_temperature_scale * self.log_temperature.tanh())

    def _c_attn_temperatures(self, x) -> torch.Tensor:
        temps = self._get_learned_temperatures().unsqueeze(-1).expand(self.n_head, self.head_dim).reshape(-1)
        ones = self.temperature_ones.to(temps.device, temps.dtype)
        scaler = torch.cat([temps, ones], dim=0).unsqueeze(-1)
        fused_weight = self.c_attn.weight * scaler
        return F.linear(x, fused_weight)

    def forward(self, x, cos, sin):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.learn_temperature:
            qkv = self._c_attn_temperatures(x)
        else:
            qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=self.gqa)
        y = y.transpose(1, 2)  # (B, T, n_head, head_dim)
        if self.head_gate:
            gate = torch.sigmoid(self.gate_proj(x)).view(B, T, self.n_head, 1)
            y = y * gate
        y = y.contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


MLP_ACTIVATIONS = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "relu2": lambda x: F.relu(x).square(),
}


def _align64(x: float) -> int:
    """round up to the nearest larger multiple of 64"""
    return math.ceil(x / 64) * 64


class GLUFeedForward(nn.Module):
    """GLU-family FFN"""

    def __init__(self, config, mlp_alpha):
        super().__init__()
        self.act = MLP_ACTIVATIONS[config.mlp_act]
        d_model = config.n_embd
        d_ff_gated = _align64((2 * mlp_alpha / 3) * d_model)
        print(f"GLU FF hidden dimension set to {d_ff_gated}")
        self.lin = nn.Linear(d_model, 2 * d_ff_gated, bias=False)
        self.out = nn.Linear(d_ff_gated, d_model, bias=False)
        with torch.no_grad():
            self.lin.weight.normal_(std=math.sqrt(4.0 / d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_gate_y_out = self.lin(x)
        y_gate, y_out = y_gate_y_out.chunk(2, dim=-1)
        g_in = y_gate
        gate_act = self.act(g_in)
        gated_state = rmsnorm(gate_act * y_out)
        return self.out(gated_state)


class MLP(nn.Module):
    def __init__(self, config, mlp_alpha):
        super().__init__()
        d_ff = _align64(mlp_alpha * config.n_embd)
        print(f"MLP FF hidden dimension set to {d_ff}")
        self.c_fc = nn.Linear(config.n_embd, d_ff, bias=False)
        self.c_proj = nn.Linear(d_ff, config.n_embd, bias=False)
        self.act = MLP_ACTIVATIONS[config.mlp_act]

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class MultiTokenPredictionHead(nn.Module):
    def __init__(self, config, offsets):
        super().__init__()
        self.offsets = tuple(offsets)
        self.proj = nn.Linear(config.n_embd, len(self.offsets) * config.n_embd, bias=False)
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, _ = x.size()
        return self.proj(x).view(B, T, len(self.offsets), self.n_embd)


class Block(nn.Module):
    def __init__(self, config, mlp_alpha, no_ffn=False):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
        self.no_ffn = no_ffn
        self.mlp_skip = config.mlp_skip
        if not self.no_ffn:
            self.mlp = {"mlp": MLP, "glu": GLUFeedForward}[config.ff_kind](config, mlp_alpha)
        elif self.mlp_skip:
            self.mlp_skip_weight = nn.Parameter(torch.ones(config.n_embd))

    def forward(self, x, cos, sin, mlp_skip_features):
        x = x + self.attn_scale * self.attn(rmsnorm(x), cos, sin)
        if not self.no_ffn:
            mlp_out = self.mlp(rmsnorm(x))
            x = x + mlp_out
            # carry this block's MLP output forward so later MLP-free blocks can skip-connect to it
            mlp_skip_features = mlp_out if self.mlp_skip else None
        else:
            if self.mlp_skip:
                x = x + self.mlp_skip_weight * mlp_skip_features
            x = x * (1 + inv_rms(x))
        return x, mlp_skip_features


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    learn_temperature: bool = False
    temperature_scale: float = 0.35
    ff_dropout: float = 0.0
    ff_kind: Literal["glu", "mlp"] = "mlp"
    mlp_act: Literal["gelu", "relu", "silu", "relu2"] = "gelu"
    mlp_alpha: float | int | list[float | int] = 4.0  # MLP/GLU hidden-dim multiplier, scalar or per-block list
    gqa: bool = False
    head_gate: bool = False
    mlp_drop_n: int = 1  # drop every N-1 mlps, keeping the first one.
    mlp_skip: bool = False  # skip-connect the last real MLP block's output into the MLP-free blocks that follow it
    mtp_offsets: tuple[int, ...] = ()  # extra future-token offsets to predict, e.g. (2, 3) for t+2, t+3. Empty disables MTP.
    mtp_lambdas: tuple[float, ...] = ()  # per-offset loss weights matching mtp_offsets order; empty defaults to equal weights summing to 1.


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.n_layer % config.mlp_drop_n == 0, f"n_layer ({config.n_layer}) must be divisible by mlp_drop_n ({config.mlp_drop_n})"

        mlp_alphas = config.mlp_alpha
        if isinstance(mlp_alphas, (int, float)):
            mlp_alphas = [mlp_alphas] * config.n_layer
        assert len(mlp_alphas) == config.n_layer, f"mlp_alpha list length ({len(mlp_alphas)}) must match n_layer ({config.n_layer})"

        blocks = []
        for i in range(0, config.n_layer, config.mlp_drop_n):
            blocks.append(Block(config, mlp_alphas[i], no_ffn=False))
            for j in range(1, config.mlp_drop_n):
                print(f"Transformer Block {i + j} disabled FF network")
                blocks.append(Block(config, mlp_alphas[i + j], no_ffn=True))

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList(blocks),
            )
        )
        self.rotary = Rotary(config.n_embd // config.n_head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        if config.mtp_offsets:
            seen = set()
            self.mtp_offsets = tuple(k for k in config.mtp_offsets if not (k in seen or seen.add(k)))  # dedup, preserve order
        else:
            self.mtp_offsets = ()
        if self.mtp_offsets:
            assert all(k >= 2 for k in self.mtp_offsets), "mtp_offsets must be >= 2 (t+1 is already covered by the standard head)"
            self.mtp_head = MultiTokenPredictionHead(config, self.mtp_offsets)
            if config.mtp_lambdas:
                assert len(config.mtp_lambdas) == len(self.mtp_offsets), (
                    f"mtp_lambdas length ({len(config.mtp_lambdas)}) must match mtp_offsets length ({len(self.mtp_offsets)}); "
                    f"order follows --mtp, e.g. --mtp 2 3 --mtp_lambdas 0.5 0.25"
                )
                self.mtp_lambdas = tuple(config.mtp_lambdas)
            else:
                self.mtp_lambdas = tuple(1.0 / len(self.mtp_offsets) for _ in self.mtp_offsets)

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        cos, sin = self.rotary(x)

        mlp_skip_features = None
        for block in self.transformer.h:
            x, mlp_skip_features = block(x, cos, sin, mlp_skip_features)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            ntp_loss = loss

            if self.mtp_offsets:
                # targets[:, k-1:] holds the token at position t+k for hidden state x[:, t]
                # (targets is already the t+1 shift of idx, so k=1 recovers the standard head above).
                mtp_hidden = self.mtp_head(x)  # (b, t, n_offsets, n_embd)
                mtp_loss = 0.0
                for i, k in enumerate(self.mtp_offsets):
                    valid_len = t - (k - 1)
                    h_k = rmsnorm(mtp_hidden[:, :valid_len, i, :])
                    logits_k = self.lm_head(h_k)  # computed and consumed one offset at a time to cap peak memory
                    targets_k = targets[:, k - 1 :]
                    mtp_loss = mtp_loss + self.mtp_lambdas[i] * F.cross_entropy(logits_k.reshape(-1, logits_k.size(-1)), targets_k.reshape(-1), ignore_index=-1)
                loss = loss + mtp_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
            ntp_loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss, ntp_loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, temperature_lr_mult=1.0):
        temperature_params = [p for n, p in self.named_parameters() if n.endswith("log_temperature")]
        other_params = [p for n, p in self.named_parameters() if not n.endswith("log_temperature")]
        param_groups = [
            {"params": other_params, "lr_scale": 1.0},
            {"params": temperature_params, "lr_scale": temperature_lr_mult, "weight_decay": 0.0, "is_temperature": True},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay, betas=betas)
        return optimizer


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # how many tokens of validation data. It's important to keep this fixed for consistent comparisons


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def inspect_model_to_log(model: GPT) -> dict:
    """Custom function for sniffing and debugging the optimizations, e.g. inspecting values of learned params"""
    transformer_blocks: nn.ModuleList = model.transformer["h"]
    log = {}
    for i, block in enumerate(transformer_blocks):
        attn_i: CausalSelfAttention = block.attn
        if attn_i.learn_temperature:
            for j, tau in enumerate(attn_i._get_learned_temperatures().detach().cpu().numpy()):
                key = f"temperature/block[{i}]_head[{j}]"
                log[key] = tau
                print0(f"{key}={log[key]}")
    return log


if __name__ == "__main__":
    import argparse
    import time

    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument(
        "--input_bin",
        type=str,
        default="data/fineweb10B/fineweb_train_*.bin",
        help="input .bin to train on",
    )
    parser.add_argument(
        "--input_val_bin",
        type=str,
        default="data/fineweb10B/fineweb_val_*.bin",
        help="input .bin to eval validation loss on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="output directory to which to write logs and checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="d12",
        help="d12|d24|d36|d48",
    )
    # token layout for each step of the optimization
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size, in units of #batch dimensions",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="number of gradient accumulation steps",
    )
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate warmup iterations",
    )
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=0,
        help="learning rate warmdown iterations",
    )
    parser.add_argument(
        "--learn_temperature",
        action="store_true",
        default=False,
        help="learn per-head per-layer temperatures for softmax",
    )
    parser.add_argument("--temperature_scale", type=float, default=0.35, help="tanh-clipped log-temperature scaling")
    parser.add_argument(
        "--temperature_lr_mult",
        type=float,
        default=1.0,
        help="multiplier applied to the base learning rate for the learned softmax temperatures",
    )
    parser.add_argument(
        "--temperature_freeze_steps",
        type=int,
        default=-1,
        help="number of initial steps during which the learned softmax temperatures are frozen (lr=0)",
    )
    parser.add_argument("--ff_kind", type=str, choices=["mlp", "glu"], default="mlp", help="kind of feedforward network used in the model")
    parser.add_argument("--mlp_act", type=str, choices=["gelu", "relu", "silu", "relu2"], default="gelu", help="activation function used in the MLP/GLU feedforward network")
    parser.add_argument(
        "--mlp_alpha",
        type=float,
        nargs="+",
        default=[4.0],
        help="MLP/GLU hidden-dim multiplier: single value, or a list of values matching the number of blocks",
    )
    parser.add_argument("--mlp_drop_n", type=int, default=1, help="share MLP weights every N consecutive blocks (n_layer must be divisible by N)")
    parser.add_argument(
        "--mlp_skip",
        action="store_true",
        default=False,
        help="skip-connect the last real MLP block's output (scaled by a learned multiplicative weight) into the "
        "output of every MLP-free block that follows it (only relevant when --mlp_drop_n > 1)",
    )
    parser.add_argument(
        "--mtp",
        type=int,
        nargs="*",
        default=[],
        help="extra future-token offsets to predict via a fused multi-token-prediction head, e.g. --mtp 2 3 additionally "
        "predicts t+2 and t+3 alongside the standard t+1 objective. Disabled by default (no flag / empty list).",
    )
    parser.add_argument(
        "--mtp_lambdas",
        type=float,
        nargs="*",
        default=[],
        help="per-offset loss weights matching --mtp order, e.g. --mtp 2 3 --mtp_lambdas 0.5 0.25. If omitted, defaults to equal weights 1/len(mtp) for each offset.",
    )
    parser.add_argument("--gqa", action="store_true", default=False, help="enable group-query attention")
    parser.add_argument("--head_gate", action="store_true", default=False, help="enable learned per-head output gating in attention")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    # evaluation
    parser.add_argument(
        "--val_loss_every",
        type=int,
        default=0,
        help="every how mant steps to evaluate val loss?",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
        help="how many batches of val to average?",
    )
    parser.add_argument(
        "--extend_training",
        action="store_true",
        default=False,
        help="if val loss hasn't reached --target_val_loss by --num_iterations, keep training past it (up to "
        "--max_iterations) instead of stopping; also stops as soon as the target is reached, even before "
        "--num_iterations. Requires --target_val_loss and --max_iterations.",
    )
    parser.add_argument(
        "--target_val_loss",
        type=float,
        default=3.3821,
        help="convergence target for validation loss, used by --extend_training",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="hard cap on iterations when --extend_training is set; the run always stops here regardless of convergence",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="every how many steps to save the checkpoint",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="log to wandb",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}
    if args.extend_training:
        assert args.target_val_loss is not None and args.max_iterations is not None, "--extend_training requires both --target_val_loss and --max_iterations to be set"
        assert args.max_iterations >= args.num_iterations, "--max_iterations must be >= --num_iterations"
        assert args.val_loss_every > 0, "--extend_training requires --val_loss_every > 0 to check convergence"
    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    assert args.grad_accumulation_steps % ddp_world_size == 0, "grad_accumulation_steps must be divisible by world size"
    args.grad_accumulation_steps //= ddp_world_size  # each gpu does its fraction of the work
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = 0  # each process gets the exact same seed
    print(f"using device: {device}")

    if args.log_wandb and master_process:
        import datetime

        import wandb

        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wandb.init(project="benchmark_gpt2", name=f"gpt2-{args.model} {start_time}")
        wandb.config.update(args)
        wandb.save("train_gpt2.py")
        wandb.save("run.sh")

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    mlp_alpha = args.mlp_alpha[0] if len(args.mlp_alpha) == 1 else args.mlp_alpha

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    # init the model from scratch
    num_vocab = 50257
    model_config = {
        "d12": GPTConfig(
            vocab_size=num_vocab,
            n_layer=12,
            n_head=12,
            n_embd=768,
            learn_temperature=args.learn_temperature,
            temperature_scale=args.temperature_scale,
            ff_kind=args.ff_kind,
            mlp_act=args.mlp_act,
            mlp_alpha=mlp_alpha,
            gqa=args.gqa,
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
            mlp_skip=args.mlp_skip,
            mtp_offsets=tuple(args.mtp),
            mtp_lambdas=tuple(args.mtp_lambdas),
        ),  # 124M GPT-2
        "d24": GPTConfig(
            vocab_size=num_vocab,
            n_layer=24,
            n_head=16,
            n_embd=1024,
            learn_temperature=args.learn_temperature,
            temperature_scale=args.temperature_scale,
            ff_kind=args.ff_kind,
            mlp_act=args.mlp_act,
            mlp_alpha=mlp_alpha,
            gqa=args.gqa,
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
            mlp_skip=args.mlp_skip,
            mtp_offsets=tuple(args.mtp),
            mtp_lambdas=tuple(args.mtp_lambdas),
        ),
        "d36": GPTConfig(
            vocab_size=num_vocab,
            n_layer=36,
            n_head=20,
            n_embd=1280,
            learn_temperature=args.learn_temperature,
            temperature_scale=args.temperature_scale,
            ff_kind=args.ff_kind,
            mlp_act=args.mlp_act,
            mlp_alpha=mlp_alpha,
            gqa=args.gqa,
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
            mlp_skip=args.mlp_skip,
            mtp_offsets=tuple(args.mtp),
            mtp_lambdas=tuple(args.mtp_lambdas),
        ),
        "d48": GPTConfig(
            vocab_size=num_vocab,
            n_layer=48,
            n_head=25,
            n_embd=1600,
            learn_temperature=args.learn_temperature,
            temperature_scale=args.temperature_scale,
            ff_kind=args.ff_kind,
            mlp_act=args.mlp_act,
            mlp_alpha=mlp_alpha,
            gqa=args.gqa,
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
            mlp_skip=args.mlp_skip,
            mtp_offsets=tuple(args.mtp),
            mtp_lambdas=tuple(args.mtp_lambdas),
        ),
        "d36": GPTConfig(
            vocab_size=num_vocab,
            n_layer=36,
            n_head=20,
            n_embd=1280,
            learn_temperature=args.learn_temperature,
            temperature_scale=args.temperature_scale,
            ff_kind=args.ff_kind,
            mlp_act=args.mlp_act,
            mlp_alpha=mlp_alpha,
            gqa=args.gqa,
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
            mlp_skip=args.mlp_skip,
            mtp_offsets=tuple(args.mtp),
            mtp_lambdas=tuple(args.mtp_lambdas),
        ),
        "d48": GPTConfig(
            vocab_size=num_vocab,
            n_layer=48,
            n_head=25,
            n_embd=1600,
            learn_temperature=args.learn_temperature,
            temperature_scale=args.temperature_scale,
            ff_kind=args.ff_kind,
            mlp_act=args.mlp_act,
            mlp_alpha=mlp_alpha,
            gqa=args.gqa,
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
            mlp_skip=args.mlp_skip,
            mtp_offsets=tuple(args.mtp),
            mtp_lambdas=tuple(args.mtp_lambdas),
        ),
    }[args.model]
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model...")
    model = torch.compile(model)  # NOTE: this might cause issues depending on your GPU, consider turning it off

    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
        temperature_lr_mult=args.temperature_lr_mult,
    )

    # learning rate decay scheduler (linear warmup and warmdown)
    # when --extend_training is set, the warmdown is stretched out over --max_iterations instead of
    # --num_iterations, so the lr hasn't already decayed to 0 by the time we might extend past num_iterations
    lr_horizon = args.max_iterations if args.extend_training else args.num_iterations

    def get_lr(it):
        assert it <= lr_horizon
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < lr_horizon - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (lr_horizon - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    run_id = str(uuid.uuid4())

    # create the logging directory if it does not exist
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "%s.log" % run_id)
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    training_time_ms = 0.0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # begin training
    # normally the run spans num_iterations; with --extend_training the hard cap is max_iterations instead,
    # and we may stop earlier than that (but not before num_iterations) once target_val_loss is reached
    for step in range(lr_horizon + 1):
        last_step = step == lr_horizon

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or step == args.num_iterations or last_step):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()  # reset the val loader so that it starts from the beginning
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):  # always fiexed number of validation steps
                    x_val, y_val = val_loader.next_batch()
                    _, _, ntp_loss = model(x_val, y_val, return_logits=False)
                    val_loss += ntp_loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
            # log to console and to file
            print0(f"step:{step}/{lr_horizon} | val loss {val_loss:.6f}")
            if master_process:
                if args.log_wandb:
                    wandb.log({"val_loss": val_loss}, step=step * tokens_per_iter)
                    wandb.log({"time": training_time_ms}, step=step * tokens_per_iter)
                    wandb.log(inspect_model_to_log(model.module), step=step * tokens_per_iter)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d val:%f\n" % (step, val_loss))

            # restart the clock
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # every rank computes val_loss identically (all_reduce above), so this branches the same way everywhere
            if args.extend_training and step >= args.num_iterations and val_loss <= args.target_val_loss:
                print0(f"converged: val loss {val_loss:.6f} <= target {args.target_val_loss:.6f} at step {step}/{lr_horizon}")
                last_step = True
            elif args.extend_training and step == args.num_iterations and val_loss > args.target_val_loss:
                print0(
                    f"val loss {val_loss:.6f} > target {args.target_val_loss:.6f} at step {step}/{args.num_iterations}; extending training up to {args.max_iterations} iterations"
                )

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss = torch.zeros(1, device=device)
        for micro_step in range(args.grad_accumulation_steps):
            model.require_backward_grad_sync = micro_step == args.grad_accumulation_steps - 1  # sync only on last micro step to avoid overhead
            # forward pass
            with ctx:
                _, loss, _ = model(x, y, return_logits=False)
                loss = loss / args.grad_accumulation_steps  # scale loss for gradient accumulation
                train_loss += loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()

        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            group_lr = lr * param_group.get("lr_scale", 1.0)
            if param_group.get("is_temperature", False) and step < args.temperature_freeze_steps:
                group_lr = 0.0
            param_group["lr"] = group_lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # time and print
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        # the 0th iteration is often an outlier (much slower) => skip logging it
        # tokens_per_second = ddp_world_size * B * T / (t1-t0)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()  # keep track of the mean loss
        print0(f"step:{step}/{lr_horizon} | loss {lossf:.6f} | train_time:{approx_training_time_ms / 1000:.2f}s | step_avg:{approx_training_time_ms / (step + 1):.2f}ms")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trn:%f\n" % (step, lossf))

        if master_process and (step + 1) % args.save_every == 0:
            log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
            os.makedirs("logs/%s" % run_id, exist_ok=True)
            torch.save(log, "logs/%s/model_step%06d.pt" % (run_id, step))

    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs("logs/%s" % run_id, exist_ok=True)
        torch.save(log, "logs/%s/final.pt" % run_id)

    # -------------------------------------------------------------------------
    # clean up nice
    destroy_process_group()
