import glob
import math
import os
import queue
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch._inductor import config
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_float32_matmul_precision('high')

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
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
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


class Block(nn.Module):
    def __init__(self, config, mlp_alpha, no_ffn=False):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
        self.no_ffn = no_ffn
        if not self.no_ffn:
            self.mlp = {"mlp": MLP, "glu": GLUFeedForward}[config.ff_kind](config, mlp_alpha)

    def forward(self, x, cos, sin):
        x = x + self.attn_scale * self.attn(rmsnorm(x), cos, sin)
        if not self.no_ffn:
            x = x + self.mlp(rmsnorm(x))
        else:
            x = x + rmsnorm(x)
        return x


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
    head_gate: bool = False
    mlp_drop_n: int = 1  # drop every N-1 mlps, keeping the first one.


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

    def forward(self, idx, targets=None, return_logits=True):
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        cos, sin = self.rotary(x)

        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, temperature_lr_mult=1.0):
        temperature_params = [p for n, p in self.named_parameters() if n.endswith("log_temperature")]
        other_params = [p for n, p in self.named_parameters() if not n.endswith("log_temperature")]
        param_groups = [
            {"params": other_params, "lr_scale": 1.0},
            {"params": temperature_params, "lr_scale": temperature_lr_mult, "weight_decay": 0.0, "is_temperature": True},
        ]
        # fused AdamW is a single CUDA kernel per step instead of one per tensor
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay, betas=betas, fused=device_type.startswith("cuda"))
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
    # cast the whole shard to the dtype the model expects once here, so that
    # next_batch() only has to slice/view (no re-casting per batch)
    return torch.from_numpy(tokens.astype(np.int64))


class DistributedDataLoader:
    PREFETCH_BATCHES = 8

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
        self._stop_prefetch()  # no worker is running past this point
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])
        self._start_prefetch()

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def _make_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        # pin here (in the background thread) so the later .cuda(non_blocking=True)
        # transfer in next_batch() is actually asynchronous
        return x.pin_memory(), y.pin_memory()

    def _prefetch_worker(self):
        while not self._stop_event.is_set():
            batch = self._make_batch()
            while not self._stop_event.is_set():
                try:
                    self._queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _start_prefetch(self):
        self._queue = queue.Queue(maxsize=self.PREFETCH_BATCHES)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()

    def _stop_prefetch(self):
        if getattr(self, "_thread", None) is None:
            return
        self._stop_event.set()
        # drain the queue in case the worker is blocked on a full put()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join()
        self._thread = None

    def next_batch(self):
        x, y = self._queue.get()
        return x.cuda(non_blocking=True), y.cuda(non_blocking=True)


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # how many tokens of validation data. It's important to keep this fixed for consistent comparisons
# validation never runs longer than this, whatever the training curriculum does:
# it is the length the run is scored at, so it has to stay comparable
VAL_SEQ_LEN_MAX = 1024


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def parse_grad_accum_schedule(spec, default_steps):
    """"0:1,0.5:4,0.9:16" -> [(0.0, 1), (0.5, 4), (0.9, 16)].

    Fractions are of the run length. An empty spec means a constant default_steps.
    """
    if not spec:
        return [(0.0, default_steps)]
    stages = sorted((float(frac), int(steps)) for frac, steps in (entry.split(":") for entry in spec.split(",")))
    assert stages[0][0] == 0.0, "schedule must start at fraction 0"
    for _, steps in stages:
        assert steps > 0, f"grad accumulation steps must be positive, got {steps}"
    return stages


def grad_accum_at(stages, progress):
    """Gradient accumulation steps for a given fraction of training completed."""
    # stages is sorted and starts at 0.0, so this always matches
    return next(steps for frac, steps in reversed(stages) if progress >= frac)


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
    parser.add_argument(
        "--grad_accum_schedule",
        type=str,
        default="",
        help="effective-batch curriculum as 'frac:steps,frac:steps,...', fractions of the run length, "
        "e.g. '0:1,0.5:4,0.9:16'. Empty means a constant --grad_accumulation_steps. The sequence length "
        "is left alone, so this varies tokens per iteration purely through update frequency.",
    )
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
    # the sequence length is constant now: the curriculum acts on the effective batch
    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}
    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    grad_accum_stages = parse_grad_accum_schedule(args.grad_accum_schedule, args.grad_accumulation_steps)
    for _, steps in grad_accum_stages:
        assert steps % ddp_world_size == 0, "every scheduled grad accumulation value must be divisible by world size"
    grad_accum_stages = [(frac, steps // ddp_world_size) for frac, steps in grad_accum_stages]  # each gpu does its fraction
    print0(f"Using grad accumulation stages {grad_accum_stages}")
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

    # the shape is fixed, so an iteration is worth fewer tokens when it accumulates
    # over fewer micro-batches
    def tokens_per_iter_at(grad_accum):
        return B * T * ddp_world_size * grad_accum

    planned_tokens = sum(tokens_per_iter_at(grad_accum_at(grad_accum_stages, s / args.num_iterations)) for s in range(args.num_iterations))
    print0(f"total tokens over the run: {planned_tokens:,}")

    mlp_alpha = args.mlp_alpha[0] if len(args.mlp_alpha) == 1 else args.mlp_alpha

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    # validation is pinned at VAL_SEQ_LEN_MAX so the score stays comparable when
    # training runs longer than the scored length
    val_T = min(T, VAL_SEQ_LEN_MAX)
    tokens_per_iter_val = args.val_batch_size * val_T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val
    val_loader = DistributedDataLoader(args.input_val_bin, args.val_batch_size, val_T, ddp_rank, ddp_world_size)

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
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
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
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
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
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
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
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
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
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
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
            head_gate=args.head_gate,
            mlp_drop_n=args.mlp_drop_n,
        ),
    }[args.model]
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    if ddp_world_size == 1:
        torch._dynamo.config.optimize_ddp = False
    print0("compiling the model...")
    model = torch.compile(model, dynamic=False)  # NOTE: this might cause issues depending on your GPU, consider turning it off

    # here we wrap model into DDP container
    # broadcast_buffers=False: the only buffer (Rotary.inv_freq) is static and
    # identical across ranks by construction, so there's nothing to re-sync
    model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False)
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
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
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
    total_tokens = 0  # actually consumed so far, which the schedule makes non-linear
    curr_grad_accum = None
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # begin training
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        # effective-batch curriculum. the shape is fixed, so fewer accumulation steps
        # means a cheaper iteration carrying proportionally fewer tokens: the model
        # sees full context throughout and only the update frequency changes.
        prev_grad_accum, curr_grad_accum = curr_grad_accum, grad_accum_at(grad_accum_stages, step / args.num_iterations)
        if curr_grad_accum != prev_grad_accum:
            print0(f"step:{step}/{args.num_iterations} | grad accum -> {curr_grad_accum} | tokens/iter {tokens_per_iter_at(curr_grad_accum):,}")

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()  # reset the val loader so that it starts from the beginning
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
            # log to console and to file
            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f}")
            if master_process:
                if args.log_wandb:
                    wandb.log({"val_loss": val_loss}, step=total_tokens)
                    wandb.log({"time": training_time_ms}, step=total_tokens)
                    wandb.log(inspect_model_to_log(model.module), step=total_tokens)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d val:%f\n" % (step, val_loss))

            # restart the clock
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss = torch.zeros(1, device=device)
        for micro_step in range(curr_grad_accum):
            model.require_backward_grad_sync = micro_step == curr_grad_accum - 1  # sync only on last micro step to avoid overhead
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = loss / curr_grad_accum  # scale loss for gradient accumulation
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
        total_tokens += tokens_per_iter_at(curr_grad_accum)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # time and print
        # no separate torch.cuda.synchronize() here: train_loss.item() below
        # already blocks until this step's GPU work (and the all_reduce) is
        # done, so a dedicated sync first would just be a redundant device drain
        # the 0th iteration is often an outlier (much slower) => skip logging it
        # tokens_per_second = ddp_world_size * B * T / (t1-t0)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()  # keep track of the mean loss
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | train_time:{approx_training_time_ms / 1000:.2f}s | step_avg:{approx_training_time_ms / (step + 1):.2f}ms")
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
    # join the prefetch workers first: a daemon thread still inside pin_memory()
    # when the interpreter tears down aborts the process (SIGABRT) on exit
    train_loader._stop_prefetch()
    val_loader._stop_prefetch()
    destroy_process_group()
