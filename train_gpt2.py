import os
import sys
import uuid
import math
import glob
import queue
import threading
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # fused AdamW is a single CUDA kernel per step instead of one per tensor
        use_fused = device_type.startswith("cuda")
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            fused=use_fused,
        )
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
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
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
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # guards T / current_position / current_shard / tokens, which the prefetch
        # thread reads while the training loop may be changing the sequence length
        self._lock = threading.Lock()

        # kick things off
        self.reset()

    def reset(self):
        self._stop_prefetch()  # no worker is running past this point
        with self._lock:
            self.current_shard = 0
            self.current_position = self.process_rank * self.B * self.T
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self._start_prefetch()

    def set_shape(self, B, T):
        """Change the batch shape without restarting the prefetch thread.

        Batches already queued at the old shape are tagged and dropped by
        next_batch(). The read position is realigned so that a larger slice
        still fits inside the current shard.
        """
        with self._lock:
            if (B, T) == (self.B, self.T):
                return
            self.B = B
            self.T = T
            self.current_position = self.process_rank * B * T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def _make_batch(self):
        # advance() is called with the lock held, so it must not take it itself
        with self._lock:
            B = self.B
            T = self.T
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]
            x = (buf[:-1]).view(B, T)  # inputs
            y = (buf[1:]).view(B, T)  # targets
            # advance current position and load next shard if necessary
            self.current_position += B * T * self.num_processes
            if self.current_position + (B * T * self.num_processes + 1) > len(
                self.tokens
            ):
                self.advance()
        # pin here (in the background thread) so the later .cuda(non_blocking=True)
        # transfer in next_batch() is actually asynchronous. x and y are views onto
        # a shard the worker may since have replaced, but pin_memory() copies.
        return (B, T), x.pin_memory(), y.pin_memory()

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
        # only the training loop calls this and set_shape(), so self.B/self.T are a
        # consistent read here: skip whatever the worker built at the old shape
        while True:
            shape, x, y = self._queue.get()
            if shape == (self.B, self.T):
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


def parse_seq_len_schedule(spec, max_seq_len):
    """"0:256,0.3:512,0.6:1024" -> [(0.0, 256), (0.3, 512), (0.6, 1024)].

    Fractions are of num_iterations. An empty spec means a constant max_seq_len.
    """
    if not spec:
        return [(0.0, max_seq_len)]
    stages = sorted(
        (float(frac), int(length))
        for frac, length in (entry.split(":") for entry in spec.split(","))
    )
    assert stages[0][0] == 0.0, "schedule must start at fraction 0"
    for _, length in stages:
        assert 0 < length <= max_seq_len, f"stage length {length} exceeds sequence_length"
        assert max_seq_len % length == 0, f"stage length {length} must divide sequence_length"
    return stages


def seq_len_at(stages, progress):
    """Sequence length for a given fraction of training completed."""
    # stages is sorted and starts at 0.0, so this always matches
    return next(length for frac, length in reversed(stages) if progress >= frac)


if __name__ == "__main__":
    import time
    import argparse

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
    parser.add_argument(
        "--sequence_length", type=int, default=64, help="sequence length"
    )
    parser.add_argument(
        "--seq_len_schedule",
        type=str,
        default="",
        help="context-length curriculum as 'frac:len,frac:len,...', fractions of "
        "num_iterations, e.g. '0:256,0.3:512,0.6:1024'. Lengths must divide "
        "--sequence_length, which is the final (largest) length. Empty means a "
        "constant --sequence_length. Tokens per iteration are held fixed by "
        "scaling batch_size inversely with the current length.",
    )
    # workload (number of steps)
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="number of iterations to run"
    )
    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate warmup iterations",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="learning rate warmup iterations"
    )
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=0,
        help="learning rate warmdown iterations",
    )
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
    args = parser.parse_args()

    # args error checking and convenience variables
    # T is the largest (final) sequence length; the curriculum ramps up to it
    B, T = args.batch_size, args.sequence_length
    seq_len_stages = parse_seq_len_schedule(args.seq_len_schedule, T)
    print0(f"Using sequence length stages {seq_len_stages}")
    assert args.model in {"d12", "d24", "d36", "d48"}
    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    assert (
        args.grad_accumulation_steps % ddp_world_size == 0
    ), "grad_accumulation_steps must be divisible by world size"
    args.grad_accumulation_steps //= (
        ddp_world_size  # each gpu does its fraction of the work
    )
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = 0  # each process gets the exact same seed
    print(f"using device: {device}")

    if args.log_wandb and master_process:
        import wandb
        import datetime

        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wandb.init(project="benchmark_gpt2", name=f"gpt2-{args.model} {start_time}")
        wandb.config.update(args)
        wandb.save("train_gpt2.py")
        wandb.save("run.sh")

    # the batch is fixed, so an iteration is worth fewer tokens at a shorter length
    def tokens_per_iter_at(seq_len):
        return B * seq_len * ddp_world_size * args.grad_accumulation_steps

    planned_tokens = sum(
        tokens_per_iter_at(seq_len_at(seq_len_stages, s / args.num_iterations))
        for s in range(args.num_iterations)
    )
    print0(f"total tokens over the run: {planned_tokens:,}")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # load tokens. the train loader is built at the largest length so that the
    # shard size checks cover the biggest slice we will ever ask for, then dropped
    # to the first stage of the curriculum.
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    curr_seq_len = seq_len_at(seq_len_stages, 0.0)
    train_loader.set_shape(B, curr_seq_len)
    x, y = train_loader.next_batch()
    # the loop below only logs on a *change*, so announce the opening stage here
    print0(
        f"step:0/{args.num_iterations} | seq len -> {curr_seq_len}"
        f" | tokens/iter {tokens_per_iter_at(curr_seq_len):,}"
    )

    # validation follows the curriculum but never past VAL_SEQ_LEN_MAX, so the score
    # stays comparable once training runs longer than the scored length. the batch is
    # fixed here too, so the *step count* varies inversely with the length and every
    # validation still covers exactly VAL_TOKENS tokens.
    val_T = min(T, VAL_SEQ_LEN_MAX)

    def val_steps_for(seq_len):
        tokens_per_iter_val = args.val_batch_size * seq_len * ddp_world_size
        assert VAL_TOKENS % tokens_per_iter_val == 0
        return VAL_TOKENS // tokens_per_iter_val

    # fail at startup rather than mid-run if any stage cannot hit VAL_TOKENS exactly
    for _, stage_len in seq_len_stages:
        val_steps_for(min(stage_len, VAL_SEQ_LEN_MAX))

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, val_T, ddp_rank, ddp_world_size
    )
    val_loader.set_shape(args.val_batch_size, min(curr_seq_len, VAL_SEQ_LEN_MAX))

    # init the model from scratch
    num_vocab = 50257
    model_config = {
        "d12": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768
        ),  # 124M GPT-2
        "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    if ddp_world_size == 1:
        torch._dynamo.config.optimize_ddp = False
    print0("compiling the model...")
    model = torch.compile(
        model, dynamic=False
    )  # NOTE: this might cause issues depending on your GPU, consider turning it off

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
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # begin training
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        # context-length curriculum. the batch is fixed, so a shorter length means a
        # cheaper iteration carrying proportionally fewer tokens rather than the same
        # tokens rearranged.
        curr_seq_len = seq_len_at(seq_len_stages, step / args.num_iterations)
        if curr_seq_len != train_loader.T:
            train_loader.set_shape(B, curr_seq_len)
            val_loader.set_shape(args.val_batch_size, min(curr_seq_len, VAL_SEQ_LEN_MAX))
            x, y = train_loader.next_batch()  # drop the batch built at the old shape
            print0(
                f"step:{step}/{args.num_iterations} | seq len -> {curr_seq_len}"
                f" | tokens/iter {tokens_per_iter_at(curr_seq_len):,}"
            )

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()  # reset the val loader so that it starts from the beginning
            # varies with the length so that every validation still sees VAL_TOKENS
            val_steps = val_steps_for(val_loader.T)
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
        for micro_step in range(args.grad_accumulation_steps):
            model.require_backward_grad_sync = (
                micro_step == args.grad_accumulation_steps - 1
            )  # sync only on last micro step to avoid overhead
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = (
                    loss / args.grad_accumulation_steps
                )  # scale loss for gradient accumulation
                train_loss += loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()

        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_tokens += tokens_per_iter_at(curr_seq_len)
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
        print0(
            f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | train_time:{approx_training_time_ms/1000:.2f}s | step_avg:{approx_training_time_ms/(step+1):.2f}ms"
        )
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trn:%f\n" % (step, lossf))

        if master_process and (step + 1) % args.save_every == 0:
            log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
            os.makedirs("logs/%s" % run_id, exist_ok=True)
            torch.save(log, "logs/%s/model_step%06d.pt" % (run_id, step))

    print0(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

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
