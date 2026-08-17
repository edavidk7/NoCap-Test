# Experiment log

Target: **val loss ≤ 3.3821**, minimising wall-clock time. All runs are `d12`, FineWeb,
single GPU (RTX PRO 6000 Blackwell), `--batch_size 16 --sequence_length 1024` unless noted.

Wall clock is proportional to **micro-batches** at a near-constant 0.0588 s each
(measured 0.05883–0.05885 across every run, old and new code). The baseline's budget is
**152,576 micro-batches = 2.4998B tokens ≈ 149.6 min**. Runs matching that budget are
directly comparable on loss; runs that don't are comparable only after normalising.

`grad_accum_schedule` is `"frac:steps,..."` over the run; `tokens/update = B × T × ga`.

---

## Runs

| wandb name | id | config | final val | wall | note |
|---|---|---|---|---|---|
| gpt2-d12 2025-04-10 20:31:58 | `64s1zc1w` | upstream modded-nanogpt, ga=32, 4768 it | **3.3821** | — | origin of the target number; different repo path, not run on this machine |
| gpt2-d12 2026-07-30 06:46:11 | `xrfvtgfa` | baseline, ga=32, 4768 it, no seed | 3.3832 | 149.6 m | pre-seed-freeze baseline; misses target by 0.0011 |
| gpt2-d12 2026-07-29 15:36:55 | `y0yn2is1` | **seq-len** schedule `0:256,0.5:1024,0.95:2048`, ga=16 | 3.3913 | — | context curriculum, not batch |
| gpt2-d12 2026-08-02 12:23:25 | `zywxz5av` | **seq-len** schedule `0:64,…,0.9:2048`, ga=16 | 3.6271 | 97.4 m | only 0.96B tokens — short budget explains the loss, says nothing about T=2048 |
| gpt2-d12 2026-08-02 18:05:43 | `rnifgafp` | **grad-accum** schedule `0:2,0.05:4,0.1:8,0.2:16,0.35:32`, 4768 it | 3.4371 | 113.6 m | 1.90B tokens. Ramps *up* to ga=32 → 85.6% of tokens at the worst setting |
| gpt2-d12 2026-08-02 20:33:17 | `9pyva6un` | same schedule, 5408 it | 3.4172 | 128.8 m | 2.15B tokens. With the run above gives `dL/dln(tokens) = −0.158` |
| **gpt2-d12 2026-08-02 22:44:59** | `g7r46qqr` | **baseline, seed 42**, ga=32, 4768 it, lr 0.0018 | **3.3799** | **149.6 m** | **control**. Passes by 0.0022 |
| gpt2-d12 2026-08-03 01:17:50 | `r805vz6k` | dev arch: `glu` + `silu`, α=6, `mlp_drop_n=2`, lr 0.0018 | 3.4330 | 139.2 m | 11.5% fewer params, only 7% faster/step; loses 0.053 |
| gpt2-d12 2026-08-03 03:40:02 | `m9zujnzi` | dev arch, **lr 0.0022** | 3.4200 | 138.8 m | higher LR worth −0.0130 on the smaller model, not enough to close the gap |
| **gpt2-d12 2026-08-03 06:54:58** | `ha8y74jf` | **ga=16**, 9536 it, warmup 512 / warmdown 2048 | **3.3717** | 149.9 m | **best**. Identical compute to baseline, −0.0082. Passes by 0.0104 |
| gpt2-d12 2026-08-03 09:30:55 | `xmqe2sga` | ga=8, 19072 it, warmup 1024 / warmdown 4096 | 3.3948 | 150.4 m | overshoots — optimum is interior, at ga=16 |
| gpt2-d12 2026-08-03 12:07:09 | `xn4vc85y` | hybrid `0:32,0.4775:8`, 7840 it, warmdown 4096 | 3.4205 | 149.7 m | **failed**: 4× batch cut landed at *peak* LR → +0.145 spike, never recovered |
| **gpt2-d12 2026-08-03 14:40:50** | — | **ga=16 shortened**, 9106 it, warmup 489 / warmdown 1956 | **3.379406** | **143.0 m** | **best time-to-target: passes by 0.0027, −4.4% vs baseline at the same loss.** Aimed 3.3790, landed within 0.0004 |
| gpt2-d12 2026-08-03 17:08:38 | — | fixed hybrid `0:32,0.7347:16,0.8232:8`, 5792 it, warmdown 2048 | 3.410423 | 149.6 m | **failed**: gentler transitions (+0.026 / +0.020 spikes vs +0.145) but still a net loss |
| **gpt2-d12 2026-08-03 19:52** | — | **ga=16 shortened further**, 8969 it, warmup 482 / warmdown 1927 | **3.380843** | **140.9 m** | **best time-to-target: passes by 0.0013, −5.8% vs baseline.** Aimed 3.3818, landed within 0.0010 |
| gpt2-d12 2026-08-03 22:20 | — | **SwiGLU at parity** α=4 drop_n=1 + ga=16 + lr 0.0022, 9536 it | 3.390100 | ~157 m | GLU at *equal params*; 3 changes at once. +0.0184 vs ga16 |
| gpt2-d12 2026-08-04 06:45 | — | **GEGLU** (gelu, rmsnorm on gated state removed) on the 8969 config | 3.408549 | 143.9 m | +0.0277 vs the MLP version of the same config. +1.7%/step (was +4.8% with rmsnorm) |
| gpt2-d12 2026-08-04 09:10 | — | ga=16 + **lr 0.0022**, 9536 it | 3.374629 | 149.7 m | **LR isolated: +0.0029 vs lr 0.0018 — a tie within noise, not the −0.0130 seen on the dev arch** |
| **gpt2-d12 2026-08-04 11:52** | — | **ga=16 + warmdown 3072** (32% of tokens vs 21.5%), 9536 it | **3.364937** | 149.7 m | **WORKS: −0.0068 vs wd 2048. Best loss of the session** |
| **gpt2-d12 2026-08-04 14:28** | — | **wd3072 shortened**: 8724 it, warmup 468, warmdown 2811 | **3.376969** | **136.9 m** | **BEST: passes by 0.0051, −8.5% vs baseline.** Aimed 3.3790, landed 0.002 under |
| gpt2-d12 2026-08-04 16:52 | — | ga=16 + **warmdown 4096** (43% of tokens), 9536 it | **3.361818** | 149.8 m | −0.0031 vs wd3072. Best loss overall; gains halving |
| **gpt2-d12 2026-08-04 19:25** | — | **wd4096 shortened**: 8549 it, warmup 459, warmdown 3672 | **3.373880** | **134.4 m** | **passes by 0.0082, −10.2%.** Landed 0.005 under its 3.3790 aim |
| gpt2-d12 2026-08-05 08:31 | — | shortened again: 8276 it, warmup 444, warmdown 3555 | **3.379210** | **131.3 m** | fastest pass, **−12.2%**, but margin only 0.0029 (inside noise) |
| gpt2-d12 2026-08-05 10:48 | 2hc22yei | **replicate of 8276 config at seed 43** | **3.379601** | **130.8 m** | reproduces seed 42 to within **0.0004**. Both pass. |
| gpt2-d12 2026-08-05 13:12 | — | **weight EMA (decay 0.998) on top of** the 8276 config, seed 42 | raw **3.380107** / EMA **3.380984** | 130.7 m | EMA leads all run, is overtaken by the warmdown in the last ~200 steps, ends **0.0009 behind** the raw weights. No gain. |
| gpt2-d12 2026-08-05 19:47 | — | FFN ablation: **SwiGLU matched**, `glu`+`silu` α4 n1 | 3.409365 / 3.410274 | 133.7 m | +0.033 time-matched. Also 2.4% *slower* per step |
| gpt2-d12 2026-08-05 22:00 | — | FFN ablation: **drop-only control**, `mlp` α4 n2 | 3.458548 / 3.459045 | 104.8 m | +0.044 time-matched. Isolates dropping FFNs |
| gpt2-d12 2026-08-05 23:52 | — | FFN ablation: **similar FFN budget**, `glu`+`silu` α4 n2 | 3.472915 / 3.473347 | 106.1 m | +0.060 time-matched, worst of the set |
| gpt2-d12 2026-08-06 01:42 | — | α6 n2 probe: **lr 0.0028** | 3.442425 / 3.443333 | 118.2 m | vs lr 0.0018's 3.442892 — **0.0005**, below the fixed-seed noise floor |
| gpt2-d12 2026-08-06 03:50 | wqtumwcj | α6 n2 probe: **ga 12**, 11,035 it, warmup 592, warmdown 4741 | 3.440474 / 3.440250 | 118.4 m | vs ga16's 3.442892 — 0.0024, within noise. +0.045 time-matched |
| gpt2-d12 2026-08-06 11:39 | letxrf8c | **T=2048, ga 8** (same 262,144 tokens/update) | 3.385583 | 145.2 m | misses by 0.0035; longer context costs ~6% per token |
| gpt2-d12 2026-08-06 22:00 | — | **flex sliding window 1024** at T=2048, ga 8 | 3.381845 | 145.2 m | clears by 0.00026 but +11.1% ms/step — dominated |
| gpt2-d12 2026-08-07 16:31 | — | **EMA horizon sweep** on the 8276 config | raw **3.380037**; ema 0.998/0.999/0.9995/0.9998 = 3.380715 / 3.387231 / 3.419085 / 3.573685 | 131.3 m | monotonic in horizon: longer is strictly worse. No EMA setting beats no EMA |
| gpt2-d12 2026-08-07 19:17 | — | **lr 0.0024 + LR floor 0.10** + ema/swa | raw 3.392667; best ema0.998 **3.382030**; swa 3.384997 | 131.3 m | averaging beats raw for the first time (−0.0106) but only heals the floor's own damage; net +0.0028 vs the plain config |
| **gpt2-d12 2026-08-08 10:32** | — | **B=64, ga=4** (same 262,144 tokens/update, same schedule) | **3.379164** | **127.1 m** | **new best: −15.0% vs baseline.** Loss identical to B=16's 3.379210; pure throughput |
| **gpt2-d12 2026-08-08 13:42** | — | **per-head attention spans** 2x64 2x128 2x512 6x1024, dense SDPA mask | **3.373024** | 158.2 m | **first architecture change to improve the loss: −0.0061 vs causal, held through the whole warmdown.** Dense mask costs +24.7%, so unusable as-is |

---

## What was learned

**Wall clock ≈ tokens.** Per-token cost is fixed by architecture, not by batching. The
grad-accum schedule never sped anything up per token — it just did less work per
iteration, so it finished sooner having learned less.

**Tokens-per-update has an interior optimum near 262,144.** 524,288 (baseline) → 262,144
(ga16) gains 0.0082; 131,072 (ga8) loses 0.023. `ga` alone is meaningless — what matters
is `B × T × ga`.

**Warmdown step count is a confound — batch schedules within a run do not work.** Across
*constant*-batch runs the warmdown drop rises with step count (1024 → 0.1682, 2048 →
0.1968, 4096 → 0.2519), which suggested concentrating updates in the warmdown. It does not
transfer. Two hybrids both lost:

| run | L at warmdown start | wd steps | wd drop | final |
|---|---|---|---|---|
| baseline (ga32 throughout) | 3.5481 | 1024 | 0.1682 | 3.3799 |
| ga16 (ga16 throughout) | 3.5685 | 2048 | **0.1968** | 3.3717 |
| hybrid, 4x cut at 100% LR | 3.5502 | 4096 | 0.1297 | 3.4205 |
| fixed hybrid, 2x cuts at 75%/50% LR | 3.5485 | 2048 | 0.1381 | 3.4104 |

At *equal* warmdown steps (rows 2 and 4) ga16 started 0.020 worse and dropped 0.059 more.
The drop is a property of having trained at that batch size throughout, not of the step
count during annealing — you cannot switch into it at the end. Every batch transition also
costs directly: +0.145 for a 4x cut at peak LR, +0.026 and +0.020 for 2x cuts at 75%/50%.
Since the most a hybrid can gain is 0.020, the transition cost exceeds the entire prize.

**Sequence length is pinned at 1024.** Measured per-token cost: T=256 −2.4%, T=512 −1.9%,
T=2048 +5.6%. Going up costs real compute; going down saves almost nothing (attention is
only ~5–7% of FLOPs at T=1024). With causal masking, a token at position *p* sees identical
context at T=1024 and T=2048, so longer training context cannot improve the positions the
eval scores — and `VAL_SEQ_LEN_MAX = 1024` pins the eval there.

**The dev architecture loses.** 11.5% fewer params buys only 7% wall clock (per-block
overheads don't scale) but costs 0.038–0.053 of loss.

**GLU-family FFNs lose here, at exact parameter parity.** Two variants tested, both with
`d_ff=2048` vs MLP's 3072 (identical param counts, verified):

| variant | vs MLP control | speed |
|---|---|---|
| SwiGLU + rmsnorm on gated state, lr 0.0022 | +0.0184 vs ga16 | +4.8%/step |
| GEGLU, no rmsnorm, lr 0.0018 | +0.0277 vs same-config MLP | +1.7%/step |

Removing the extra `rmsnorm` recovered most of the speed penalty but not the loss gap.
This is contrary to the usual literature result for GLU variants at parity.

**Warmdown length matters and was badly mistuned.** Sweep at identical compute
(9536 x 16 micro-batches):

| warmdown | % of tokens | final val | gain |
|---|---|---|---|
| 2048 | 21.5% (stock) | 3.371701 | — |
| 3072 | 32% | 3.364937 | −0.0068 |
| 4096 | 43% | **3.361818** | −0.0031 |

Total **−0.0099** from the stock setting — larger than the ga=16 effect. Gains are halving
each step, so the optimum is near 43% and further extension is not worth a run.

**LR was already tuned.** lr 0.0022 on the stock architecture gives 3.374629 vs
3.371701 at lr 0.0018 — **+0.0029, a tie within the 0.003–0.005 noise floor**. The
−0.0130 improvement seen on the dev architecture did *not* transfer; that model had 11.5%
fewer parameters. This also means the SwiGLU run's +0.0184 deficit is only ~0.003
attributable to its higher LR — the rest is the FFN.

**Noise floor ≈ 0.003–0.005.** `xn4vc85y` was accidentally a perfect control: identical
config and seed to `g7r46qqr` through step 3743, yet the two diverged by up to 0.0047 from
floating-point non-determinism alone (bf16 + `torch.compile`). Two baseline seeds differ by
0.0033. **ga16's 0.0082 advantage is only ~2–3σ and has not been replicated.**

**Margin converts to wall clock** at `dL/dln(tokens) = −0.158` — about 0.0158 of loss per
10% of tokens. Under a "just get under 3.3821" objective, ga16's extra margin is worth
~9 minutes. Confirmed: shortening ga16 from 9536 to 9106 iterations landed at 3.379406
(predicted 3.3790, error 0.0004) in 143.0 min.

### Micro-batch size is worth 3.1%, free

At a constant 262,144 tokens/update the split between batch size and accumulation is a
pure throughput choice -- exact arithmetic leaves the optimization untouched:

| B | ga | ms/step | peak mem |
|---|---|---|---|
| 16 | 16 | 940.4 | 10.6 GB |
| 32 | 8 | 948.5 | 18.0 GB |
| **64** | **4** | **911.5** | 32.3 GB |
| 128 | 2 | 935.0 | 62.3 GB |

Confirmed by a full run: **3.379164 in 127.1 min** against the reference's 3.379210 in
131.3 -- statistically identical loss, 3.1% less wall clock, taking the total to **-15.0%**
versus the 149.6 min baseline. Non-monotonic (B=32 is slower than B=16), which points at
inductor kernel selection per shape rather than a smooth utilisation curve, so it is
specific to this GPU and torch version. Also needs 32 GB, so it is out of reach on the
24 GB 4090 the README benchmarks.

### Per-head attention spans: specialisation is what costs

flex_attention supports a span per head, which lets heads specialise on local context.
Benchmarked at B=64/ga=4 against SDPA's 911.5 ms/step:

| config | ms/step | vs SDPA |
|---|---|---|
| flex, all heads full | 954.6 | +43.1 |
| flex, 6 @128 + 6 global | 928.5 | +17.0 |
| flex, 10 @128 + 2 global | 911.5 | 0.0 |
| flex, **12 @128** | 902.7 | **-8.8 (-1.0%)** |
| flex, 10 @256 + 2 global | 924.2 | +12.7 |

It decomposes exactly: maximum masking saving is 51.9 ms (so attention is ~5.7% of the
step) against a fixed flex overhead of 43.1 ms, and savings are linear in the number of
short heads. Break-even is therefore `51.9 x short/12 > 43.1`, i.e. **>=10 of 12 heads
short** -- confirmed, 10 short lands on 911.5 exactly.

**So keeping any head global costs more than specialising the others saves.** The only
configuration faster than SDPA is uniform 128 for every head, worth 1.0% (~1.3 min), and
that is not specialisation at all.

Note the flex overhead is batch-dependent: +53.8 ms at B=16 versus +43.1 ms at B=64, since
larger kernels amortise it. An earlier claim here that no mask could ever beat SDPA was
derived at B=16 and is wrong at B=64.

Implementation floor: inductor requires the mask block to be divisible by the flex kernel's
128-wide tile, so mask granularity cannot go below 128. A 32- or 64-token span therefore
costs exactly what 128 costs -- it touches the same two key blocks per query block.

### Attention is ~5% of step time, so no attention optimisation can help

Longer context was hoped to raise throughput: pack 2x the tokens per micro-batch at T=2048
and cut grad accumulation to match. Measured at a constant 262,144 tokens/update:

| config | ms/step | vs baseline |
|---|---|---|
| **SDPA, T=1024, ga16** | **942.7** | — |
| flex_attention, T=1024, ga16, window 1024 | 996.5 | +5.7% |
| SDPA, T=2048, ga8 (full causal) | 1002.7 | +6.4% |
| flex_attention, T=2048, ga8, window 1024 | 1047.3 | +11.1% |

Two separable effects:

- **flex_attention costs +5.7% over SDPA at identical math** (window 1024 at T=1024 *is*
  plain causal), the Triton kernel losing to fused flash/cuDNN.
- **Windowing saves almost nothing.** T=2048 with window 1024 keeps ~1.57M of 2.10M
  attention pairs, a 25% cut, but recovered only 12.6 ms against the implied flex
  full-causal cost — **1.2% of the step**. That implies attention is only **~5% of step
  time** at 124M params and this context. MLP and projections dominate.

A 25% cut to a 5% slice cannot repay a 5.7% kernel tax, so the sliding window ends up
slower than the full-context run it was meant to fix. The premise that longer context is
nearly free holds (~6%), but there is no attention cost worth reclaiming.

A full T=2048/ga8 run confirms it end to end: **3.385583**, missing 3.3821 by 0.0035, in
2h25m — the same optimisation point at ~6% more wall clock.

`--attn_window` defaults to 0 (exact SDPA path), so existing configs are unaffected.
Equivalence checked: window 1024 at T=1024 gives step-0 val 11.010715 vs SDPA's 11.010701.

**Incidental finding:** `apply_rotary_emb` multiplies bf16 q/k by fp32 cos/sin, so q and k
leave the rotary as **fp32** while v stays bf16. SDPA is on the autocast list and silently
downcasts; flex_attention validates and rejects it. Casting cos/sin to bf16 at the source
may be a small free win, untested — it would perturb the validated baseline's numerics.

### An LR floor makes averaging work, but only heals its own wound

The horizon sweep failed because the trapezoid anneals LR to exactly zero, leaving the
average nothing to cancel. Testing that diagnosis: `--lr_final_frac 0.10` holds LR at 10%
of peak (what torch's own `SWALR` does), with lr raised to 0.0024 and four estimators
logged at once. 8276 it, 131.3 min — identical wall clock to the reference.

| | val loss | vs raw |
|---|---|---|
| raw weights | 3.392667 | — |
| **ema 0.998** | **3.382030** | **−0.0106** |
| swa (equal-weight, last 828 steps) | 3.384997 | −0.0077 |
| ema 0.999 | 3.390368 | −0.0023 |
| ema 0.9995 | 3.425420 | +0.0327 |
| *reference: lr 0.0018, no floor, no averaging* | *3.379210* | |

**The mechanism is confirmed** — averaging beat the raw weights for the first time. But
the floor plus the higher LR cost 0.0135 in raw terms, and averaging recovered only
0.0106 of it, leaving the best estimator **0.0028 behind the plain config at equal wall
clock**. A self-inflicted wound, mostly healed.

Two secondary results:

- **Equal-weight SWA lost to exponential** (3.384997 vs 3.382030), against theory. LR still
  drifts 3x across the SWA window (0.00074 -> 0.00024), so the iterate is not sampling a
  stationary ball and discounting the older, higher-LR samples is correct after all.
  Testing SWA properly needs a genuinely constant-LR tail.
- **Horizon ordering stays monotonic** (0.998 < 0.999 < 0.9995), so shorter is still better
  — but raw is now *worse* than 0.998, so there is an **interior optimum below 500 steps**.
  Without the floor the optimum was at horizon zero.

Not worth pushing: even fully recovering the 0.0028 buys ~2 minutes.

### The FFN architecture branch is closed

Five FFN variants, all at lr 0.0018 / 8276 it / warmup 444 / warmdown 3555 / ga16 / seed
42, rescaled to the baseline's 130.5 min with `dL/dln(t) = −0.158`:

| variant | wall | as-run | @130.5 m | vs baseline |
|---|---|---|---|---|
| **Baseline** `mlp` α4 n1 | 130.5 m | 3.380107 | **3.3801** | — |
| SwiGLU matched `glu` α4 n1 | 133.7 m | 3.409365 | 3.4132 | +0.033 |
| Drop-only `mlp` α4 n2 | 104.8 m | 3.458548 | 3.4239 | +0.044 |
| Similar budget `glu` α4 n2 | 106.1 m | 3.472915 | 3.4403 | +0.060 |
| α6 n2 `glu` α6 n2 | 118.3 m | 3.442892 | 3.4274 | +0.047 |

Every variant loses time-matched, and the two effects separate cleanly:

- **Dropping every other FFN costs +0.044.** It buys a genuine 20% speedup (104.8 vs
  130.5 min) that does not come close to paying for the loss given up.
- **GLU costs a further +0.016–0.033 on top**, at both n1 and n2. At α4 it is a *narrower*
  FFN (hidden 2048 vs `mlp`'s 3072 — the 2/3 GLU convention) with no speed benefit; at n1
  it is 2.4% slower per step.
- Widening GLU to α6 recovers about half the GLU penalty (+0.060 → +0.047) but never
  reaches plain drop-only, let alone the baseline.

**The α6/n2 config is capacity-limited, not tuning-limited.** Both hyperparameter probes,
run at an identical 132,416 micro-batch budget, came back null:

| probe | final val | vs α6n2 at lr 0.0018 / ga16 |
|---|---|---|
| lr 0.0028 (a 56% increase) | 3.442425 | **0.0005** — below the fixed-seed noise floor |
| ga 12, 11,035 it | 3.440474 | 0.0024 — within noise |

A 56% learning-rate increase moving the result by 0.0005 is the signature of a model that
has run out of capacity. The gap that needed closing was 0.047.

### Weight EMA is redundant with the warmdown

EMA (decay 0.998, ~500-step horizon) run *on top of* the 8276 config, evaluated on the
same data as the raw weights at every validation. The averaged weights lead comfortably
for the whole run and are then overtaken by the warmdown in the final ~200 steps:

| step | raw | EMA | EMA lead |
|---|---|---|---|
| 7168 | 3.426965 | 3.400193 | −0.0268 |
| 7680 | 3.401400 | 3.389592 | −0.0118 |
| 7936 | 3.394387 | 3.386732 | −0.0077 |
| 8064 | 3.387623 | 3.384199 | −0.0034 |
| 8192 | 3.381934 | 3.382071 | +0.0001 |
| **8276** | **3.380107** | **3.380984** | **+0.0009** |

Both are doing the same job — cancelling the gradient noise the raw iterate carries — so
stacking them buys nothing. This is the third time the same shape has appeared (see ga16,
and the batch-size hybrids): **a large lead during the stable phase is not evidence of a
final win; the warmdown is what converts progress into final loss, and it closes leads.**

EMA costs nothing to run: 946.30 ms/step vs 946.80 ms/step for the reference, +476 MiB.
It is a passive observer — swap-in/swap-out restores the parameters exactly — so the raw
trajectory is unaffected. The +0.0009 between this run's raw weights (3.380107) and the
reference (3.379210) is therefore **pure bf16/compile nondeterminism at a fixed seed**, a
tighter read on the noise floor than the 0.0033 seed-to-seed spread.

The live possibility is EMA as a **substitute** for part of the warmdown rather than an
addition: at step 7936 the average was already at 3.3867, which the raw weights did not
reach until ~step 8100. Shortening the warmdown and letting EMA supply the variance
reduction could return those tokens. Untested.

**Horizon sweep — EMA is closed, not just "0.998 is closed".** Four decays evaluated in a
single run of the 8276 config (EMA is a passive observer, so they cost only extra
evaluation passes):

| | final val | vs raw |
|---|---|---|
| **raw weights** | **3.380037** | — |
| ema 0.998 (~500 steps) | 3.380715 | +0.0007 |
| ema 0.999 (~1000) | 3.387231 | +0.0072 |
| ema 0.9995 (~2000) | 3.419085 | +0.039 |
| ema 0.9998 (~5000) | 3.573685 | +0.194 |

Strictly monotonic and steep: every lengthening of the window averages in more weights
from when the LR was high. Because the trend is monotone decreasing in horizon and the
limit as horizon -> 0 *is* the raw weights, there is no sweet spot at the short end
either. **No EMA setting beats no EMA on this schedule.**

Note the bias-correction ramp had to be fixed first: `min(decay, (step+1)/(step+10))`
does not release 0.998 until step 4490 and 0.9998 until step 45,000, so every long horizon
was clamped to the same curve. `min(decay, step/(step+1))` is a plain running mean until
the horizon is filled, releasing each decay exactly at `1/(1-decay)` steps.

## Current best

**`B=64, ga ramp 0:2/0.57:4`, 10566 iterations, warmup 507, warmdown 4543, head spans
v1, emb_aux λ=0.1, ema0.98 early stopping, lr 0.0018, seed 42**
-> first crossing ≤3.3821 at step 10368 = **112.0 min**, **−25.1% vs baseline**.
Final raw 3.378547, final ema0.98 3.378646.

Previous best without aux/spans: **`B=64, ga=4`, 8276 iterations, warmup 444, warmdown
3555, lr 0.0018, seed 42** -> **3.379164 in 127.1 min**, **-15.0% wall clock**.

```
--batch_size 64 --sequence_length 1024 --grad_accumulation_steps 4 \
--num_iterations 8276 --warmup_iters 444 --warmdown_iters 3555 \
--learning_rate 0.0018 --weight_decay 0.1 --seed 42
```

Three changes from stock, all existing knobs:

| change | worth |
|---|---|
| tokens/update 524,288 -> **262,144** (ga 32 -> 16) | -0.0082 loss |
| warmdown 21.5% -> **43%** of tokens | -0.0099 loss |
| micro-batch B=16 -> **64** at the same tokens/update | -3.1% time, loss-neutral |

The first two bought ~0.018 of loss margin, which the token-scaling law
`dL/dln(tokens) = -0.158` converts into ~12% less wall clock by shortening 9536 -> 8276
iterations. The third is pure throughput and stacks on top.

Replication: the B=16 form of this config was run at two seeds, 3.379210 and 3.379601, a
spread of 0.0004. Fixed-seed nondeterminism is ~0.0009. The margin against 3.3821 is
~0.003, thin but outside both.

Conservative fallback: **8549 it, warmup 459, warmdown 3672 -> 3.373880**, margin 0.0082,
comfortably outside any noise estimate. Single seed, B=16.

---|---|---|
| 42 | 3.379210 | 131.3 m |
| 43 | 3.379601 | 130.8 m |

Both clear 3.3821. Versus the baseline's 3.379890 in 149.6 min: **−12.4% wall clock**.
Reproduce with:

```
--batch_size 16 --sequence_length 1024 --grad_accumulation_steps 16 \
--num_iterations 8276 --warmup_iters 444 --warmdown_iters 3555 \
--learning_rate 0.0018 --weight_decay 0.1 --seed 42
```

The two seeds differ by **0.0004** — far tighter than the 0.0033 spread measured between
baseline seeds. The margin against 3.3821 is only ~0.0027, but the *seed* variance at this
config is evidently much smaller than that, so the thin margin is not the liability it
looked like from one sample.

Conservative fallback: **8549 it, warmup 459, warmdown 3672 → 3.373880 in 134.4 min
(−10.2%)**, margin 0.0082, comfortably outside any noise estimate. Single seed.

All shortened runs were sized by the token-scaling model `dL/dln(tokens) = −0.158`, which
has now predicted five shortened runs to within 0.005, and three to within 0.001. Notably
it errs *conservatively* — several runs landed **below** their aim, never meaningfully
above.

### Embedding auxiliary loss sweep

An auxiliary loss `L_aux = λ * ||h_last - E[y_true]||²` pulling the final hidden state
toward the (detached) target embedding. Computed inside forward(), mixed in the training
loop so torch.compile sees a static graph. All runs on the "fastest schedule":
B=64, ga ramp `0:2,0.57:4`, 10566 it, warmup 571, warmdown 4543, head spans v1
(`0:64,1:64,2:64,3:64,4:128,5:128`), tile 64, lr 0.0018, ema 0.98/0.97/0.96, seed 42.

**No-aux baseline** on this schedule: raw **3.379695**, ema0.98 **3.379784**, first
crossing ≤3.3821 at step **10496** (raw 3.380668).

| wandb id | config | final raw | final ema0.98 | crossing step | crossing metric | margin |
|---|---|---|---|---|---|---|
| — | no-aux baseline | 3.379695 | 3.379784 | 10496 | raw 3.380668 | — |
| `8k3ypxde` | **λ=0.1** | **3.378547** | 3.378646 | **10368** | ema0.98 3.381657 | 0.000443 |
| `6kydrgzc` | λ=0.5 | 3.380636 | 3.380776 | 10496 | ema0.98 3.381569 | — |
| `b4y4fp0c` | λ=0.2 | 3.379389 | 3.379483 | 10496 | ema0.98 3.380196 | — |
| `krubhck1` | λ=0.05 | 3.379479 | 3.379594 | 10496 | ema0.98 3.380306 | — |
| `0p6a2i73` | λ=0.025 | 3.378366 | 3.378449 | **10368** | ema0.98 3.381524 | 0.000576 |
| `v9mabzvc` | λ=0.015 | 3.378324 | 3.378419 | **10368** | ema0.98 3.381528 | 0.000572 |
| `f3iqg0yo` | λ=0.1→0.01 (linear) | 3.379215 | 3.379299 | 10496 | ema0.98 3.379976 | — |

Cosine aux mode (λ=0.1) and bottleneck-512 variants were killed early — cosine diverged,
bottleneck added parameters and was far worse.

**Key findings:**

- **All constant lambdas ≤0.2 improve final raw loss** vs no-aux, but the effect is small
  (0.0003–0.0012) and non-monotonic: 0.1 and 0.015–0.025 are best, 0.05 is oddly weak.
- **Crossing time improves by 128 steps** (10496→10368, ~112 min vs ~115 min) for
  λ∈{0.1, 0.025, 0.015}. All cross via ema0.98 at step 10368; eval granularity was 128.
- **λ=0.1 has the widest crossing margin** (0.000443 vs ~0.00057 for small lambdas),
  making it the most robust choice.
- **Linear annealing (0.1→0.01) did not help** — crossed later (10496) than constant 0.1.
- **Higher lambda helps early but hurts late.** λ=0.2 leads until ~70% of training then
  falls behind. This is consistent with the aux loss conflicting with the LR warmdown.

### Warmup reduction experiments

Testing whether shorter warmup saves wall clock, using λ=0.025 and eval_every=64 for
finer crossing resolution. All on the same fastest schedule except as noted.

| wandb id | config delta | final raw | final ema0.98 | crossing step | crossing metric |
|---|---|---|---|---|---|
| `c3tdsdgq` | **warmup 507** (−64), iters 10566 | **3.378073** | 3.378188 | **10368** | ema0.98 3.381347 |
| `7kwb7lj0` | warmup 507, **iters 10502** (−64) | 3.380663 | 3.380841 | 10432 | raw 3.381626 |
| `umotv3ru` | warmup 507, iters 10502, **ga 0.5735** | (killed at 10048) | — | — | — |

**Key findings:**

- **Shorter warmup (507 vs 571) at full iteration count is neutral to slightly positive**
  — final raw improved by 0.0003, same crossing step.
- **Removing 64 iterations hurt** — crossed 64 steps later (10432 vs 10368) and worse
  final loss (3.380663 vs 3.378073). The token reduction cost more than the warmup saved.
- The finer eval granularity (every 64 vs 128) confirmed no crossing between 10304 and
  10368 — the true crossing is between 10304 and 10368 for the best configs.
- Adjusting the GA transition fraction to compensate for shorter total (0.5735 vs 0.57)
  was tracking ~0.0008 behind and killed early.

---

## Untested

- **Warmdown beyond 43%** — swept 2048/3072/4096; gains halving, so likely near optimal.
- **Weight EMA as a substitute for part of the warmdown.** Stacking it on top is now
  tested and is a no-op (above). Shortening the warmdown and leaning on the average
  instead is the version that could still return tokens.
- A replicate of ga16 at a second seed, which its 2–3σ result needs. (The *shortened*
  8276 config has now been replicated — 0.0004 spread — but the ga16-vs-ga32 comparison
  at full compute is still one seed each.)
- A second seed for the 8549 fallback config.
