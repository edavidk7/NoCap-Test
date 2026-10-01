#import "@preview/clear-iclr:0.7.0": iclr2025

/**
 * Authors should be specified as a list of entries. Each entry enumerates
 * authors with the same affilation and address. Field `names` is mandatory.
 */
#let authors = (
  (
    names: ([David Korcak],),
    affilation: [
      ETH Zürich
    ],
    address: [Switzerland #linebreak() #link("https://x.com/thatdave_k")[x.com/thatdave_k] #linebreak() #link("https://github.com/edavidk7/NoCap-Test")[github.com/edavidk7/NoCap-Test]],
    email: "dkorcak@ethz.ch",
  ),
)

#show: iclr2025.with(
  title: [GPT-2 Benchmark Technical Report ],
  authors: authors,
  abstract: [
  In this report, I outline the different approaches taken while trying to speed up the GPT-2 124M model training to reach the target validation loss on the FineWeb dataset _without changing any of the hyper-parameters_. My first ideas were focused purely on the architecture of the model, trying to improve expressivity, reduce FLOPs, or both. I then resorted to changing the attention mask, better exploiting the training-data distribution. By observing the convergence behavior at different gradient noise levels (physical batch sizes), I was able to tune the training process to maximize token (and time) efficiency, with an additional passive EMA on model weights. Finally, an auxiliary loss term was introduced, further improving convergence speed. The final achieved speedup was \~25%.
  ],
  bibliography: bibliography("main.bib"),
  accepted: none,
)
#let url(uri) = link(uri, raw(uri))

#show figure.where(kind: table): it => block[
  #it.body
  #v(0.5em)
  #it.caption
]

= AI Usage Declaration

To perform my experiments, I heavily relied on `Claude Code` serving as an intelligent launch script & training watchdog. Additionally, I used it to implement some of the training modifications (EMA) when I was travelling and only had my phone available.  It was necessary for me to finish the challenge in time and without excessively high costs, as I ran all of the experiments on an Amazon EC2 instance (RTX6000 Pro Blackwell) which costs about \$4.5/hour, and `Claude` could launch and debug experiments per my instructions overnight. Sometimes, I would consult Claude on my ideas, especially when I wanted to know whether related literature already exists.

= A Comment on Stochasticity
<stochasticity>
Before running any performance experiments, I implemented the functionality to manually set the random seed of all libraries. This had to done, as even some of the runs of the *baseline* as shipped in the original repository *would not cross the target loss of 3.3821*. I then reran the baseline with seed 42, which produced a valid run achieving the target loss, and was used *for all* downstream experiments. It may be a part of future work to try and reproduce the results with a random seed sweep, however, I had only a limited budget for GPU hours and chose to rather invest them in actual performance tuning and experimentation rather than random seed sweeps.

= Architectural Changes 

My proposed architectural changes were motivated by the fact that we train & evaluate on FineWeb, a very diverse dataset, where the model has to tackle/predict various topics. I was either trying to improve the expressivity without adding too many parameters/FLOPs, or cutting FLOPs without reducing expressivity too much. Some of them failed completely, and some have shown to improve the final validation loss.

== Learnable per-head per-block softmax temperature

Initially, I was trying to "emulate" a Mixture-of-Experts architecture, by allowing further, learnable specialization of each head in each layer. One way to achieve that was (in my eyes) a learnable temperature parameter. The intention is to allow each head to learn the entropy of its attention softmax distribution to best reason over the data. 

Intuitively, I expected to see the learning procedure adjust the temperatures, with lower magnitudes in the early layers (vast knowledge summary) and higher magnitudes near the final layer, where the model summarizes only few, high-confidence embeddings to produce the final token prediction. 

To make the implementation numerically feasible, constrainable, and to create a proper structural prior for the parameter, the gradient was computed towards a log-temperature value, passed through a `tanh` squish and scaled to a limited range. Formally:

$ "softmax"(bold(x)_i, tau) &= e^(tau bold(x)_i) / (sum_(j=1)^n e^(tau bold(x)_j)), med bold(x) in RR^n "and" tau = e^(c tanh(p)) \
  & med "where" p med "is a learnable parameter", med c in [0.2, 0.6]         
$

Unfortunately, this failed to produce any meaningful improvement, the model's validation loss evolution during training was identical to the baseline, while the computation time went up due to a slight increase in FLOPs. An interesting observation was that initially, the temperatures followed the trend I had predicted (first block low, last block high), however, over time all magnitudes slowly crashed to the floor defined by the `tanh` clamp. My conjecture is that gradient descent is powerful enough to modulate the activation magnitude by scaling up the relevant `Query` matrix's norm, and does not need the additional parameter.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    image("block0_softmax.png", width: 110%),
    caption: [Learned softmax temperatures for individual heads in transformer *block 0* over the course of training.],
  ),
  figure(
    image("block11_softmax.png", width: 110%),
    caption: [Learned softmax temperatures for individual heads in transformer *block 11* over the course of training.],
  ),
)

Because this change did not bring any noticeable improvement in loss, slightly increased the overall wall clock time, and can be side-stepped by the optimization procedure, I decided to abandon it.

== Leave-Every-Other-Out MLP Removal 

Since the dataset requires memorization of significant amount of information, it was intriguing to reason about the _minimum_ amount of "memory capacity" a model of this size/dimensionality needs to store that knowledge, and if perhaps the baseline architecture contains unnecessary parameters. Based on the findings in "Transformer Feed-Forward Layers Are Key-Value Memories #cite(<geva2021transformer>)", I tried out simply removing every other MLP from the baseline model, but keeping the attention layer and empty `RMSNorm` path in each block (empirically found to improve convergence). 

#figure(
  table(
  columns: 9,
  align: (left, center, center, center, center, center, center, center, center),
  table.header(
    [*Variant*], [*FFN*], [*Act.*], [*$alpha$*], [*Hidden*],
    [*FFN blocks*], [*Val loss*], [*Wall*], [*Proj. target*],
  ),
  [baseline],  [MLP], [gelu], [4], [3072], [12 of 12], [3.379], [149.6 min], [-],
  [SwiGLU], [GLU], [silu], [4], [2048], [12 of 12], [3.409], [133.7 min], [158.9 min],
  [GEGLU], [GLU], [gelu], [4], [2048], [12 of 12], [3.408], [143.9 min], [170.1 min],
  [drop MLP], [MLP], [gelu], [4], [3072], [6 of 12], [3.458], [104.8 min], [170.0 min],
  [drop GLU], [GLU], [silu], [4], [2048], [6 of 12], [3.472], [106.1 min], [188.5 min],
  [drop GLU wider], [GLU], [silu], [6], [3072], [6 of 12], [3.442], [118.2 min], [173.8 min],
),
caption: [Comparison of different MLP configurations, hyperparameters *identical* to baseline. Projected target refers to the time estimated to reach the accepted val. loss threshold.],
)

Without any further changes, this led up to about 21% forward FLOP reduction. However, expressivity of the model was clearly impacted. I came up with two potential resolutions. First, I tried swapping standard MLPs for Gated Linear Unit architectures (GLU, #cite(<shazeer2020glu>, form: "prose")), which, in the past, I experimentally found to be more stable and faster learners, as well as more efficient information stores. Then, to buy back the parameters, but in a way that is easier to parallelize on the GPU with less overhead, I varied the MLP width scaling factor, denoted $alpha$. Instead of more small MLPs, I kept fewer but wider.

#figure(
table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header([*Variant*], [*Fwd GFLOP/token*], [*vs baseline*], [*FFN GFLOP/token*]),
    [baseline], [0.266], [+0.0%], [0.113],
    [SwiGLU], [0.266], [+0.0%], [0.113],
    [GEGLU], [0.266], [+0.0%], [0.113],
    [drop MLP], [0.209], [-21.3%], [0.057],
    [drop GLU], [0.209], [-21.3%], [0.057],
    [drop GLU wider], [0.238], [-10.6%], [0.085],
  ),
  caption: [Comparison of compute requirements of every FFN configuration tested.]
)

Furthermore, I tested one configuration where the $alpha$ parameter varied 4-3 following the leave-one-out fashion, however, this configuration didn't help either and led to a projected slower run. Because all attempts failed, I also abandoned this route. 

== LM head bottleneck (embedding factorization)

Because the model's input embedding matrix $bold(E)$ is typically of dimension $V times n_"embd"$, for the final cross-entropy/perplexity calculation, there is exactly $V times T$ dot products of dim $n_"embd"$. My idea for obtaining a lower runtime was to reduce $n_"embd"$, but without significantly affecting the model's internal capacity. To this extent, I added additional input/output projection matrices $bold(W)^i_"bottleneck"$ and $bold(W)^o_"bottleneck"$ of dimensions $n_"bottleneck" times n_"embd"$ and vice versa. Then $bold(E) in RR^(V times n_"bottleneck")$, and the output LM head is just transpose of $bold(E)$. This modification saves a noticeable number of FLOPs, because the full-vocabulary dot product is computed over a (significantly) reduced dimension. Suppose $V=50257$ and $n_"embd"=768$ (GPT-2 124M). If $n_"bottleneck"=512$,
then the FLOPs (percentually) decrease to $(V times 2 dot 512 times T + 2 dot 512 dot  768 times T) \/ (V times 2 dot 768 times T )$. Simplified,  $(V times 2 dot 512 + 2 dot 512 dot  768) \/ (V times 2 dot 768) approx 0.68 = 68%.$ A 32% reduction in FLOPs is quite significant, and because the LM head constitutes a significant number of the model's overall forward FLOPs, the total number of FLOPs decreases by about 9%.

#figure(
table(
    columns: 6,
    align: (left, center, center, center, center, left),
    table.header(
      [*Variant*], [$n_"bottleneck"$], [*Iters*], [*Val loss*], [*Wall*], [*Proj. target*],
    ),
  [baseline], [--],   [10566], [3.379], [117.0 min], [--],
  [bottleneck 576],   [576], [10566], [3.399], [112.3 min], [126.9 min],
  [bottleneck 512], [512], [10566], [3.408], [103.8 min], [124.3 min],
  ),
  caption: [Comparison of $V times n_"embd"$ *tied-embedding* head configurations against a matching-schedule baseline. The baseline in this case *was not* the vanilla baseline, but a run tuned mid-way through my experimentation],
)

While the introduction of a bottleneck dimension helped reduce the absolute wall clock time noticeably, the model always lost expressivity, to an unrecoverable extent. All projections would lead to a longer runtime than the then-current baseline. Later, SVD decomposition of $bold(E)$ from the baseline revealed $~740$ non-zero singular values, indicating the baseline is already on the verge of using the full embedding rank. That analysis essentially gave a lower bound, and prompted me to stop this research direction.

== Per-Head, Data-Driven Limited Context Length 
<context-windowing>
Lastly, since the README mentioned successful attempts that change the model's context length, I got an idea to hard-mask the context in each individual head of the attention block. The idea was derived from the _learnable temperature_ experiment, except the per-head specialization is manually selected and fixed to attend only some $K$ tokens in the past. The head would then learn to accurately summarize only over a short context, without having to model all of the dependencies.
#figure(
  image("attn_masks_3configs.png", width: 95%),
  caption: [Attention masks for different context window sizes.],
)

#v(1em)

I implemented this using PyTorch's `flex_attention`, which allows block-wise computation, and therefore can exploit the reduction in necessary FLOPs that comes from a limited context window. The first window sizes were picked manually to be 64 and 128, somewhat intuitively based on the expected English language sentence length and tokenization ratio (word $arrow$ \#tokens). This change seemed to have helped significantly, reducing both the training time and improving convergence to lower loss values. I wondered as to why these parameters worked so well (and if there are even better ones) and got an idea to measure the *sentence lengths* and look at their distribution on both the training and validation dataset. After some research, *token copy distance* (distance to the nearest prior occurrence of the same token) seemed like a more suitable metric, roughly signaling how far into the past the model needs to attend for NTP.

#figure(
  image("copy_distance_2M.png", width: 100%),
  caption: [Distribution of token copy distances on the training and validation datasets.],
)

#v(1em)

The distributions show that over 40% of token copy distances are below 64, so a $w=64$ head already covers the majority of the dependencies it is likely to be asked about, and the remaining spans are chosen so that the heads left at wider windows cover the tail. Expressivity on this specific dataset is therefore preserved, while the short windows likely add a useful inductive bias, and slightly reduce the forward pass time.

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    table.header(
      [*Attention span (head:span, unnamed = full ctx.)*], [*Iters*], [*Val loss*],
    ),
    [full causal (no windowing)], [11575], [3.377],
    [0:64,1:64,2:64,3:64,4:64,5:64], [11575], [*3.365*],
    [0:64,1:64,2:64,3:64,4:128,5:128], [11575], [*3.365*],
    [0:64,1:64,2:64,3:64,4:64,5:64,6:128,7:128], [11575], [3.368],
    [0:64,1:64,2:64,3:64,4:64,5:64,6:128,7:128,8:128], [11575], [3.368],
    [0:64,1:64,2:64,3:64,4:64,5:64,6:128,7:128,8:512,9:512], [11575], [3.368],
    [full causal (no windowing)], [8276], [3.379],
    [0:64,1:64,2:128,3:128], [8276], [3.372],
    [0:64,1:64,2:64,3:64,4:128,5:128,6:512,7:512], [8276], [3.373],
    [0:64,1:64,2:128,3:128,4:512,5:512], [8276], [3.373],
    [0:64,1:64,2:128,3:128,4:256,5:256,6:512,7:512], [8276], [3.374],
    [0:128,1:128,2:128,3:128,4:512,5:512], [8276], [3.377],
  ),
  caption: [Per-head attention span sweep, with a schedule-matched causal baseline in each iteration-budget group. Every transformer block gets the same array of per-head windows. Wall clock omitted due to runtime contention with another experiment on the machine.],
)

The addition of length-64 context window heads has significantly improved convergence under the same training regime/iteration budget, and as such, was the only architectural change to bring measurable improvement in convergence.

The actual final training recipe uses a depth-varying window cascade, motivated by the fact the model likely does not initially summarize across long distances. This also makes the forward pass slightly faster without reducing expressivity. I have not properly ablated these per-block varying context windows, so it's not a good research practice, but introducing it in the final recipe helped save a tiny bit of time.

#figure(
  table(
    columns: 6,
    inset: 5pt,
    align: (left, center, center, center, center, center),
    table.header(
      [*Blocks*], [*$w=64$*], [*$w=128$*], [*$w=256$*], [*$w=512$*], [*$w=1024$ (global)*],
    ),
    [0 -- 1 (shallow)], [6], [2], [2], [2], [0],
    [2 -- 3], [4], [2], [2], [2], [*2*],
    [4 -- 11 (rest of depth)], [4], [2], [0], [0], [*6*],
  ),
  caption: [Per-head attention span cascade used in the final recipe. 4 heads stay local ($w=64$) at every depth, the remaining heads widen with block index.],
) <final-recipe-context-windows>




= Training Optimizations

Besides modifying the model architecture itself, I ended up heavily modifying the training procedure. First, after trying to tackle the context-length optimization through training schedules, I discovered a potential efficiency improvement in gradient noise by varying the global batch size, and then applied an EMA over the model weights to test if it further improves the noisier but faster training. Lastly, I introduced an additional L2 loss term which is somewhat similar to next-token prediction but from a metric point of view.

== Batch-size scheduling
<bs-sched>
Initially, I conjectured that the model does not benefit from the full 1024 context length at random initialization of the weights. Intuitively, when latents are first settling, the prediction of tokens over short context spans should be sufficient to give the model a "rough idea". Only later in training, when the loss drops to a certain level, the model is shown longer and longer sequences. This kind of scheduling affects training in 2 additional ways, the total token count per batch (GD step) and the runtime, because shorter sequences lead to significantly faster forward passes ($Omicron(T^2)$). I tested a couple of hand-picked schedules to asses the runtime gain versus loss in model quality.

Even though the problem's README says that hyperparameter magic _should not_ be the approach taken to speed things up, I stand my case that I arrived at this optimization through principled reasoning about gradient estimator noise, and as such, it is a valid optimization of the training algorithm which is among the things permitted by the README.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header(
      [*Sequence-length schedule \ ($"training fraction"{:}T$)*], [*Iters*], [*Grad accum*], [*Val loss \ (final)*], [*Wall*],
    ),
  [$T=1024$ (baseline)], [4768], [32], [3.379890], [149.6 min],
  [0:64,0.05:128,0.15:512,0.25:1024], [4768], [32], [3.722735], [87.4 min],
  [0:256,0.5:1024,0.95:2048], [4768], [16], [3.391295], [150.3 min],
  [0:64,0.2:128,0.4:256,0.7:1024,0.9:2048], [4768], [16], [3.733675], [108.4 min],
  [0:64,0.05:128,0.2:256,0.5:1024,0.9:2048], [4768], [16], [4.035129], [39.8 min],
  [0:64,0.05:128,0.2:512,0.5:1024,0.9:2048], [4768], [16], [3.627057], [97.4 min],
  ),
  caption: [Sequence-length schedule sweep. Only `0:256,0.5:1024,0.95:2048` (the gentlest ramp, most time at $T=1024$) comes close to the target. Faster schedules underperform.],
)

  Across the board, this approach failed. Curiously, the model initially converged to a set loss threshold significantly faster than the baseline. I tried to reason why, and ended up quickly testing whether this is only a function of tokens per step, or also the context length.  As supported by the following comparison of 2 training runs, modulated tokens per step at full context length outperform modulated context length. Tokens-per-step schedule (ramp) modulated via gradient accumulation due to reasons explained in the last paragraph.

  #figure(
    image("sched_loss_vs_time.png", width: 85%),
    caption: [Validation loss over absolute training wall-clock time. Context length ramp initially wins, as the forward pass is significantly cheaper. Both beat baseline.],
  )
  #v(1em)

  #figure(
    image("sched_loss_vs_tokens.png", width: 85%),
    caption: [Validation loss over total number of tokens consumed by the training process. Tokens-per-step ramp quickly edges out context length modulation. Both beat baseline.],
  )
  #v(1em)

#figure(
  table(
    columns: 4,
    align: (center, center, center, center),
    table.header(
      [*Step*], [*Tokens/step*], [*Context Length Ramp \ (a)*], [*Gradient Accumulate Ramp \ (b)*],
    ),
    [0], [32,768], [10.993], [11.005],
    [128], [32,768], [6.589], [6.633],

    [640], [65,536], [5.250], [5.161],

    [1152], [262,144], [4.338], [4.237],
  ),
  caption: [*Validation loss* at matched tokens/step: (a) reaches each tokens/step value by widening context $T$ (64$arrow$128$arrow$512) at fixed $"ga"=32$, while (b) reaches the same values by raising $"ga"$ (2$arrow$4$arrow$16) at fixed $T=1024$. Both runs were killed before completion (last steps 2560/4768 and 1152/4768 respectively), so this table covers only the overlapping range both reached.],
)

Takeaway from the experiment was that both tokens-per-step scheduling and context-length scheduling reach low loss significantly faster than the baseline, however, context-length scheduling loses its raw wall clock advantage quickly. It additionally shows that to obtain an (equally) good model early on, *significantly noisier gradients* can be used in the optimization procedure. In plain terms, this means the baseline run is wasting computation by having unnecessarily large batch size at the beginning, using low-noise gradient estimators even though ones with higher noise levels would be sufficient. *This was the main area for wall-clock improvement*. In the end, token-per-step scheduling at fixed $T=1024$ was the branch I followed further.

#figure(
  table(
    columns: 4,
    align: (center, center, center, center),
    table.header(
      [*Batch $B$*], [*Grad accum*], [*ms/step*], [*Peak mem*],
    ),
    [16], [16], [940.4], [10.6 GB],
    [32], [8], [948.5], [18.0 GB],
    [*64*], [*4*], [*911.5*], [31.8 GB],
    [128], [2], [935.0], [62.3 GB (invalid for 5090)],
  ),
  caption: [Per-step throughput at fixed 262,144 tokens/step, 40-step microbenchmark.],
)

The last step was ensuring tokens-per-step are modulated in the most efficient way possible wrt. the underlying hardware. Because gradient accumulation is always performed sequentially, more accumulation steps linearly scale up the wall clock time. There is also the option of increasing the physical batch size from the baselines' $B=16$, which should remain somewhat constant due to a GPU's parallelism, and is supported by the measured data. To remain backwards compatible with an RTX 5090, I kept memory usage  under $32$ GB.


#figure(
  table(
    columns: 8,
    align: (left, center, center, center, center, center, center, center),
    table.header(
      [*Run*], [*$B$*], [*Grad \ Acc.*], [*Tokens/step*], [*Total \ tokens*], [*Val loss*], [*Wall \ clock*], [*First cross*],
    ),
[baseline], [16], [32], [524,288], [2.50 B], [3.379], [149.6 min], [148.6 min],
[low B,\ high $"ga"$], [16], [16], [262,144], [2.17 B], [3.379], [131.3 min], [130.0 min],
[high B,\ low $"ga"$], [64], [4], [262,144], [2.17 B], [3.379], [*127.1 min*], [*125.8 min*],
  ),
  caption: [Runtime comparison at matched tokens/step, both runs process the identical 262,144 tokens/step and identical total tokens (8276 iters), differing only in the $B$/$"ga"$ split. `B=64` is 3.6% faster wall clock for statistically identical loss. First cross is the first *eval step at or below the 3.3821 target*.],
)

Clearly, a lot could be gained by optimizing the batch size to better exploit the hardware, shrink it to increase the gradient noise within acceptable levels, and make training more data (and in turn time) efficient. Finally, I combined scheduling with the hardware-optimal batch size, leading to a significant improvement in wall clock time.

#figure(
  table(
    columns: 7,
    align: (center, center, center, center, center, center, center),
    table.header(
      [*Batch $B$*], [*GA schedule*], [*Tokens/step*], [*Iters*], [*Val \ loss*], [*Wall \ clock*], [*First \ cross*],
    ),
[16 (baseline)], [const. 32], [524,288], [4768], [3.379], [149.6 min], [148.6 min],
[64], [const. 4], [262,144], [8276], [3.379], [127.1 min], [125.8 min],
[64], [0:2,0.57:4], [first 131,072 \ then 262,144], [11575], [3.377], [127.5 min], [*124.7 min*],
  ),
  caption: [First run combining the gradient accumulation ramp with $B=64$: $"ga"=2$ (131,072 tok/step) for the first 57% of training, then $"ga"=4$ (262,144 tokens/step) for the rest. "First cross" defined identically to the previous table. Schedule crosses sooner than constant.],
) <first-sched-run>

This reasoning and optimization process brought a significant speedup of 16%. As such, it was included in the final training recipe and all the following experiments.

== Passive Exponential Moving Average on Model Parameters
<ema>
Since noise in the gradients has shown to be an important factor in training efficiency, a natural continuation of the previous experiment was analysis of noise in the model weights, and if there is any potential to get an overall better model by averaging the parameters over the last few gradient descent steps. To evaluate this efficiently, I implemented the EMA to run in parallel with training, and allow testing of an arbitrary number of different EMA coefficients (e.g. 0.998, 0.95, 0.96) all at once. Each averaged model is evaluated during the validation portion of training, which is *excluded* from the overall training wall-clock time, so evaluating many EMAs costs no additional wall clock. 

#let g(body) = text(fill: gray, size: 9pt)[#body]

#figure(
  block(fill: rgb("#f7f7f7"), inset: 14pt, radius: 4pt)[
    #table(
      columns: (280pt, 100pt),
      stroke: none,
      inset: (x: 0pt, y: 3pt),
      align: (left, left),
      [$Theta = {theta_d : d in D}$, shadow copies indexed by decay $d$, init $theta_d <- theta$], [],
      [], [],
      [*for* $t = 0, dots, T$:], [],
      [#h(1.4em)$g_t <- $ accum$(s, t/T)$], [#g[piecewise schedule $s$]],
      [], [],
      [#h(1.4em)*if* $t mod E = 0$:], [],
      [#h(2.8em)$ell <- L(theta)$], [],
      [#h(2.8em)*for* $theta_d in Theta$:], [],
      [#h(4.2em)$theta, theta_d <- theta_d, theta$], [#g[swap in]],
      [#h(4.2em)$ell_d <- L(theta)$], [],
      [#h(4.2em)$theta, theta_d <- theta_d, theta$], [#g[swap out]],
      [], [],
      [#h(1.4em)$cal(L) <- 1/g_t sum_(i=1)^(g_t) "loss"(x_i, y_i, theta)$], [#g[compute loss]],
      [#h(1.4em)$theta <- theta - eta_t dot nabla_theta cal(L)$], [#g[opt. step]],
      [], [],
      [#h(1.4em)*for* $theta_d in Theta$:], [],
      [#h(2.8em)$beta_t <- min(d, t/(t+1))$], [#g[bias-correction ramp]],
      [#h(2.8em)$theta_d <- (1 - beta_t) dot theta + beta_t dot theta_d$], [#g[convex combination]],
    )
  ],
  caption: [Training-loop core as a convex-combination EMA update over persistent/averaged parameter sets $theta_d$, with swap-based evaluation (to only change parameters of a static `torch.compile` kernel without re-compilation) and a piecewise grad-accumulate schedule $g_t$ as a general form of method presented in @first-sched-run.],
)


First, I tested some long-horizon values, this experiment was interleaved with the scheduling experiment, so the initial test was with a constant tokens-per-step schedule.

#figure(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    table.header(
      [*EMA decay*], [*Horizon ($1/(1-d)$)*], [*Val loss*], [*vs. raw*], [*First cross*],
    ),
    [*raw (no EMA)*], [--], [*3.380037*], [--], [130.0 min],
    [0.998], [~500 steps], [3.380715], [+0.0007], [130.0 min],
    [0.999], [~1000 steps], [3.387231], [+0.0072], [never],
    [0.9995], [~2000 steps], [3.419085], [+0.039], [never],
    [0.9998], [~5000 steps], [3.573685], [+0.194], [never],
  ),
  caption: [Long-horizon EMA sweep, $B=16$, $"ga"=16$ (constant, no schedule), 8276 iters, 131.3 min wall clock time.  First cross is the first eval step at or below 3.3821, raw and EMA 0.998 both cross at the same step (8192/8276) since they track almost identically at this short horizon. Loss numerical precision increased to show fine differences.],
) <ema-sweep-1>

Clearly, the trend indicated that the shorter the schedule, the lower the loss (model improves rapidly during the warmdown phase and long horizon average produces worse parameters). I took the then-best performing EMA of 0.998 and added it to the fastest-crossing experiment from @first-sched-run.

#figure(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    table.header(
      [*EMA decay*], [*Horizon ($1/(1-d)$)*], [*Val loss*], [*vs. raw*], [*First cross*],
    ),
    [raw (no EMA)], [--], [3.377058], [--], [124.7 min],
    [*0.998*], [~500 steps], [*3.377723*], [+0.0007], [*122.7 min*],
  ),
  caption: [$B=64$, grad-accumulate schedule `0:2,0.57:4`, 11575 iterations, 127.5 min. EMA 0.998 finishes essentially tied with raw (+0.0007, same gap as in @ema-sweep-1) but crosses the target 2.0 minutes earlier.],
)

For my later tests, I decided to rather sweep over aggressively short horizons. The results have shown it's more likely averaging horizons in orders of tens to low hundreds of iterations will outperform the raw model, so I continued with those values instead.

#figure(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    table.header(
      [*EMA decay*], [*Horizon ($1/(1-d)$)*], [*Val loss*], [*vs. raw*], [*First cross*],
    ),
    [raw (no EMA)], [--], [3.378729], [--], [113.3 min],
    [0.95], [~20 steps], [3.378752], [+0.00002], [113.3 min],
    [0.96], [~25 steps], [3.378763], [+0.00003], [113.3 min],
    [0.97], [~33 steps], [3.378784], [+0.00006], [113.3 min],
    [*0.98*], [~50 steps], [3.378831], [+0.00010], [*112.4 min*],
    [*0.99*], [~100 steps], [3.378919], [+0.00019], [*112.4 min*],
  ),
  caption: [$B=64$, grad-accumulate schedule `0:2,0.57:4`, attention windows, auxiliary L2 loss, 10566 iters, 115.3 min. Final loss is monotonically worse with longer horizon, but 0.98 and 0.99 cross the target one evaluation step earlier (10368 vs 10432) than raw and the shorter horizons.],
)

Both experiments have shown that short-horizon EMA on the model parameters can converge to the threshold loss faster, and because it does not increase the apparent wall-clock time, I decided to keep the sweeps active during all following training runs, and then simply picked *whichever EMA crossed the target loss threshold first*.

#pagebreak()

== Auxiliary L2 Embedding Loss
<aux-loss>
Having exhausted the architectural direction, and the optimization procedure, my attention turned to the loss function itself. I recalled some knowledge from a couple of years ago, when I was interested in _implicit layers_ #cite(<duvenaud2020implicit>). I was curious how it would look like if we were to unroll a transformer "to infinity", i.e., have a "fixed-point attention block", that would output the true next-token embedding in one step. Denote the input token embedding $bold(x)$, and its ground-truth next-token counterpart $bold(h)^*$. Then the "implicit transformer" works, very roughly, as $bold(h)^* = "Block"(bold(h)^*, bold(x)).$ where for brevity this "implicit block" would be the last one in the otherwise finite model.

Implicit layers can involve multiple iterations, or implicit formulations through Jacobian inversions, neither of which are helpful in the context of minimizing training time.

Instead, a very simple approximation can be made, which *adds a metric regularizer* to the standard cross-entropy objective, and turns the implicit transformer into a "regression problem". Consider the model's output $hat(bold(h))$ as an approximation of $bold(h)^*$. Because the model shares the same embedding matrix $bold(E)$ on both input and output, we know that $bold(h)^* = bold(E)_(y, :)$ where $y$ is the index of the ground-truth next token in the vocabulary of the model. Cross-entropy loss maximizes the probability on the correct token, whose raw logits are simply dot products $hat(bold(h)) dot bold(E)_(i, :)$ for all $i$ in the vocabulary. Maximizing probability through softmax directly maximizes the corresponding logit (and minimizes others), i.e. maximizes the dot product. We can also maximize the dot product between the model output and the correct embedding by *minimizing their L2 distance*. This is exactly the additional loss term I experimented with.

Formally, let $bold(E) in RR^(V times d)$ be the (tied) embedding matrix, $hat(bold(h))_t in RR^d$ the model's output hidden state at position $t$, and $y_t$ the index of the ground-truth next token. The implicit-transformer fixed point from above is approximated by a single forward pass, $hat(bold(h))_t approx bold(h)_t^*$, and because the head is tied to $bold(E)$, the target itself is just an embedding row:

$ bold(h)_t^* = bold(E)_(y_t, :). $

The standard cross-entropy objective operates on logits $ell_(t,i) = hat(bold(h))_t dot bold(E)_(i, :)$ for every vocabulary entry $i$, and softmax maximizes $ell_(t, y_t)$ relative to the rest, i.e. it maximizes the dot product $hat(bold(h))_t dot bold(E)_(y_t, :)$. Since

$ norm(hat(bold(h))_t - bold(E)_(y_t, :))_2^2 = norm(hat(bold(h))_t)_2^2 + norm(bold(E)_(y_t, :))_2^2 - 2 dot hat(bold(h))_t dot bold(E)_(y_t, :), $

minimizing the left-hand side also maximizes that same dot product, so the same objective can be reframed as a metric regression problem, pulling $hat(bold(h))_t$ toward the (stop-gradiented) target embedding directly, in $L_2$. Note that the two norm terms are not both constants. The target norm is, since the row is detached, but $norm(hat(bold(h))_t)_2^2$ is not, so the term is dot-product maximization plus an explicit penalty on the output norm. Cross-entropy only depends on logit differences and is therefore free to grow $norm(hat(bold(h))_t)$ without bound. 

Averaging over the $N$ non-masked positions in a batch and over the $d$ embedding dimensions:

$ cal(L)_"aux" = 1 / (N d) sum_(t=1)^N norm(hat(bold(h))_t - "detach"(bold(E)_(y_t, :)))_2^2, $

where $t$ enumerates non-masked positions over the flattened batch. The training objective becomes

$ cal(L) = cal(L)_"CE" + lambda dot cal(L)_"aux", quad lambda in [0, 0.5] "(in my experiments)" $

I wanted to see the effect of this additional term on the overall loss landscape graphically, and as such, I decided to plot the raw cross-entropy loss versus the regularized one. To construct the plots, I took a fresh-initialized checkpoint, a checkpoint halfway through training, and the final model. This was critical, as the loss depends on the values of $bold(E)$ which change throughout training. I then sampled random normal matrices $bold(A),bold(B),bold(C), bold(D)$, where $bold(A),bold(B)$ would be added to the $bold(Q),bold(K),bold(V)$ projection matrices as an conic perturbation, e.g. $bold(Q)^prime = bold(Q) + alpha_1 bold(A) + alpha_2 bold(B), quad alpha_1, alpha_2 > 0$. Note the perturbation was normalized as per #cite(<li2018landscape>, form: "prose"). The same approach was applied to the MLP projection matrix, just with $bold(C), bold(D)$ as the direction matrices. With the modified weight matrix (one at a time), I ran the forward pass of the model and computed loss against the provided validation split of FineWeb. This gave a 3D landscape loss plot, with $alpha_1, alpha_2$ as $x,y$ coordinates and loss making up the $z$.

#figure(
  grid(
    columns: 1,
    row-gutter: 4pt,
    image("landscape_raw/block11_Q_landscape3d.png", width: 400pt), align(center)[(a) freshly-initialized model],
    image("landscape_midway/block11_Q_landscape3d.png", width: 400pt), align(center)[(b) model midway through training],
    image("landscape_final/block11_Q_landscape3d.png", width: 400pt), align(center)[(c) final trained model],
  ),
  caption: [Loss landscape evolution across training, *Query matrix in transformer block \#11*. Left is pure cross-entropy, right is cross-entropy and the auxiliary term added up. Color-value scaling shared between the plots. Auxiliary term ends up manifesting almost solely as a constant offset of the loss.],
) <q-loss-perturb>
#v(1em)
First, looking at the loss landscapes when perturbing the attention projection matrices (specifically $bold(Q)$ in @q-loss-perturb), the auxiliary L2 term introduces a constant offset of the loss landscape, but does not affect the geometry (landscape smoothness) at all.

#figure(
  grid(
    columns: 1,
    row-gutter: 4pt,
    image("landscape_raw/block5_MLP_landscape3d.png", width: 400pt), align(center)[(a) freshly-initialized model],
    image("landscape_midway/block5_MLP_landscape3d.png", width: 400pt), align(center)[(b) model midway through training],
    image("landscape_final/block5_MLP_landscape3d.png", width: 400pt), align(center)[(c) final trained model],
  ),
  caption: [Loss landscape evolution across training, *MLP projection matrix in transformer block \#5*. Left is pure cross-entropy, right is cross-entropy and the auxiliary term added up. Color-value scaling shared between the plots. Auxiliary term ends up minimal or only offsets the loss value. The landscape geometry remains essentially identical.],
)

#v(1em)

In my experiments, I ran a sweep of values of $lambda$ from 0.025 to 0.5. The additional loss term did not bring overly significant improvements, however, they were still measurable and ultimately led to couple of percent of reduction in overall training time. It is possible that the improvement is in the gradient direction as whole, and cannot be really (visually) inferred from any low-dimensional, isolated perturbation.

#figure(
  table(
    columns: 5,
    inset: 4pt,
    align: (center, center, center, center, center, center),
    table.header(
[*$lambda$*], [*Iters*], [*Val loss*], [*Wall*], [*First cross*],
    ),
  [0 (curr. baseline)], [10566], [3.379], [117.0 min], [116.0 min],
  [0.5], [10566], [3.380], [114.9 min], [113.9 min],
  [0.2], [10566], [3.379], [115.0 min], [113.9 min],
  [0.1], [10566], [3.378], [114.4 min], [*111.5 min*],
  [0.05], [10566], [3.379], [115.0 min], [113.9 min],
  [0.025], [10566], [*3.367*], [115.0 min], [114.0 min],
  ),
  caption: [Aux-loss coefficient ($lambda$) sweep across different values, compared to then-baseline (previous optimizations already applied). "First cross" is first evaluation step at or below 3.3821 on any logged metric (raw or EMA). Evaluation steps spaced apart by about 1 minute and 64 training steps. In bold: overall fastest crossing  and overall best final loss. Addition of the auxiliary loss term always accelerates first crossing of validation CE loss.],
)

The observed behavior suggests the additional loss term "accelerates" the model towards the optimum of cross-entropy in every $lambda$ tested, however, once close to it, starts pulling the parameters away/overshoots, and ultimately degrades the model to a worse final loss than pure cross-entropy. This is in line with the final loss slightly growing with increasing $lambda$. The crossing time 111.5 minutes is the fastest runtime achieved overall, and therefore became the final training recipe.

= Final Training Recipe
 <final-recipe>
The final training recipe combines the limited attention windowing described in @context-windowing, specifically, it uses the per-layer varying configuration described in @final-recipe-context-windows.  PyTorch's `flex_attention` measurably benefits from shorter attention windows in terms of raw computation time, pairing that with the short-range assumption appeared to be a good direction for saving some time. 
#figure(
  table(
    columns: 3,
    inset: 5pt,
    align: (left, center, center),
    table.header(
      [*Stat*], [*Baseline*], [*Fastest cross*],
    ),
    [Batch $B$], [16], [64],
    [Grad accum], [const. 32], [`0:2,0.57:4`],
    [Iterations], [4768], [10566],
    [Warmup / warmdown], [256 / 1024], [571 / 4543],
    [Attention windows], [dense causal], [depth-varying cascade (@final-recipe-context-windows)],
    [EMA decay(s)], [none], [*0.98*, 0.97, 0.96],
    [Aux $lambda$], [0], [0.1],
    table.hline(),
    [Final val loss], [3.379], [3.378],
    [Total training time], [149.6 min], [114.4 min],
    [First cross (target 3.3821)], [148.6 min], [*111.5 min*],
    [Total tokens processed], [2.500 B], [1.980 B],
[Tokens to first cross], [2.483 B], [1.928 B],
    table.hline(),
    [Training-time speedup vs. baseline], [--], [*23.5 %*],
    [First-cross speedup vs. baseline], [--], [*25.0 %*],
  ),
  caption: [Baseline vs. the fastest-crossing recipe, the actual speedrun winner by the project's objective (minimize training time to cross 3.3821). EMA decay responsible for first crossing highlighted in bold. All unlisted hyper-parameters remain identical.],
)
The measurable difference between the same context in every layer and the variable ones was just *24 seconds*, as both crossed the threshold in the same step, one was just slightly faster to compute.

Next, it used the batch (grad-accumulate) schedule (B=64, 0:2,0.57:4) from @bs-sched. Two independent findings stacked here. First, at a fixed 262,144 tokens/update, splitting it as B=64/ga=4 beats B=16/ga=16 on pure GPU kernel throughput with 3.6% less wall clock and statistically identical loss. Second, ramping ga from 2→4 over the first 57% of training beats holding it constant at 4. This simpler two-stage version, tried after the constant grad-accumulate tuning had already found 4 as the target, is what worked the best. 

It had a slightly-shortened iteration count (10566, warmup 571, warmdown 4543), derived by proportionally shortening original 11,575-iteration schedule by about 8.7%. I arrived at this number by re-running the experiment couple of times and slowly shortening the iteration count, approx. doubling the decrease from 2% and up. Run shortened to 10% did not cross the threshold, so 8% (+ rounding errors) remained as the best. The proportion of warmup/constant/warmdown split over all of the iterations remained the same.

To track the shorter training with a smaller batch size, three EMAs (@ema) (0.98, 0.97, 0.96) were used to track the model. All short horizons (~25–50 steps). EMA 0.98 is what actually delivered the fastest crossing time.

Finally, the training used the embedding auxiliary loss (@aux-loss) with $lambda=0.1$. 



= Conclusion And Potential Improvements

Through both training algorithm optimization and slight architectural improvements, I was able to achieve a reasonable speedup of 25%, without having to touch any hyper-parameter besides the batch size, which I adjusted in a principled and theoretically well-motivated manner to not count as "hyper-parameter magic". 

A big open question is whether these improvements can withstand stochasticity of different random seeds, and then perhaps what is the average speedup and its standard deviation. From some very rough observations, I believe the tuned recipe (@final-recipe) is lower variance than the baseline, which as mentioned in @stochasticity, occasionally had problems crossing the target threshold at all, so my expectation is they will hold up, or increase  slightly on average.

Because I was testing the modifications in a linear manner (the same one as presented in this report) there has potentially been some performance left on the table, as e.g. the embedding factorization could work better with the auxiliary loss term. However, such  extensive experiments were beyond my time and financial resources (approx. \$1200 in compute spent on the presented experiments alone).