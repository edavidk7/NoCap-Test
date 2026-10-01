# My Submission: ~25% Faster to Target Loss with Identical Hyperparameters

Full write-up, including the experiments that didn't work is in a comprehensive report: [`report/NoCapTest-report.pdf`](report/NoCapTest-report.pdf).

What made it into the final recipe (no other hyperparameters changed):

- **Per-head attention windows** (`flex_attention`): 4 heads per block always attend to only the last 64 tokens, and the other heads get wider windows as depth increases (128/256/512 in the shallow blocks, up to the full 1024-token context deeper in the model). Window sizes are based on the token copy-distance distribution of FineWeb.
- **Batch-size / grad-accumulation schedule**: physical batch `B=64` instead of 16, better GPU throughput at the same tokens/step. Grad accumulation is ramped `2 → 4` at 57% of training (131k → 262k tokens/step), because noisier gradients early on are good enough. Iteration count is 10566, and warmup/warmdown are scaled proportionally.
- **Passive weight EMA**: short-horizon EMAs (decays 0.98, 0.97, 0.96) evaluated alongside the raw model at no extra wall-clock cost. EMA 0.98 crossed the target first.
- **Auxiliary JEPA-like L2 embedding loss**: `λ · MSE(h_out, detach(E[y]))` pulls the final hidden state toward the tied embedding of the target token. `λ = 0.1`.

| Stat | Baseline | Fastest cross |
| - | - | - |
| Batch `B` | 16 | 64 |
| Grad accum | const. 32 | `0:2,0.57:4` |
| Iterations | 4768 | 10566 |
| Warmup / warmdown | 256 / 1024 | 571 / 4543 |
| Attention windows | dense causal | depth-varying cascade |
| EMA decay(s) | none | **0.98**, 0.97, 0.96 |
| Aux λ | 0 | 0.1 |
| Final val loss | 3.379 | 3.378 |
| Total training time | 149.6 min | 114.4 min |
| First cross (target 3.3821) | 148.6 min | **111.5 min** |
| Total tokens processed | 2.500 B | 1.980 B |
| Tokens to first cross | 2.483 B | 1.928 B |
| **Training-time speedup vs. baseline** | – | **23.5 %** |
| **First-cross speedup vs. baseline** | – | **25.0 %** |

All runs used seed 42 on a single RTX 6000 Pro Blackwell, and memory stayed under 32 GB so the recipe also fits on an RTX 5090.

---

# GPT-2 Benchmark

![logo](img/logo.png)
Hey there! Are you interested in LLMs? Do you like experimenting with neural networks, implementing different ideas and testing them out? Would you like to do that for a living? Then you're in the right place!
This is an official open test for people interested in joining [BottleCapAI](https://www.bottlecapai.com).

This project is a fork of [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) :heart:, rewritten with minimal changes to run on a single GPU (e.g. RTX 3090/4090)

---
## 📌 About BottleCapAI

At **BottleCapAI**, we’re making large language models **radically more efficient** — aiming for **100× improvements** over today’s approaches. 🚀  

### 👥 Founders
- Tomas Mikolov – creator of *word2vec*, pioneer of neural language models.  
- Jaroslav Beck – co-founder of *Beat Games* (*Beat Saber*, 10M+ copies sold, acquired by Meta).  
- David Herel – creator of Thinking Tokens, co-founder of an AI trading startup, and Amazon Alexa Prize finalist.

### 🌍 Our vision
Training frontier LLMs costs **tens of millions** today. Our new algorithms already cut that by **~50%** — and we’re just getting started. We’re building a European hub to push AI forward through **algorithms, not brute force**.  
 
📧 **hey(at)bottlecapai.com** · 🌐 [bottlecapai.com](https://www.bottlecapai.com)  

---

## 🏆 First Competition has Finished. 

**Your self-paced submissions are always welcome! Competition or no-competition, let us know if you have speed up!💪**

**[Update November 13: We have winners! See the anouncement 🏆](https://x.com/BottleCapAI/status/1989014669806432502?s=20)**
- **1st place:** $3,000 USD  Jan Chleboun
- **2nd place:** $2,000 USD  Andrej Nosov
- **3rd place:** $1,000 USD  Dominik Jurko

Competition deadline: 11.11.2025.
Prizes were awarded based on the best validated results shared via the submission process below. Ties may be broken by total training time and clarity of write-up.

---

## Objective

Prototype your idea on a subset of the **FineWeb** dataset using **1 GPU**.  
**(Optional) goal:** reach a validation loss of **≤ 3.3821** faster than the baseline.

You can achieve this by:
- making your model faster (so that it sees more data in so shorter time)
- making your training more efficient (so that in less steps your model makes better progress).

---

## What's the point?

We're not here to optimize learning rates and torch.compile flags.  
We're here to explore **algorithmic ideas that might scale**, and if that means writing your own CUDA kernel, even better.

This benchmark is meant for:
- People with limited hardware
- People with ideas and curiosity

You're encouraged to try new techniques to speed up language modeling such as but not exclusively:
- Modify the loss function
- Add auxiliary losses (multi-token prediction?)
- Modify the architecture (Mixture of Experts? Different attention?)
- Come up with a different training algorithm
- Modify the training data
- New architecture!

You're **not** expected to:
- Just bump up the learning rate
- Beat everyone with hyperparameter magic
- Do 50 runs to grid search Adam betas
- Benchmark arcane PyTorch flags
- Copy speedups from [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt)
- Modify a specific hidden layer size to align better with the number of TensorCores on your GPU 

We're interested in your own ideas, not how well you can copy other's. These ideas should be general and work on different setups and not be hardcoded to a very specific one.

You have a budget of **5B** tokens available for training, but the baseline only uses **2.5B**, so you've got room to train on more data if you make your model faster, or on less but *better* data. 

The dataset is pre-tokenized so that you don't have to do that yourself (saves time) but if you want to explore the original text, you can decode it using the GPT-2 tokenizer (`tiktoken.get_encoding("gpt2")`).

---

## Running the baseline 

To run the baseline, run the following commands.
```bash
git clone https://github.com/BottleCapAI/modded-nanogpt && cd modded-nanogpt
pip install -r requirements.txt

# you can skip this if you don't want to use W&B, in which case you should remove the --log_wandb argument from run.sh
wandb login
wandb sync wandb/run-20250410_203158-64s1zc1w # synchronizes the baseline run to your W&B account for reference

python data/cached_fineweb10B.py
./run.sh
```

---

## Benchmarks

Below is a reference leaderboard. Beating it is awesome, but **sub-baseline runs are still valuable when they demonstrate a creative idea.**

**Train a neural network to ≤ 3.3821 validation loss on FineWeb using 1 GPU.**

| # | Record time | Description                                                   | Date     | Log | Contributors |
| - | - |---------------------------------------------------------------|----------|-----| - |
1 | 5.401 hours | [baseline](https://github.com/KellerJordan/modded-nanogpt) | 11/04/25 | [log](pylog124M/14e37fbb-cc64-4185-a1a7-5ef956b56ac7.log)   | [contributors-of-modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)
2 | 4.86 hours | gated embedding projection | 20/08/25 | -   | [adam-osusky](https://github.com/adam-osusky)
3 | 3.88 hours | custom attention mask, increased context length, variable context length | 28/06/25 | -   | [filipmihal](https://github.com/filipmihal)



Note: The baseline used one RTX 4090. It took 4768 steps/iterations and used in total 2.5B tokens.

![wandb_loss](img/wandb_loss.png)

## Rules

1. **Optional:** reach validation loss ≤ 3.3821 in shorter time.
2. Do **not** introduce new datasets, but feel free to modify the current.
3. Document your idea in `IDEA.md` (motivation, method, results). Negative results are welcome—share what you learned!

If you use a different GPU than RTX 4090, benchmark the baseline and compare your speedup to that result, for example, if the baseline takes 10 hours on your setup, but your solution takes only 8 hours, then thats your speedup that you can report to us! Keep the comparison fair, if you increase the learning rate for your solution, try increasing it also for the baseline.

## Submission

To submit your results, run:
```bash
git bundle create <first name>-<last name>.bundle --all
```
Then send us your .bundle file to hey(at)bottlecapai.com with subject in format: \<first name\>-\<last name\> \<percentage speedup (dont worry if it's negative)\>.

**Didn’t beat the baseline?** No worries – send the bundle anyway **plus a short `IDEA.md`** describing:  
• what you tried & why 
• what worked 
• what didn’t.  

**Beat the baseline?** Great! Add a `RESULTS.md` with timing, settings, and hardware so others can reproduce it.

At this moment, we are interested mainly in candidates willing to relocate to Prague. (If you’re an exceptional fit, we’re happy to discuss possible support options.)

---
## Technical Notes

While this project is designed to run on **1 GPU**, there are a few things to keep in mind:

- Batch Size, Sequence Length and Gradient Accumulation:
  The current setup requires ~ 13GB of GPU memory, which might not be available to you (if you have no GPU we suggest using [Google Colab](!https://colab.research.google.com/)), in which case, you might need to tune down some hyperparameters. We recommend starting with validation batch size - this one will not affect performance but validation will take a bit longer. Next, you might tune down batch size which you might then compensate by increasing gradient accumulation to retain the same effective batch size, be careful about changing learning rate and other hyperparameters should you change effective batch size.

- **torch.compile Considerations:**  
  On some RTX cards, aggressive kernel auto-tuning via `torch.compile` can lead to shared memory issues. If you encounter errors or persistent warnings (e.g., about insufficient SMs for max autotune GEMM mode), you may have to **disable `torch.compile`** or adjust your model settings accordingly. Although this may lead to slightly slower performance, it typically resolves hardware compatibility issues.

- Multi-GPU Runs:
  This code should be ready for distributed training, if you happen to have access to multiple GPUs. In that case, make sure that Gradient Accumulation Steps is divisible by number of GPUs.

---

### Comment on the target metric

The target metric is cross-entropy loss on the FineWeb val set. The goal of the speedrun is to obtain a probability model of language which assigns a probability of at least `math.exp(-3.3821 * 1048576)` to the first 1,048,576 tokens of the FineWeb valset. Hence, we allow evaluation at any sequence length, so long as we still have a valid probability model of language on the **entire** validation set.

