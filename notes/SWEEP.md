# The sweep, stage by stage

What `experiments/overopt.py` did to produce `results/`, in the order it did
it, with every number pointed at the line it comes from. Nothing is read from
a config file. The values live in the code and this page only says where.

## When, on what

- 2026-08-26. That is the date on the logbook entry for the curve and the
  date `results/` was first committed (`notes/LOGBOOK.md`, `git log results/`).
- An M4 laptop CPU (`notes/LOGBOOK.md`, second entry; `README.md`, Running
  it). `results/run-meta.json` records `"device": "cpu"` and
  `"torch": "2.13.0"`.
- 617 s wall clock for the whole process (`run-meta.json`, `wall_clock_s`
  617.38). The `wall_s` column of `results/methods.csv` sums to 526 s. The
  other 91 s is SFT and reward model training, which have no timing column,
  plus the snapshots taken outside the timed loops.

## The command

    python -m experiments.overopt --seeds 0 1 2 --betas 0.2 0.05 0.01 0.0 --ppo-steps 70

from `README.md`, Running it. `run-meta.json` was written from the parsed
arguments (`experiments/overopt.py:166`) and agrees: seeds 0 1 2, betas 0.2
0.05 0.01 0.0, ppo_steps 70, pairs 4000, rm_steps 600. The last two are the
defaults at `overopt.py:83` and `:84`; the command does not override them.

## Per seed, in order

Each seed runs the five stages below from nothing. `torch.manual_seed(seed)`
is the first thing `setup` does (`overopt.py:66`).

### 1. Reference policy

`train_sft` at `overopt.py:68`, 1200 steps.

| value | where |
|---|---|
| d_model 64, 2 layers, 4 heads, max_len 32 | `rlhf/model.py:13` defaults, taken by `TinyLM(VOCAB)` at `overopt.py:67` |
| vocab 16 | `rlhf/gold.py:32` |
| 20,000 sequences from a fixed Markov chain, chain seed 1234, logits sharpened by 1.5 | `rlhf/sft.py:29`, `:17`, `:18` |
| batch 128, length 24 | `rlhf/sft.py:26` |
| AdamW, lr 3e-3, weight decay 0.01, OneCycle with pct_start 0.1 | `rlhf/sft.py:26`, `:30`, `:31` |
| grad norm clipped to 1.0 | `rlhf/sft.py:41` |

### 2. Preference pairs

4000 pairs sampled from the reference and labelled by the gold reward
(`overopt.py:69`, `rlhf/reward_model.py:42`). Generator seed `seed + 2`
(`overopt.py:69`). Pairs that tie on gold are dropped
(`reward_model.py:57`), so the reward model sees slightly fewer than 4000.
Label noise is 0 (`reward_model.py:43` default, never overridden).

### 3. Reward model

`train_reward_model` at `overopt.py:71`, 600 steps.

| value | where |
|---|---|
| same trunk shape as the policy, one scalar head on the last token | `rlhf/reward_model.py:31` |
| batch 128, AdamW lr 1e-3, weight decay 0.01 | `reward_model.py:61`, `:64` |
| grad norm clipped to 1.0 | `reward_model.py:75` |
| frozen once trained | `overopt.py:73` |

Its agreement with gold on reference samples, `rm_agreement` column of
`results/overopt-curve.csv` at step 0: 0.759 (seed 0), 0.680 (seed 1),
0.756 (seed 2). That is the "0.68 to 0.76" the README quotes. The lowest
value anywhere in the column is 0.407.

### 4. The PPO sweep

Four betas per seed, 70 steps each, a fresh copy of the reference per beta
(`overopt.py:112`). Every step is its own one step `ppo_train` call seeded
`seed * 100 + step` (`overopt.py:116`), so the rollout generator inside
`ppo_train` is reseeded each step (`rlhf/ppo.py:90`, `:103`). A snapshot is
taken every 5 steps and at step 69 (`overopt.py:118`): 15 per run, 4 runs
per seed, 3 seeds, the 180 rows of `overopt-curve.csv`.

Everything not on the command line is a `PPOConfig` default
(`rlhf/ppo.py:37` to `:52`):

| batch 256 | length 24 | lr 3e-4 | epochs 4 | minibatch 64 |
|---|---|---|---|---|
| clip 0.2 | vf_coef 0.5 | vf_clip 0.2 | ent_coef 0.0 | gamma 1.0, lam 0.95 |

Advantage whitening on (`ppo.py:50`), grad norm clipped to 1.0
(`ppo.py:126`), AdamW (`ppo.py:97`).

Snapshots score 1024 sequences of length 24 (`overopt.py:47`) drawn with a
generator seeded 99 (`overopt.py:51`), the same for every snapshot, so
consecutive rows differ only by the policy.

Wall clock per PPO run, `methods.csv` column `wall_s`: 22.8 s to 32.5 s
over the 12 runs.

### 5. The alternatives

Same reward model, same reference, same seed.

| method | setting | where | `wall_s`, median of 3 |
|---|---|---|---|
| Best-of-4, 16, 64 | 1024 groups, generator seed `seed + 3`, KL by the closed form | `overopt.py:132`, `:134`, `rlhf/alternatives.py:40` | 1.6 s, 7.5 s, 41.3 s |
| DPO | 400 steps on the stage 2 pairs, batch 64, lr 3e-4, beta 0.1 | `overopt.py:148`, `alternatives.py:46` | 6.6 s |
| RLOO | 40 steps, kl_beta 0.02, k 4, batch 64, lr 3e-4 | `overopt.py:150`, `alternatives.py:107` | 5.3 s |
| GRPO | as RLOO, group mean and std as the baseline | `overopt.py:150`, `alternatives.py:113` | 5.0 s |

## What came out

- `results/overopt-curve.csv`, 180 rows, one per snapshot.
- `results/methods.csv`, 33 rows, one per seed and method.
- `results/run-meta.json`, the arguments and the wall clock.

Written at `overopt.py:158` to `:168`. No weights are written anywhere.
There is no `torch.save` in the repository, and the policies are gone when
the process exits. Getting a point on the curve back means rerunning the
command above.

## Gold reward constants

Fixed, never swept: motif (3, 7), hoard threshold 3, weights 1.0 motif, 0.6
repeat, 1.4 hoard (`rlhf/gold.py:33` to `:37`).
