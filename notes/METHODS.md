# Methods and detail

Long form detail moved out of the README.


## Method comparison


All five optimise the same reward model, so all five are exposed to the same
blind spot. What separates them is how much gold they buy per unit of drift.

| method | KL | proxy | gold | wall clock |
|---|---:|---:|---:|---:|
| SFT reference | 0.00 | +0.143 | −1.081 | 0 s |
| Best-of-4 | 0.64 | +6.025 | −0.454 | 2 s |
| Best-of-16 | 1.84 | +8.739 | −0.189 | 7 s |
| Best-of-64 | 3.17 | +9.507 | −0.054 | 41 s |
| PPO (beta=0.05) | 9.83 | +9.613 | +1.000 | 27 s |
| PPO (beta=0.01) | 19.59 | +9.818 | +1.320 | 26 s |
| **DPO** | 18.96 | +9.388 | **+2.125** | 7 s |
| RLOO | 5.81 | +8.391 | +0.600 | 5 s |
| GRPO | 6.39 | +8.320 | +0.597 | 5 s |

![gold against KL for every method](../results/methods.png)

Two things stand out.

**Best-of-N never gets off the floor.** Even at N=64 it is still slightly worse
than the reference on gold, despite reaching a proxy score of +9.507. Its KL is
known in closed form, log N minus (N−1)/N, which is 3.17 at N=64, so it simply
does not travel far enough to find the good region. It is also the only method
here whose cost is entirely at inference.

**DPO comes out best, and I do not fully trust it.** It reaches +2.125 gold at a
KL comparable to PPO's, but its diagnostics show 4.99 motifs and a hoarding
penalty of 2.04, meaning it is deep into the region where the proxy is wrong and
is doing well anyway. It also sees the raw preference pairs rather than the
fitted reward model, so it is not exposed to the reward model's generalisation
error in the same way. That is a genuine advantage of DPO, but it makes this a
comparison between two different things rather than two optimisers on one
reward, and I would not quote the number without that caveat.


## PPO details that matter


These are the ones in code rather than in the paper, all implemented in
`rlhf/ppo.py`:

- **Token level KL penalty.** The reward is the reward model at the final token
  minus beta times the per token log ratio, so credit lands where the divergence
  happened rather than being smeared over the sequence.
- **Advantage whitening.** A Bradley-Terry reward is only identified up to a
  constant, so without normalising the advantages the update size depends on an
  arbitrary scale.
- **Ratio and value clipping**, and multiple epochs per rollout, which is where
  the importance ratio stops being 1 and clipping starts doing anything.
- **GAE** over the token sequence with the reward model score as terminal.


## What I got wrong


**Freezing the reference policy silently disabled training for every later run.**
`ppo_train` sets `requires_grad=False` on the reference it is handed, and every
policy in the sweep is a deepcopy of that same reference. So the first
configuration trained and every one after it started from frozen parameters. It
does not fail where the mistake is: the optimiser just has nothing to update, and
it surfaces much later as `element 0 of tensors does not require grad`. Fixed in
two places, and there are now tests for both: `fresh()` returns a trainable copy,
and `ppo_train` copies before it freezes so it cannot mutate the caller's model.

**I built the gold reward so that overoptimization was possible, and I want to be
explicit that this is a design choice rather than a discovery.** The hoarding
term exists because a proxy trained on near-reference samples cannot learn it.
If the gold reward were fully learnable from the preference data, the curve would
not turn over, and that would also be a true finding about a different situation.
What the experiment demonstrates is the mechanism, not a claim about how often it
occurs in practice.
