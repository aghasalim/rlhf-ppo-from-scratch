# Logbook

## 2026-08-26, freezing the reference silently disabled training for every later run
**Tried:** first sweep over KL penalties. The beta=0.05 run trained fine; the next one crashed.
**Measured:**`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`, raised from backward, several function calls away from the cause.
**Concluded:**`ppo_train` sets requires_grad=False on the reference policy it is handed, and every policy in the sweep is`deepcopy(ref)`. So after the first run each new policy inherited frozen parameters and had nothing to optimise. Two fixes, both tested:`fresh()` re-enables grad on the copy it returns, and`ppo_train` copies the reference before freezing so it cannot mutate the caller's model. Worth remembering the shape of this failure: mutating an argument is invisible at the call site, and the symptom appeared in a different function on a later iteration.

## 2026-08-26, the overoptimization curve, three seeds
**Tried:** swept the KL penalty over 0.2, 0.05, 0.01 and 0, 70 PPO steps each, 3 seeds, recording proxy and gold every 5 steps. 617 s on an M4 CPU.
**Measured:** proxy rises monotonically from +9.004 at the tightest penalty to +9.966 with none. Gold peaks at +1.320 (beta=0.01, KL 19.59) and collapses to -1.426 (beta=0, KL 61.55), which is below the reference policy's -1.081. Median gold peaks at sqrt(KL) around 4.6. The mechanism is in the diagnostics: motif count climbs to 6.59 on seed 1 and the hoarding penalty the proxy never learned goes from 0.01 to 3.59.
**Concluded:** the thing worth having seen. Optimising harder made the model worse than not optimising at all, and no signal available during training would have told you. Logging the reward model score, which is what a real run logs, would have shown the last configuration as the best. The KL penalty is not a stability trick here, it is the only defence against a reward model's blind spots, and its right value is not the smallest one that trains stably.

## 2026-08-26, Best-of-N never gets off the floor, and DPO looks too good
**Tried:** compared PPO against Best-of-N (4, 16, 64), DPO, RLOO and GRPO on the same reward model.
**Measured:** Best-of-64 reaches +9.507 proxy but only -0.054 gold, still slightly worse than the reference, at an analytic KL of 3.17. DPO reaches +2.125 gold at KL 18.96, the best of any method. RLOO and GRPO land near +0.60 at KL around 6.
**Concluded:** Best-of-N simply does not travel far enough; its KL is bounded by log N - (N-1)/N so N=64 buys 3.17 nats and the good region starts further out. The DPO number I do not fully trust and said so in the README: its diagnostics show 4.99 motifs and 2.04 hoarding penalty, so it is deep in the region where the proxy is wrong and doing well anyway. It also trains on the raw preference pairs rather than the fitted reward model, so it never sees that model's generalisation error. That is a real advantage of DPO but it means this is not two optimisers on one reward, and quoting the number without the caveat would be misleading.
