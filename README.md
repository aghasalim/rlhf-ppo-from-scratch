# rlhf-ppo-from-scratch

Reward modelling from preferences, PPO fine-tuning of a language model, and a deliberate study of reward overoptimization — plus the simpler alternatives (DPO, GRPO, RLOO, best-of-n) at matched compute.

> **Status: scaffold. Nothing here is built or measured yet.**
> This repo currently holds the project specification, the shared agent conventions,
> and an empty logbook. Every number in the tables below is a `TODO` because no
> experiment has been run. The `prompts/` task specs referenced in the wave table
> are not written yet either.
>
> Nothing in this repo is estimated or taken from a paper. When a table has a number
> in it, that number came from a run in `results/`.

---

## Why

PPO for RLHF has a reputation for being finicky, and the reputation is earned: four models in memory, a dozen implementation details that each silently cost you performance if wrong, and a reward signal that your policy is actively trying to exploit. Most of the tricks are in code, not papers.

The part worth building for its own sake is **task 04**: deliberately overoptimize against the reward model and watch true quality peak and then fall while the proxy score keeps climbing. That's Goodhart's law with a plot, on your own hardware, and it's the single most useful thing to have actually seen if you want to reason about RLHF.

## Hardware

- **GPU:** `TODO — python -m scripts.env`
- 16GB minimum, 24GB comfortable. **The constraint is four models at once** (policy, reference, reward, value). Use Pythia-410M or Qwen2.5-0.5B, not a 7B. LoRA on the policy and a shared backbone for value/reward buy you headroom if you need it.

## Results

TL;DR summarization, win rate vs the SFT baseline (judged, held-out):

| Method | Win rate ↑ | KL to ref | RM score | GPU-hrs |
|---|---:|---:|---:|---:|
| SFT | 50% (ref) | 0 | TODO | TODO |
| Best-of-4 | TODO | TODO | TODO | TODO |
| Best-of-16 | TODO | TODO | TODO | TODO |
| PPO | TODO | TODO | TODO | TODO |
| DPO | TODO | TODO | TODO | TODO |
| GRPO | TODO | TODO | TODO | TODO |
| RLOO | TODO | TODO | TODO | TODO |

The overoptimization curve — proxy RM score and gold score against √KL:

`results/overoptimization.png` — TODO

## Waves

```
00 bootstrap + eval harness              (serial)
   ├─ 01 theory: PG → TRPO → PPO         ┐
   └─ 02 SFT + reward model              ┘ parallel
        └─ 03 PPO implementation         (serial — the hard one)
             ├─ 04 overoptimization study┐
             └─ 05 DPO/GRPO/RLOO/BoN     ┘ parallel
                  └─ 06 comparison + writeup
```

| Task | OWNS | READS |
|---|---|---|
| 00 | `scripts/`, `Makefile`, `rlhf/eval/`, `data/` | — |
| 01 | `notes/00-ppo.md`, `rlhf/ref/` | `scripts/` |
| 02 | `rlhf/sft.py`, `rlhf/reward_model.py`, `train/` | `data/`, `rlhf/eval/` |
| 03 | `rlhf/ppo/` | `rlhf/reward_model.py`, `rlhf/ref/` |
| 04 | `experiments/overopt/` | `rlhf/ppo/`, `rlhf/eval/` |
| 05 | `rlhf/alternatives/` | `rlhf/reward_model.py`, `rlhf/sft.py` |
| 06 | `bench/`, `notes/paper.md`, `README.md` | everything |

See [`CONVENTIONS.md`](CONVENTIONS.md).

## Author

Aghasalim Mustafazada — third-year AI student at Howest, Belgium.

<p align="center">
  <a href="https://github.com/aghasalim">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="github"></a>
  <a href="https://www.kaggle.com/aghasalimmustafazada">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="kaggle"></a>
  <a href="https://linkedin.com/in/mustafazada">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"></a>
  <a href="https://orcid.org/0009-0001-8746-4582">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="orcid"></a>
</p>

## License

MIT — see [LICENSE](LICENSE).
