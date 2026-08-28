# NeMo Gym x Fireworks RL Integration

A confirmed-working proof of concept: **NeMo Gym controls the RL environment**
(task generation, a real multi-turn tool-calling agent harness, verification)
while **Fireworks AI runs training** (Dedicated GPU trainer + inference
deployment, GRPO optimization, weight hotloading, checkpointing), wired
together through Fireworks' own `training.recipes.async_rl_loop.main()`
framework.

Live-verified against two structurally different models with **zero code
changes** between them:

| | Dense (`qwen3p5-27b`) | MoE (`qwen3p5-35b-a3b`, 35B total / 3B active) |
|---|---|---|
| Real optimizer steps | 3 | 3 |
| Reward split (real, non-degenerate) | 9 succeeded / 11 failed | 4 succeeded / 16 failed |

## Start here

**[`NeMo_Gym_Fireworks_AsyncRLLoop_Walkthrough.ipynb`](NeMo_Gym_Fireworks_AsyncRLLoop_Walkthrough.ipynb)**
-- a standalone, Run-All-able notebook. From nothing but a Fireworks API key
and `git`, it clones both repos, applies the two patches below, builds both
environments, demonstrates the NeMo Gym environment booting for free, then
runs a real (paid) training loop end to end. It also covers the architecture,
the key design decisions behind the two patches, cost, and how to scale up
beyond the smoke-test config used for verification.

## What's in this repo

This integration lives as a small addition on top of two upstream projects
that this repo does not vendor -- the notebook clones fresh copies of both
and applies these patches to them:

| File | What it does |
|---|---|
| `nemo_gym_multistep_new_files.patch` | Adds the NeMo Gym <-> Fireworks bridge (`proxy.py`, `train.py`) as a new example under [`fw-ai/cookbook`](https://github.com/fw-ai/cookbook)'s `training/examples/rl/`. |
| `inference_provider_rollout_id.patch` | A small middleware fix to [`NVIDIA-NeMo/Gym`](https://github.com/NVIDIA-NeMo/Gym)'s `inference_provider` model server, needed to correlate concurrent rollouts correctly (see the notebook's architecture section for why). |

Neither patch has been upstreamed yet -- both are living here until they are
(or until the maintaining teams determine a different owning location).

## Requirements

- A Fireworks account with billing enabled and an API key.
- `git` on `PATH`. Everything else (`uv`, both repos, both Python
  environments) is bootstrapped by the notebook itself.
- A machine that can run a Fireworks Dedicated training deployment for the
  duration of a run (the notebook's smoke-test config finishes in
  ~10-15 minutes; see the notebook's safety-switch section for cost).

## Known limitations

- Verified at smoke-test scale (5 rows, 3 optimizer steps) -- proves the
  pipeline is mechanically correct, not that it produces a meaningfully
  improved model. See the notebook's "Scaling up beyond the smoke test"
  section for how to go further.
- Only tested against one NeMo Gym environment (`example_multi_step`).
  `train.py` takes `--resources-server`/`--agent-name`/`--dataset-path` to
  point at a different one, but that path is untested live.
- The `inference_provider_rollout_id.patch` workaround exists because NeMo
  Gym's own rollout-correlation mechanism isn't wired up on model servers --
  worth upstreaming a proper fix rather than carrying this patch long-term.
