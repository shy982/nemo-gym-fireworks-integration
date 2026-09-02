# NeMo Gym x Fireworks: RL Integration

A one-stop, cohesive integration between **NeMo Gym** -- environment, agent
harness, verification -- and **Fireworks AI** -- Dedicated GPU trainer +
inference deployment, GRPO, weight hotloading, checkpointing -- wired
together through Fireworks' own `training.recipes.async_rl_loop.main()`
framework.

The point of this repo is that switching environment or model is a **config
change**, not a new codebase. The same two patches and one notebook have been
run, unmodified, against two structurally different models (dense and MoE)
and **three** structurally different NeMo Gym environments -- including one
(`toolsandbox`) with a genuinely different agent harness and continuous
reward, not just a different dataset -- plus a second, independent Fireworks
API surface (Deployments: promote a checkpoint, deploy it on-demand, query
it, compare against the base model). See [`RESULTS.md`](RESULTS.md) for
every run and exact numbers.

## Start here

**[`Setup_and_Run.ipynb`](Setup_and_Run.ipynb)** -- the single entrypoint.
Standalone, Run-All-able: from nothing but a Fireworks API key and `git`, it
clones both upstream repos, applies the two patches below, builds both
environments, demonstrates the chosen NeMo Gym environment booting for free,
then (once you flip its safety switch) runs a real, paid training loop end to
end. Which environment and model it targets are two small config blocks
inside the notebook (`ENV_CONFIG`, `MODEL_CONFIG`) -- swap those, nothing
else changes.

**[`DESIGN.md`](DESIGN.md)** -- the engineering narrative: ownership split,
architecture, the two integration points that make this generic across
environments/models, the key design decisions and why alternatives were
rejected, and what's still open.

**[`RESULTS.md`](RESULTS.md)** -- the evidence: every environment/model
combination actually run, exact per-step numbers, logs, and known caveats.

## What's in this repo

This integration lives as a small addition on top of two upstream projects
that this repo does not vendor -- the notebook clones fresh copies of both
and applies these patches to them:

| File | What it does |
|---|---|
| `nemo_gym_multistep_new_files.patch` | Adds the NeMo Gym <-> Fireworks bridge (`proxy.py`, `train.py`) as a new example under [`fw-ai/cookbook`](https://github.com/fw-ai/cookbook)'s `training/examples/rl/`. |
| `inference_provider_rollout_id.patch` | A small middleware fix to [`NVIDIA-NeMo/Gym`](https://github.com/NVIDIA-NeMo/Gym)'s `inference_provider` model server, needed to correlate concurrent rollouts correctly (see `DESIGN.md` for why). |

Neither patch has been upstreamed yet -- both are living here until they are
(or until the maintaining teams determine a different owning location).

`runs/` holds sanitized full logs from real, confirmed runs, referenced from
`RESULTS.md`.

## Requirements

- A Fireworks account with billing enabled and an API key.
- `git` on `PATH`. Everything else (`uv`, both repos, both Python
  environments) is bootstrapped by the notebook itself.
- A machine that can run a Fireworks Dedicated training deployment for the
  duration of a run (the notebook's smoke-test config finishes in
  ~10-15 minutes; see `RESULTS.md` for cost).

## Adding a new NeMo Gym environment

In the common case, no code changes -- add a new preset to `ENV_CONFIG` in
the notebook's §7 (resources server name, agent name, dataset path) and run.
See `DESIGN.md`'s "Adding a new environment" section for the details and a
free-dry-run tip before spending anything live.

## Known limitations

- Verified at smoke-test scale (5 rows, 2-3 optimizer steps per run) --
  proves the pipeline is mechanically correct, not that it produces a
  meaningfully improved model. The base-vs-fine-tuned comparison in
  `RESULTS.md` is likewise a pipeline check, not a quality eval (rank-8 LoRA,
  3 steps, a training-set row). See the notebook's "Scaling up beyond the
  smoke test" section and `DESIGN.md`'s open items for how to go further.
- `toolsandbox`'s full scenario set (hundreds+ of generated scenarios) is
  unbuilt/unused here -- only its 5-row smoke set has been run.
- The `inference_provider_rollout_id.patch` workaround exists because NeMo
  Gym's own rollout-correlation mechanism isn't wired up on model servers --
  worth upstreaming a proper fix rather than carrying this patch long-term.
