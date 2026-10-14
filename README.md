# NeMo Gym x Fireworks: RL Integration

> **Disclaimer:** This is a personal, independent project by [@shy982](https://github.com/shy982),
> not an official NVIDIA product, integration, or endorsement. It isn't
> supported or maintained by NVIDIA, and nothing here represents NVIDIA's
> position on NeMo Gym, Fireworks AI, or how the two should be integrated.
> Shared for evaluation and experimentation purposes only -- expect rough
> edges, and use at your own discretion.

A one-stop, cohesive integration between **NeMo Gym** -- environment, agent
harness, verification -- and **Fireworks AI** -- Dedicated GPU trainer +
inference deployment, GRPO, weight hotloading, checkpointing -- wired
together through Fireworks' own `training.recipes.async_rl_loop.main()`
framework.

The bridge itself now lives upstream: it is the
[`training/examples/rl/nemo_gym/`](https://github.com/fw-ai/cookbook/tree/main/training/examples/rl/nemo_gym)
example in [`fw-ai/cookbook`](https://github.com/fw-ai/cookbook) (contributed in
[#704](https://github.com/fw-ai/cookbook/pull/704)), and NeMo Gym's side is the
`correlate_via_user_field` option in [`NVIDIA-NeMo/Gym`](https://github.com/NVIDIA-NeMo/Gym)
([#3783](https://github.com/NVIDIA-NeMo/Gym/pull/3783)). **No patches are needed
against either project.** This repo is the notebook, the narrative, and the evidence.

The point of this repo is that switching environment or model is a **config
change**, not a new codebase. The same integration code and one notebook have been
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
clones both upstream repos, builds both
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

Everything the integration needs is in two upstream projects, which this repo
does not vendor -- the notebook clones fresh copies of both:

| Where | What |
|---|---|
| [`fw-ai/cookbook`](https://github.com/fw-ai/cookbook) `training/examples/rl/nemo_gym/` | The NeMo Gym <-> Fireworks bridge: `proxy.py` (`RecordingChatProxy`, records exact token ids/logprobs per rollout) and `train.py` (starts the proxy and `gym env start`, runs `async_rl_loop`). Its own README documents flags and behavior. |
| [`NVIDIA-NeMo/Gym`](https://github.com/NVIDIA-NeMo/Gym) `main` | `inference_provider`'s opt-in `correlate_via_user_field` (#3783, merged 2026-10-08), which `train.py` enables in the `env.yaml` it writes. Use any `main` from that date on. |

The two patch files this repo used to ship (`inference_provider_rollout_id.patch`
for Gym and `nemo_gym_multistep_new_files.patch` for the cookbook) were removed
once both changes landed upstream. The logs under `runs/` were produced with
those patches; see `RESULTS.md`.

In this repo:

| File | What it is |
|---|---|
| [`Setup_and_Run.ipynb`](Setup_and_Run.ipynb) | Run-All walkthrough |
| [`DESIGN.md`](DESIGN.md) | Engineering narrative |
| [`RESULTS.md`](RESULTS.md) | Run evidence |
| `runs/` | Sanitized full logs from real, confirmed runs, referenced from `RESULTS.md` |

`train.py` writes `env.yaml` in the NeMo Gym checkout (and refuses to
overwrite one it didn't generate), so use a dedicated clone -- the notebook's
`nemo-gym/` is one.

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
- Harnesses that fork or rewrite conversation history are unsupported: the
  proxy treats each rollout's history as append-only.
- `toolsandbox`'s full scenario set (hundreds+ of generated scenarios) is
  unbuilt/unused here -- only its 5-row smoke set has been run.
