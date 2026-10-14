# Design

This is a reference integration between **NeMo Gym** (environment: task
generation, agent harness, verification) and **Fireworks AI** (training:
Dedicated GPU trainer + inference deployment, GRPO, weight hotloading,
checkpointing), wired together through Fireworks' own
`training.recipes.async_rl_loop.main()` framework. This document is the
engineering narrative -- what's held constant across every environment/model
combination, why it's built the way it is, and what's still open.

## Ownership split

| | NeMo Gym owns | Fireworks owns |
|---|---|---|
| Task generation, dataset | Yes | -- |
| Agent harness (multi-turn tool-calling loop) | Yes | -- |
| Verification / reward | Yes | -- |
| Trainer + inference deployment lifecycle | -- | Yes |
| Rollout fan-out/admission, GRPO group assembly | -- | Yes |
| Optimizer, weight hotload, checkpointing | -- | Yes |

Neither side reimplements the other's job. The integration surface is
deliberately small: a recording proxy (in the cookbook) and one NeMo Gym config option (below).

## Architecture

```
                        NeMo Gym owns this box                     Fireworks owns this box
        +-------------------------------------------------+   +--------------------------------+
        |  resources server   agent harness   model server |   |  Dedicated trainer (GRPO)      |
        |  (verifier, tools)  (e.g. simple_agent, (inference_ |   |  Dedicated inference deployment |
        |                      toolsandbox_agent) provider,   |<--+  (hotloaded every step)         |
        |                      )              )             |   |                                  |
        +-------------------------------------------------+   +--------------------------------+
                        |                          |
                        |  /run (direct HTTP)       |  proxy.py: RecordingChatProxy
                        |  from rollout_fn           |  (OpenAI-compatible HTTP server,
                        v                          v   samples from the live deployment,
              async_rl_loop.main()  <----------------  records exact tokens/logprobs)
              (GRPO groups, optimizer,
               checkpointing, hotload)
```

`train.py`'s `rollout_fn` makes one direct HTTP `/run` POST per sample
straight to the live NeMo Gym agent harness (setting NeMo Gym's own
`_ng_task_index`/`_ng_rollout_index` correlation fields itself plus a unique
`_ng_rollout_id` per call, and passing the full dataset row through -- this is the same contract
`rollout_collection.py` inside NeMo Gym itself uses), then drains the
matching `proxy.py` session into a `RolloutRun`/`RolloutSample` for the
recipe.

## The stable contract: two integration points, everything else is config

**1. The cookbook's `nemo_gym` example** (`proxy.py`/`train.py` in
[`fw-ai/cookbook`](https://github.com/fw-ai/cookbook) `training/examples/rl/nemo_gym/`,
upstreamed in cookbook#704) is the only code that has to exist for *any* NeMo
Gym environment to plug in -- switching environments is
`--resources-server`/`--agent-name`/`--dataset-path`, switching models is
`--base-model`/`--tokenizer-model`/`--training-shape-id`. No code changes
between the runs in `RESULTS.md`. It targets the Dedicated
trainer + deployment recipe (`async_rl_loop`) only.

**2. NeMo Gym's `correlate_via_user_field` option.** Every concurrent
rollout needs its own proxy session, so the proxy must know which rollout each
model call belongs to. NeMo Gym's `inference_provider` model server forwards
its per-rollout correlation id as the outbound OpenAI `user` field when
`correlate_via_user_field: true` and `observability_enabled: true` are set
(NVIDIA-NeMo/Gym#3783, merged to `main` 2026-10-08). `train.py` writes both
into `env.yaml`. An explicit `_ng_rollout_id` in the `/run` body takes
precedence over the contextvar, and `train.py` makes it unique per call so a
retry can never append onto a stale session. The proxy rejects any request
without a `user` field (HTTP 400) rather than sharing one session across
rollouts.

Both pieces used to be patches carried by this repo (a Gym middleware patch
and a cookbook patch). Both are upstream now and the patches are gone.

## Key design decisions

**`async_rl_loop.main()`, not a hand-rolled training loop.** An earlier
iteration of this integration drove the raw RLOR/`service_mode` API directly
and, separately, hand-rolled a GRPO/forward-backward loop on top of the
serverless Tinker protocol. Both were abandoned: `service_mode=True` requires
an active `forward_backward`/`optim_step` stream to actually train --
without it, a job silently reports `JOB_STATE_COMPLETED` having processed
zero requests. `async_rl_loop.main()` (the same framework Fireworks' own
`harbor_rl_opencode` example uses) owns that stream correctly, along with
trainer/deployment lifecycle, rollout admission, and checkpointing --
reimplementing it was solving an already-solved problem.

**Direct `/run` HTTP calls, not `gym eval run` subprocesses.** The recipe's
`rollout_fn` contract is one call per individual sample. NeMo Gym's own
`rollout_collection.py` (what `gym eval run` uses internally) posts a raw
dict row -- including `_ng_task_index`/`_ng_rollout_index` -- to the agent's
`/run` endpoint over HTTP. `rollout_fn` replicates that exact call directly
instead of shelling out to the `gym eval run` CLI per sample, which would
mean one subprocess per rollout draw.

**`DeploymentSampler`, not a raw Tinker session client.** Early attempts
called the low-level Tinker `SamplingClient` directly against the serverless
shared-trainer pool. This failed unpredictably under real load (session
heartbeat death, unbounded retries with no visible backoff). Fireworks'
`DeploymentSampler` wraps the same call with a bounded, structured
retry/timeout budget built for exactly the post-hotload warm-up window that
was causing the raw client to hang -- switching to it (plus moving to the
Dedicated trainer+deployment path, since `DeploymentSampler` is built for
that path, not the serverless one) resolved it.

**Trajectory tracking: rollout-session identity, not content hashing.**
`proxy.py` needs to recognize when an incoming request is turn N+1 of an
in-progress rollout rather than a fresh one. The first approach
(`training.utils.rl.rollout.MessageTrajectoryAssembler`, built for callers
that drive raw chat messages in-process) checks exact dict equality between
what the proxy recorded and what the harness sends back. That assumption
breaks for any agent that round-trips through NeMo Gym's Responses API
between turns (confirmed: `simple_agent` reconstructs a fresh chat-messages
list from Responses-format state on every turn, not guaranteed
byte-identical to what was returned the turn before) -- it hard-crashed on
turn 2 every time.

The next iteration (`TrainingSessionTree` + `turn_matching
.MessageHashFingerprinter`) degraded gracefully on a hash mismatch instead of
crashing -- but on `workplace_assistant` (longer, more tool-call-heavy
conversations than `example_multi_step`) this degraded far enough that
*zero* turns were ever recognized as continuations. Root cause, traced fully:
Qwen3.5's renderer emits a `reasoning_content` key whenever the model
produces non-empty chain-of-thought; NeMo Gym's `ResponsesConverter` silently
drops that key on its Responses<->chat round-trip unless
`uses_reasoning_parser=true` (never set here, previously). The proxy hashed
the pre-drop message; the harness's next request reflected the post-drop
one -- any turn where the model actually thought hashed differently on the
way back in. `workplace_assistant`'s 27-tool tasks elicit reasoning on nearly
every turn; `example_multi_step`'s 2-tool tasks let the model skip it often
enough to occasionally match by chance, which is why the bug looked
environment-specific rather than universal.

**Where it ended up.** Two fixes, both live-validated (see `RESULTS.md`):
(1) `train.py` sets `uses_reasoning_parser: true` on the generated
`env.yaml` -- an independent correctness fix, since without it the model's
reasoning was silently discarded from NeMo Gym's own trajectory history on
*every* run using this proxy, not just this bug. (2) The proxy trusts
rollout-session identity and does not compare message hashes at all:
`simple_agent`'s episode loop (`responses_api_agents/simple_agent/app.py`) is
strictly sequential per rollout and never forks or rewrites history, so any
request against an in-progress session is necessarily a continuation of that
session's last turn. Each rollout's history is therefore **append-only** (the
upstreamed `RolloutSession` is a plain chain over `DeploymentTrainingSession`),
and harnesses that fork or rewrite history are unsupported. When a turn's
prompt is not a token-prefix of the previous prompt + output, the rollout is
split into several training segments rather than silently truncated; all
segments are returned, and the per-rollout log line reports turns, segments
and trained tokens.

**Sampling follows the recipe, not the request.** The proxy samples with
`RolloutSetup.sample_kwargs` (temperature, `top_p=1.0`, `top_k=0`,
`max_seq_len`, ...) so the proxy's samples are on-policy exactly as the recipe
configures them. A per-request `temperature`/`top_p` from NeMo Gym is ignored,
and a request's `max_tokens` can only lower `--max-completion-tokens`. The
reported `finish_reason` comes from the sampler (`length` when the cap was
hit), and `usage` carries real token counts.

**Failure handling.** If `/run` errors, times out, or returns no reward,
`rollout_fn` releases that rollout's proxy session before re-raising, so
nothing leaks across retries. `gym env start` readiness is polled against the
head server (`/server_instances`) with a configurable timeout
(`--gym-start-timeout-s`, default 900s -- a first start on a fresh checkout
builds one venv per server), and the process is stopped if startup fails.

**Tool-call formatting fix.** `tinker_cookbook`'s renderer emits
`tool_calls[].id = None` and `function.arguments` as a parsed dict, not a
JSON string. NeMo Gym's own strict OpenAI-schema validation correctly
rejects both -- meaning every real tool call the model attempted was
silently 500ing until `proxy.py` normalizes the shape (`_fix_tool_calls_for_openai`).

**Filtering out `toolsandbox`'s internal user-simulator, without a second model server (opt-in).**
`toolsandbox` is the first environment validated with a genuinely different
agent harness (`toolsandbox_agent`, not `simple_agent`) and a dual-role
serving setup: the resources server itself drives a simulated-user role via
an independent `/v1/chat/completions` call to the same rollout's policy
model, triggered from *inside* `toolsandbox`'s own `/step` handler, not from
the agent harness. NeMo Gym's rollout-correlation contextvar propagates
unconditionally by URL path regardless of which process makes the call
(`nemo_gym/server_utils.py`, `nemo_gym/rollout_correlation.py`), so this
user-simulator call reaches the proxy tagged with the *same* rollout id as
the agent-under-test's turns -- naively, it would get woven into the same
training-sample token chain, making the simulated user's tokens look like
the policy's own action.

ToolSandbox's own tool-visibility rules (vendored, unmodified, from the
upstream Apple ToolSandbox project) mean the agent-under-test's turns always
carry the scenario's real (multi-entry) toolset, while the user-simulator
either gets no tools at all, or exactly one -- `end_conversation` -- which is
never visible to the agent role. `_is_user_simulator_call()` uses exactly this
to sample and serve the user-simulator's calls normally while excluding them
from the training chain -- no second model server and no per-environment
config to keep in sync.

Because the heuristic would misclassify any environment whose *agent* turns
legitimately carry no tools (plain single-turn tasks), the upstreamed version
applies it **only when asked**: pass `--exclude-user-simulator` for
`toolsandbox`. It is off by default. Live-validated on the merged code: 33
proxy calls, 11 excluded, 22 agent turns trained (see `RESULTS.md`).

**A second, independent Fireworks API surface: Deployments.** Everything
above uses `async_rl_loop.main()`'s SDK-managed rollout deployment --
provisioned, hotloaded, and torn down entirely inside the training loop.
`fireworks.resources.deployments.DeploymentsResource.create()` is a
genuinely separate surface: standing up an on-demand deployment for an
*already-promoted* model (via `--output-model-id`, which routes through
`training/examples/tools/promote_checkpoint.py`'s underlying promotion
path), independent of any training run being active. Two things learned
doing this live: (1) inference deployments require a power-of-2 world size
(`accelerator_count=3`, which is what the training shape used, fails with
`world_size must be a power of 2`; `4` works) -- training and inference
accelerator topology constraints aren't the same. (2) Deploying a promoted
`HF_PEFT_ADDON` (LoRA) model by passing its own resource name as
`base_model` to `deployments.create()` triggers a live-merge deployment
automatically -- no `--enable-addons`/multi-LoRA setup needed for the
single-adapter case. See `RESULTS.md`'s "Base vs. fine-tuned" section for
the live query-and-compare result.

## Adding a new environment

No code changes needed in the common case:

```bash
python -m training.examples.rl.nemo_gym.train \
    --resources-server <resources-server-name> \
    --agent-name <agent-process-name> \
    --dataset-path <path-to-jsonl>
```

Do a free dry-run first (`gym env start --resources-server <name>
--model-type inference_provider`, watch for `All N / N servers ready`) before
spending anything live -- catches missing per-server dependencies or config
issues at zero cost.

## Open items

- **TITO integration.** The cookbook's SDK-side exact-token path (`TITOSidecar`)
  is environment-local: one endpoint plus bearer token per trajectory, with a
  drift policy (`new_segment`/`reject`). NeMo Gym's shared model server tags
  rollouts with the `user` field instead, so using it would need a small adapter
  that routes by `user` to per-rollout TITO trajectories. Not built; the proxy's
  own recording works today.
- **Serverless.** The example is Dedicated-only. A serverless variant
  (`async_rl_loop_serverless`) was tried and removed from the upstream PR:
  tokenizer remote-code handling and intermittent Gym-side rollout hangs were
  unresolved.
- ~~Fix trajectory-tracking turn linkage~~ -- done, see `RESULTS.md`.
- ~~Validate against a genuinely different agent harness~~ -- done
  (`toolsandbox`/`toolsandbox_agent`), see `RESULTS.md`.
- ~~Validate against a non-binary reward shape~~ -- done, `toolsandbox`'s
  milestone-similarity reward is genuinely continuous (0.500-0.920 range
  observed), see `RESULTS.md`.
- ~~Exercise a second Fireworks API~~ -- done, the Deployments API
  (promote -> on-demand deploy -> query -> compare -> tear down), see
  `RESULTS.md`.
- Scale one environment past the smoke-test config to a real training run
  with a genuine held-out before/after eval comparison (the current
  base-vs-fine-tuned comparison reuses a training-set row and a rank-8/
  3-step LoRA, so it proves the deployment *pipeline*, not fine-tuning
  *quality* -- see `RESULTS.md`'s honest framing of that result).
  `workplace_assistant`'s own config already references a 1260-prompt
  HuggingFace dataset (`nvidia/Nemotron-RL-agent-workplace_assistant`), so
  this doesn't require hand-generating data.
- `toolsandbox`'s full scenario set (`prepare_toolsandbox.py`, likely
  several hundred to 1000+ scenarios per the vendored Apple ToolSandbox
  generators) is unbuilt/unused here -- only the 5-row smoke set has been
  run. Building and validating against it would be a natural scale-up
  target alongside `workplace_assistant`'s.
- A fourth+ environment, if useful: `genrm_compare` (still `simple_agent`,
  but a cohort-based *relative* reward shape rather than a fixed-target
  score -- a different generalization axis than anything validated so far).
