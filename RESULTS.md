# Results

Every row below used the *same, unmodified* integration code (the two patches
in this repo) -- only the `--resources-server`/`--agent-name`/`--dataset-path`
and `--base-model`/`--tokenizer-model`/`--training-shape-id` flags changed
between rows. See `DESIGN.md` for what's actually being held constant here and
why that's the point.

| Environment | Agent harness | Model | Steps | Reward split | Turn linkage | Log |
|---|---|---|---|---|---|---|
| `example_multi_step` | `simple_agent` | `qwen3p5-27b` (dense) | 3 | 9✓ / 11✗ (20 total) | Not measured this run | see history below |
| `example_multi_step` | `simple_agent` | `qwen3p5-35b-a3b` (MoE, 35B/3B active) | 3 | 4✓ / 16✗ (20 total) | Not measured this run | see history below |
| `workplace_assistant` (pre-fix) | `simple_agent` | `qwen3p5-27b` (dense) | 2 | 6✓ / 10✗ (16 total) | 0 APPEND / 16 NEW / 40 WIPE -- **fixed, see below** | [`runs/workplace_assistant_qwen3p5-27b_2026-08-28.log`](runs/workplace_assistant_qwen3p5-27b_2026-08-28.log) |
| `workplace_assistant` (post-fix) | `simple_agent` | `qwen3p5-27b` (dense) | 3 | real mixed reward, 0.000→0.375→0.750 mean/step | **43 APPEND / 20 NEW / 0 WIPE** | [`runs/workplace_assistant_qwen3p5-27b_fixed_2026-09-01.log`](runs/workplace_assistant_qwen3p5-27b_fixed_2026-09-01.log) |
| `toolsandbox` | `toolsandbox_agent` (dual-role: agent-under-test + internal user-simulator) | `qwen3p5-27b` (dense) | 3 | continuous reward, 0.813→0.829→0.714 mean/step | 64 APPEND / 20 NEW / 0 WIPE / **55 user-simulator turns correctly excluded** | [`runs/toolsandbox_qwen3p5-27b_2026-09-01.log`](runs/toolsandbox_qwen3p5-27b_2026-09-01.log) |

All runs: real trainer + Dedicated deployment provisioning, real rollouts
against NeMo Gym's live verifier, real GRPO optimizer steps with weight
hotload back into the live deployment, clean automatic teardown (verified
against the Fireworks API afterward, not just trusted from the log).

## Per-step detail

**`example_multi_step` / dense:** step 1 = 0.625 (8 samples), step 2 = 0.125
(8 samples), step 3 = 0.750 (4 samples).

**`example_multi_step` / MoE:** step 1 = 0.250 (8 samples), step 2 = 0.125 (8
samples), step 3 = 0.250 (4 samples). Provisioning this shape (6x B200 vs. the
dense model's 3x B200) took ~11 minutes to reach `RUNNING`, notably longer
than the dense model -- budget a longer no-progress timeout for larger
training shapes.

**`workplace_assistant` / dense, pre-fix:** step 1 = 0.375 (8 samples), step 2
= 0.375 (8 samples). Real multi-step agentic behavior: search a CRM for
customers matching filter criteria, delete the matching records, report back
correctly -- but see the turn-linkage caveat below; the reward and rollout
were correct, the *training sample* likely wasn't.

**`workplace_assistant` / dense, post-fix:** step 1 = 0.000 (8 samples), step
2 = 0.375 (8 samples), step 3 = 0.750 (4 samples). Same task, same model.
Token-ancestry logs now show real 2-6 turn training samples per rollout
(`[trajectory] token ancestry split N turns into N segments`, N up to 6) --
previously every sample was truncated to 1 turn regardless of how many turns
the rollout actually took.

**`toolsandbox`:** step 1 = 0.813 (8 samples), step 2 = 0.829 (8 samples),
step 3 = 0.714 (4 samples). Individual rollout rewards ranged 0.500-0.920 --
genuinely continuous, not a rebadged 0/1 (confirmed against the verifier's own
ROUGE-L/geometric-mean milestone-similarity math, not just the environment's
README claim). The internal user-simulator (a second, independent model call
issued by the resources server itself mid-rollout, sharing the same
rollout_id as the agent-under-test) was correctly identified and excluded
from every training sample -- see `DESIGN.md`'s "Key design decisions" for
how.

## Fixed: turn linkage on `workplace_assistant`

**Root cause (traced, not guessed):** Qwen3.5's renderer emits a
`reasoning_content` key on the assistant message whenever the model produces
non-empty chain-of-thought. NeMo Gym's `ResponsesConverter` silently drops
that key on its Responses<->chat-completions round-trip unless
`uses_reasoning_parser=true` is set on the model server -- which this
integration never did. The proxy's turn-continuation check hashed the
pre-drop message; the harness's next request reflected the post-drop one.
`workplace_assistant`'s 27-tool tasks elicit reasoning on nearly every turn
(near-100% mismatch, 0 real `APPEND`s observed); `example_multi_step`'s 2-tool
tasks let the model skip reasoning often enough to occasionally match by
chance.

**Fix, two parts:**
1. `train.py` now sets `uses_reasoning_parser: true` on the generated
   `env.yaml` -- an independent correctness fix (previously the model's
   reasoning was silently discarded from NeMo Gym's own trajectory history on
   every run, not just this one).
2. `proxy.py`'s turn-continuation check now trusts rollout-session identity
   as the primary signal (any request against an in-progress rollout is
   necessarily a continuation of that rollout's current leaf turn --
   `simple_agent`'s episode loop is strictly sequential per rollout and never
   rewrites history), keeping the content-hash comparison only as a logged
   consistency check, not a hard gate.

Live-validated: 43 real `APPEND` / 0 `WIPE` / 20 `NEW` on the re-run (vs. 0
`APPEND` / 40 `WIPE` / 16 `NEW` before), with real 2-6 turn training samples
now captured per rollout.

## Base vs. fine-tuned: exercising a second Fireworks API

Beyond `async_rl_loop.main()` (the training/rollout API), this also exercises
Fireworks' **Deployments API** (`fireworks.resources.deployments`) directly --
a genuinely separate surface from the SDK-managed rollout deployment
`async_rl_loop` stands up internally:

1. Trained `example_multi_step` with `--output-model-id`, producing a real
   promoted Fireworks model (`kind: HF_PEFT_ADDON`, LoRA r=8, base
   `qwen3p5-27b`) via `training/examples/tools/promote_checkpoint.py`'s
   underlying promotion path -- state `READY` immediately after training.
2. Stood up a short-lived on-demand deployment for the **base** model
   (`accelerator_type=NVIDIA_B200_180GB`, `accelerator_count=4` -- inference
   deployments require a power-of-2 world size, unlike the 3x-accelerator
   training shape; `accelerator_count=3` fails with `world_size must be a
   power of 2`), queried it with a real `POST
   /inference/v1/chat/completions` call (full addressing:
   `<base-model>#<deployment-resource-path>`) against `example_multi_step`
   row 0's prompt + tool schema, then tore it down.
3. Repeated the same for the **fine-tuned** model
   (`base_model=<promoted-model>` -- passing a `HF_PEFT_ADDON` model's own
   resource name to `deployments.create` triggers a live-merge deployment,
   no `--enable-addons` needed), same prompt, then tore that down too.

**Result:** both models correctly called `get_synonym_value` for `"Blazing"`
then `"Warm"` (the exact pattern the prompt's own worked example teaches) --
structurally identical output on this particular row. Reported honestly
rather than manufactured: at this smoke-test scale (LoRA rank 8, 3 optimizer
steps, single held-out row that was also part of the 5-row training set), a
detectable base-vs-fine-tuned behavioral delta on an already-easy first turn
isn't expected. What this demonstrates is the *mechanical* pipeline --
promote -> deploy on-demand -> query with a real request -> compare -> tear
down -- working end to end against a second, independent Fireworks API. A
real evaluation of fine-tuning *quality* needs the scaled run + held-out eval
described in `DESIGN.md`'s open items.

Both deployments verified deleted via the Fireworks API afterward (not just
trusted from the SDK response).

## Cost

Every attempt provisions a real Fireworks Dedicated trainer + inference
deployment; rates vary by shape (roughly $40-52/hr for the dense model's 3x
B200 shape, $80-100/hr for the MoE model's 6x B200 shape at time of writing --
confirm current rates at https://fireworks.ai/pricing). `cleanup_on_exit=True`
plus the async loop's own circuit breaker reliably tear both down within
minutes of completion or failure; every training run above cost under $15.
The base-vs-fine-tuned on-demand deployments (§ above) ran ~3.5 minutes each
before teardown. One training attempt during this pass hit a transient
Fireworks hotload timeout (`Hotload failed for sampler snapshot`, a known
platform flakiness, not an integration bug) -- `cleanup_on_exit` tore the
resources down correctly even on that crash path, confirmed via the API; the
retry succeeded cleanly.
