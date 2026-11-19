# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPINAL researches humanoid fall detection and fall-damage mitigation control inspired by spinal reflexes. The system is designed around two policies plus a risk/value predictor:

1. **Nominal policy** — clock-free, command-conditioned G1 locomotion with robust tracking and balance under disturbances.
2. **Spinal policy** — fast reactive safety control for damage mitigation during falls (contact-aware protective behavior).
3. **Risk/value predictor** — decides when safety intervention is needed; a future proactive stage may use its signal to refine the nominal policy.

Do not introduce additional control policies without an explicit research requirement, and preserve the nominal vs. spinal distinction.

Stack: **Isaac Lab 2.3.0 / Isaac Sim 5.1.0 / Python 3.11 / PyTorch** on Ubuntu 22.04 or 24.04, Miniconda env `isaaclab`. Primary robot is Unitree G1 (29-DoF); R1 scaffolding exists as well.

## Install / Run

```bash
git submodule update --init          # fetch AISL-RL
pip install -e AISL-RL               # provides top-level packages `lib`, `wrapper`
pip install -e .                     # provides top-level packages `nominal_policy`, `spinal_policy`

# Train (defaults: --task G1-gait --algorithm MAPPO --model Shared --num_envs 4096, always headless)
python src/nominal_policy/main/train.py --task G1-gait --algorithm MAPPO
# Play (defaults: --task G1-gait-play --num_envs 1)
python src/nominal_policy/main/play.py --task G1-gait-play --checkpoint <path/to/agent_N.pt>
```

- `--algorithm`: `PPO | MAPPO` (SAC/TD3 are listed in `choices` but not implemented). `--model`: `MLP | Shared` — `Shared` routes MAPPO to `CooperativeMAPPO`.
- Logs and checkpoints go to `logs/<yaml agent.experiment.directory>/<timestamp>_<algo>/` **relative to the current working directory** (TensorBoard events + `agent_<step>.pt`; `.onnx` is written only for single-agent runs).
- There is **no test suite and no linter**. Validation is done by running training. When a checkpoint is saved, `train.py` reloads it (and the ONNX if present) and compares inference outputs against the live agent; any mismatch raises a `RuntimeError`.
- Long-running GPU training runs require explicit user approval before you kick them off.

## Architecture

### Repository split: AISL-RL (submodule) vs. SPINAL (this repo)

- **AISL-RL** (`DiligentYoon/AISL-RL`, mounted at `AISL-RL/`): the RL framework. Agents (`PPO`, `MAPPO`, `CooperativeMAPPO`), `RolloutBuffer`, `ModelFactory`, `IsaacLabWrapper` / `RecordVideo` wrappers, domain-randomization/commander utilities (`lib.domain_randomizer.commander`), and the shared **G1 base env** `lib.env.G1.base.G1_base_env{,_cfg}`. See `AISL-RL/CLAUDE.md` for its internals. Reuse this framework — do not swap in RSL-RL or similar.
- **SPINAL** (this repo): only task environments. G1 tasks subclass the submodule's `G1BaseEnv` / `G1BaseEnvCfg`; R1 tasks use a local base at `src/nominal_policy/envs/R1/R1_base_env.py`.

### Package layout

- `src/nominal_policy/` — nominal (normal-locomotion) policy. Tasks under `envs/G1/G1_gait` and `envs/R1/R1_loco`, plus `main/train.py` and `main/play.py`.
- `src/spinal_policy/` — fall-response policy. `envs/G1/G1_spinal` was copied from `G1_gait` and modified (e.g., self-collision switched from allowed-body to forbidden-body semantics). There is no spinal-specific `main/` yet, and **`src/spinal_policy/envs/__init__.py` is empty**, so `G1-spinal` / `G1-spinal-play` are not auto-registered even though their `gym.register(...)` entries exist. Train/play scripts currently import `nominal_policy.envs` only.

### Task-registration order (do not reorder)

1. `train.py` / `play.py` parse CLI args, then launch `AppLauncher` (always `headless=True`).
2. **After** the simulator is up, they `import lib` and `import nominal_policy.envs`. Isaac Lab modules cannot be imported before the app is running — keep this order in any new entrypoint.
3. `nominal_policy/envs/__init__.py` calls `isaaclab_tasks.utils.import_packages(__name__)`, which recursively imports every subpackage; each task's `__init__.py` then runs `gym.register(...)`.
4. Registration kwargs: `env_cfg_entry_point` (the `EnvCfg` class), `rl_ppo_cfg_entry_point` and `rl_mappo_cfg_entry_point` (paths into `cfg/*.yaml`). `train.py` picks the YAML with `rl_{algorithm}_cfg_entry_point`.

### Task directory conventions (`envs/<robot>/<task>/`)

- `<task>_env.py` — implements `_setup_scene`, `_get_observations`, `_get_states` (critic-side state for async actor-critic), `_get_rewards`, `_get_dones`, `_reset_idx`, `_compute_intermediate_values`.
- `<task>_env_cfg.py` — `@configclass` EnvCfg. Reward weights are the `w_*` fields. Multi-agent is declared with `possible_agents = ["arm", "leg"]` plus per-agent dict `action_space` / `observation_space` / `state_space`; single-agent variants remain as commented-out alternatives.
- `cfg/ppo_cfg.yaml` and `cfg/mappo_cfg.yaml` — sections `models / buffer / agent / train`. Only `buffer.buffer_size: -1` is supported (it is replaced at runtime with `agent.rollouts`); any other value raises because off-policy replay is not implemented.
- To surface a reward term in TensorBoard, return it as an entry in the `infos["reward"]` dict from the env step.

### Simulation / tensor rules

- Prefer existing Isaac Lab and AISL-RL abstractions; keep physics timestep and control decimation configurable.
- Keep everything batched on the correct device — no per-environment Python loops in the physics/control path.
- Respect Isaac Lab reset/termination semantics; do not mutate active PhysX collision geometry at runtime unless the behavior is verified.
- The baseline G1 foot collision is the official four-sphere model; alternative foot geometries are experimental.
- Do not silently change observation/action dimensions or the semantic meaning of reward terms — reward terms are explicit with documented weights.

## Working Rules

- Documentation (`solution.md`, this file, etc.) is written in Korean; code comments and docstrings are written in English. This matches the AISL-RL convention.
- `.gitignore` excludes every `*.md` except `README.md`, `CLAUDE.md`, `AGENTS.md`, `solution.md`, and `consult.md`. `solution.md` and `consult.md` are kept **untracked** by git despite the whitelist (see repo policy), so treat them as scratch work docs.
- `AISL-RL/` is a git submodule — changes to the shared `lib` / `wrapper` packages belong in that repo, not here. Do not stage submodule bumps unless the user asks.
- Follow the mode/handshake rules in `~/.claude/CLAUDE.md` (SINGLE by default; TEAMWORK only on explicit user command).

## External Agent Configs

Both an OpenAI Codex config (`~/.codex/config.toml`, plus `AGENTS.md` and a `skills/` tree) and a Gemini CLI config (`~/.gemini/settings.json`, `GEMINI.md`) exist on this machine. To import user-level items (MCP servers, slash commands, subagents, skills, instructions) into Claude Code, reply `/import` to scan and list what is importable, then `/import --yes=<digest>` to apply. If `/import` is unavailable on this surface, run `claude import` from a terminal.
