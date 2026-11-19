# SPINAL Research — Codex Instructions

## Scope and Global Rules

Follow `~/.codex/AGENTS.md` together with these project-specific instructions. Read both before starting work. Do not duplicate or replace the global workflow with a separate project protocol.

- SINGLE is the default for a new session. Preserve an explicitly selected mode across tasks; `/init` and task completion do not change it.
- In SINGLE, obtain explicit approval of `consult.md` before changing source code.
- In TEAMWORK, Codex owns `consult.md` and `.agentbus/feedback.md`; Claude owns `solution.md` and approved source changes. Approval of `consult.md` authorizes design only; approval of the final `solution.md` authorizes implementation.
- Use `~/.claude/teamwork.sh` for TEAMWORK communication and follow SEND → ACK → RESULT → NEXT ACTION. An ACK is not completion. Stop dependent work on failed or missing results; inspect timed-out requests before retrying.
- Only an explicit user command changes modes. Never activate TEAMWORK solely because an agent message arrived or leave it automatically after completing a task.
- Never stage, commit, or push Git changes. Preserve existing user changes and avoid destructive Git operations.

## Research Objectives

SPINAL implements learning-based safety control for humanoid robots. The current primary platform is the Unitree G1, 29-DoF.

The system contains two policies:

1. **Nominal policy:** clock-free, command-conditioned locomotion with robust tracking and balance under disturbances.
2. **Spinal policy:** fast reactive safety control, fall-damage mitigation, and contact-aware protective behavior.

A risk/value predictor determines when safety intervention is needed. A future proactive training stage may optimize the nominal policy using safety information from the spinal policy. Do not introduce additional control policies without an explicit research requirement.

## Repository and Dependencies

Resolve paths relative to the actual repository root; do not assume a particular username or checkout location.

- `src/nominal_policy/`: nominal task environments and training/play entry points.
- `src/nominal_policy/envs/G1/G1_gait/`: G1 locomotion environment, configuration, registration, and algorithm YAML files.
- `src/nominal_policy/envs/R1/`: existing R1 environment code.
- `src/spinal_policy/envs/G1/G1_spinal/`: G1 spinal environment, configuration, registration, and algorithm YAML files.
- `AISL-RL/`: Git submodule declared in `.gitmodules`, sourced from `https://github.com/DiligentYoon/AISL-RL.git`.
- `pyproject.toml`: setuptools package configuration, Python requirement `>=3.11`, and ONNX dependencies. It does not install the complete simulation stack.

AISL-RL is the custom PyTorch RL framework. Reuse its agents, buffers, wrappers, curriculum, and environment abstractions rather than replacing it with RSL-RL or another library. G1 task classes import `G1BaseEnv` and `G1BaseEnvCfg` from `lib.env.G1.base`.

Inspect the actual checkout before choosing classes to modify. AISL-RL source was unavailable during this initialization; do not infer its current implementation from old consultation documents. Check submodule availability and applicable nested instructions before working there.

## Environment and Commands

Target environment: Ubuntu 22.04/24.04, NVIDIA Isaac Sim 5.1, Isaac Lab 2.3-compatible APIs, Python 3.11, PyTorch, and the `isaaclab` Conda environment. Verify installed versions, import paths, and dependencies locally before source changes; these targets are not proof of installed versions.

From the repository root, after preparing the simulation environment and obtaining the submodule:

```bash
pip install -e AISL-RL
pip install -e .
```

Existing entry points:

```bash
python src/nominal_policy/main/train.py --task G1-gait --algorithm MAPPO
python src/nominal_policy/main/play.py --task G1-gait-play --checkpoint <checkpoint-path>
```

Training defaults to 4096 environments; play defaults to one. Both scripts currently force headless mode. Long-running GPU training requires explicit user approval; the training command above is a reference, not a routine validation command.

Logs and checkpoints are written beneath `logs/<agent.experiment.directory>/` relative to the working directory, with timestamped run directories.

## Architecture and Registration

- Launch `AppLauncher` before importing simulation-dependent task modules and AISL-RL. Preserve this ordering in entry points.
- `src/nominal_policy/envs/__init__.py` recursively imports tasks using `isaaclab_tasks.utils.import_packages`.
- Task `__init__.py` files register Gym environments and configuration entry points. Algorithm YAML selection uses `rl_<algorithm>_cfg_entry_point`.
- The existing train/play scripts import `nominal_policy.envs` only. At initialization, `src/spinal_policy/envs/__init__.py` is empty, so importing that package alone does not discover its G1 task registrations. Do not assume the nominal entry points can launch `G1-spinal` without registration work.
- Keep environment behavior in the existing task classes, simulation and reward configuration in their configuration files, and algorithm settings in `cfg/*.yaml`.
- Do not infer algorithm support solely from CLI choices. The training script rejects buffer sizes other than `-1` because its off-policy replay path is not implemented.

## Simulation and Research Constraints

- Prefer existing Isaac Lab and AISL-RL abstractions and preserve existing interfaces.
- Keep physics timestep and control decimation configurable.
- Preserve batched PyTorch operations, tensor device placement, shapes, and dtypes. Avoid per-environment Python loops in the physics/control loop.
- Respect Isaac Lab reset and termination semantics.
- Do not change active PhysX collision geometry at runtime without verified support.
- Validate collision geometry, rigid-body settings, and USD asset paths.
- Use the official G1 four-sphere foot collision as the baseline; alternative collision models are experimental variants.
- Preserve the nominal/spinal policy distinction. Prefer simple, robust, deployable methods and avoid unnecessary control-loop complexity.
- Keep reward terms explicit, with documented mathematical definitions and weights. Do not silently change observation/action dimensions or reward semantics.

## Editing and Validation

Read relevant source before editing, preserve existing style, and change only what the approved solution requires. Write short English comments for variables and methods as required by the global instructions. Report unrelated issues without modifying them.

For implementation changes:

1. Check imports and configuration in the intended environment.
2. Check tensor shapes, dtypes, and devices.
3. Validate relevant asset spawning and collision behavior.
4. Use a small environment count before scaling up; do not launch default-size training as a smoke test.
5. Report exact commands and distinguish passing checks from untested functionality.

No dedicated test suite or test configuration was found during initialization. Inspect available checks again for the task at hand. Syntax checks alone do not establish simulation correctness. The training script includes checkpoint reload and inference comparisons, with ONNX comparisons when an ONNX checkpoint is produced; these do not replace environment validation.

Keep this file in English. Respond in Korean unless the user explicitly requests another language.
