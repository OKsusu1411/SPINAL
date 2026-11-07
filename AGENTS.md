# SPINAL Research — Codex Instructions

## Project Overview

This repository implements learning-based safety control for humanoid robots, using Unitree G1 as the current simulation platform.

Main project:
- /home/aisl/Repos/SPINAL

RL framework:
- /home/aisl/Repos/AISL-RL
- https://github.com/DiligentYoon/AISL-RL

AISL-RL is a custom PyTorch RL library. Reuse its existing architecture rather than replacing it with RSL-RL or another training library.

## Development Environment

- Ubuntu 22.04 / Ubuntu 24.04
- NVIDIA Isaac Sim 5.1
- Isaac Lab, currently targeting the 2.3-compatible API
- Miniconda environment: isaaclab
- Python 3.11
- PyTorch
- Robot: Unitree G1, 29-DoF

Verify exact installed versions and dependencies locally before changing code.

## Research Objectives

The overall system contains two policies:

1. Nominal policy
   - Clock-free G1 locomotion.
   - Command-conditioned walking.
   - Robust tracking and balance under disturbances.

2. Spinal policy
   - Fast reactive safety control.
   - Damage mitigation during falls.
   - Contact-aware protective behavior.

A risk/value predictor determines when safety intervention is needed.

A future proactive training stage may optimize the nominal policy using safety-related information from the spinal policy.

Do not introduce additional control policies without an explicit research requirement.

## Existing Project Structure

Inspect the actual repository before deciding which classes to modify.

Likely relevant components:
- SPINAL/src/nominal_policy/
- SPINAL/src/nominal_policy/envs/G1/G1_gait/
- AISL-RL/src/lib/env/
- AISL-RL/src/lib/env/G1/
- AISL-RL/src/lib/agent/
- AISL-RL/src/lib/buffer/

Do not assume the current working tree matches a previous snapshot.

## Simulation Requirements

- Prefer existing Isaac Lab and AISL-RL abstractions.
- Keep physics timestep and control decimation configurable.
- Preserve batched PyTorch operations.
- Avoid per-environment Python loops in the physics/control loop.
- Keep all tensors on the correct device.
- Respect Isaac Lab reset and termination semantics.
- Avoid changing active PhysX collision geometry at runtime without verified support.
- Validate collision geometry, rigid-body settings, and USD asset paths.

## Research Constraints

- Prioritize simple, robust, deployable methods.
- Avoid unnecessary complexity in the control loop.
- Preserve the distinction between nominal and spinal policies.
- Use the official G1 four-sphere foot collision as a baseline.
- Treat alternative foot collision models as experimental variants.
- Keep reward terms explicit, with documented mathematical definitions and weights.
- Do not silently change observation/action dimensions or reward semantics.

## Coding Rules

- Read relevant source files before editing.
- Preserve existing code style and architecture.
- Explain important architectural decisions.
- Do not run destructive Git operations.
- Follow the rules in `~/.codex/AGENTS.md`

## Validation

For every implementation:
1. Check imports and configuration.
2. Check tensor shapes and dtypes.
3. Validate asset spawning and collision behavior.
4. Test a small environment count before scaling up.
5. Report the exact commands executed.
6. Clearly distinguish passing tests from untested functionality.

Long-running GPU training requires explicit user approval.

Respond in Korean unless explicitly requested otherwise.
