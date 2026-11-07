# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

SPINAL: 척수 반사(spinal reflex)에서 착안한 휴머노이드 낙상 감지 및 낙상 피해 완화 제어 연구.
**Isaac Lab 2.3.0 / Isaac Sim 5.1.0 / Python 3.11** 기반이며, RL 프레임워크는 git submodule `AISL-RL/`(별도 저장소 `DiligentYoon/AISL-RL`)에 있습니다.

## 설치 / 실행

```bash
git submodule update --init          # AISL-RL 가져오기
pip install -e AISL-RL               # top-level 패키지 `lib`, `wrapper` 제공
pip install -e .                     # top-level 패키지 `nominal_policy`, `spinal_policy` 제공

# 학습 (기본값: --task G1-gait --algorithm MAPPO --model Shared --num_envs 4096, 항상 headless)
python src/nominal_policy/main/train.py --task G1-gait --algorithm MAPPO
# 재생 (기본값: --task G1-gait-play --num_envs 1)
python src/nominal_policy/main/play.py --task G1-gait-play --checkpoint <path/to/agent_N.pt>
```

- `--algorithm`: `PPO | MAPPO` (SAC/TD3는 선택지에만 있고 미구현). `--model`: `MLP | Shared` (Shared → `CooperativeMAPPO`).
- 로그/체크포인트는 **현재 작업 디렉토리 기준** `logs/<yaml의 agent.experiment.directory>/<timestamp>_<algo>/`에 저장됩니다 (TensorBoard + `agent_<step>.pt`, 단일 에이전트일 때만 `.onnx`).
- 테스트/린트 설정은 없습니다. 검증은 학습 실행으로 합니다. 체크포인트 저장 시 `train.py`가 저장→재로딩→(ONNX) 추론 결과를 비교해 불일치면 예외를 발생시킵니다.

## 아키텍처

### 두 저장소의 역할 분담
- **AISL-RL (submodule)**: 에이전트(PPO/MAPPO/CooperativeMAPPO), `RolloutBuffer`, `ModelFactory`, `IsaacLabWrapper`, 도메인 랜덤화/커맨드(`lib.domain_randomizer.commander`), 그리고 **G1 베이스 환경** `lib.env.G1.base.G1_base_env(_cfg)`. 자세한 내용은 `AISL-RL/CLAUDE.md` 참조.
- **SPINAL (이 저장소)**: 태스크 환경만 정의. G1 태스크는 submodule의 `G1BaseEnv`/`G1BaseEnvCfg`를 상속하고, R1은 이 저장소 안의 `src/nominal_policy/envs/R1/R1_base_env.py`를 베이스로 사용합니다.

### 패키지 구성
- `src/nominal_policy/` — 정상 보행(nominal) 정책. `envs/G1/G1_gait`, `envs/R1/R1_loco`, 그리고 학습/재생 스크립트 `main/train.py`, `main/play.py`.
- `src/spinal_policy/` — 낙상 대응(spinal) 정책. `envs/G1/G1_spinal`은 `G1_gait`를 복사해 수정한 형태(충돌 판정이 allowed→forbidden body 방식 등). 아직 전용 main 스크립트가 없고 `spinal_policy/envs/__init__.py`가 비어 있어 태스크가 자동 등록되지 않습니다.

### 태스크 등록 흐름
1. `train.py`/`play.py`는 `AppLauncher`로 시뮬레이터를 먼저 띄운 **뒤에** `import lib`, `import nominal_policy.envs`를 합니다 (isaaclab 모듈은 앱 실행 후에만 import 가능 — 이 순서를 지켜야 함).
2. `nominal_policy/envs/__init__.py`가 `isaaclab_tasks.utils.import_packages`로 하위 패키지를 재귀 import → 각 태스크 `__init__.py`의 `gym.register(...)` 실행.
3. 등록 kwargs: `env_cfg_entry_point`(EnvCfg 클래스), `rl_ppo_cfg_entry_point` / `rl_mappo_cfg_entry_point`(`cfg/*.yaml`). `train.py`는 `rl_{algorithm}_cfg_entry_point`로 YAML을 로드합니다.

### 태스크 디렉토리 규칙 (`envs/<robot>/<task>/`)
- `<task>_env.py`: `_setup_scene`, `_get_observations`, `_get_states`(critic용 state), `_get_rewards`, `_get_dones`, `_reset_idx`, `_compute_intermediate_values` 구현.
- `<task>_env_cfg.py`: `@configclass` EnvCfg. 보상 가중치는 `w_*` 필드. 멀티에이전트는 `possible_agents = ["arm", "leg"]`와 agent별 dict 형태의 `action_space`/`observation_space`/`state_space`로 정의 (단일 에이전트 설정은 주석으로 남아 있음).
- `cfg/ppo_cfg.yaml`, `cfg/mappo_cfg.yaml`: `models / buffer / agent / train` 섹션. `buffer.buffer_size: -1`(= `agent.rollouts`)만 지원.
- 보상 항목은 `infos["reward"]` dict로 반환하면 TensorBoard에 자동 기록됩니다.

## 작업 규칙
- 논의 및 `solution.md`, `CLAUDE.md` 등 문서는 한국어로, 코드 주석·docstring은 영어로 작성합니다 (AISL-RL 규칙과 동일).
- `.gitignore`가 `README.md`, `CLAUDE.md` 외의 모든 `*.md`를 무시하므로 `solution.md`는 추적되지 않는 작업 문서입니다.
- `AISL-RL/`은 submodule이므로 공용 라이브러리(`lib`, `wrapper`) 변경은 해당 저장소에서 별도로 관리됩니다.
- Follow the main rules in `~/.claude/CLAUDE.md`.
