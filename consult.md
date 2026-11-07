# G1 spinal: TerrainGenerator의 n × 3 지형과 순차 curriculum

작성일: 2026-11-03. **설계 보고이며 구현 파일에는 적용하지 않았다.** 현재 사용자의 요청에 따라 보고서 이름은 `consult.md`를 사용한다. 아래 기존 보고서는 이력으로 보존하며, 이번 설계가 이전의 실행별 단일 유형 선택안을 대체한다.

## Conclusion

**`TerrainGenerator`로 n행 × 3열의 정적 지형을 한 번 생성하고, AISL-RL `CurriculumManager`가 정한 목표 단계에 맞춰 reset 대상 로봇의 origin을 이동시키는 구성을 권장한다.**

사용자가 확인한 두 조건은 다음과 같다.

- n은 동일 장면의 복제 수가 아니라 **각 유형의 난이도 단계 수**다.
- 진행 순서는 **Floor n단계 → Wall n단계 → Table+Wall n단계**다.

| 행 / 열 | 열 0: Floor | 열 1: Wall | 열 2: Table+Wall |
| --- | --- | --- | --- |
| 0 | 단계 0 | 단계 n | 단계 2n |
| 1 | 단계 1 | 단계 n+1 | 단계 2n+1 |
| … | … | … | … |
| n−1 | 단계 n−1 | 단계 2n−1 | 단계 3n−1 |

Isaac Lab 설정은 `num_rows=n`, `num_cols=3`이고, 그림을 전치해 보면 사용자가 표현한 3 × n 행렬과 같다. 물리 지형 tile은 3n개지만, 병렬 로봇 수 E=`scene.num_envs`와는 별개다. E를 3n으로 강제하지 않는다.

전체 단계 s에 대한 좌표는 다음과 같다.

```text
row = s % n
column = s // n
origin[e] = terrain_origins[row[e], column[e]]

F0 → F1 → … → F(n−1) → W0 → … → W(n−1) → TW0 → … → TW(n−1)
```

지형 생성·로봇 배치·학습 진도를 분리한다.

| 역할 | 담당 |
| --- | --- |
| 지형 배치와 origin 행렬 | Isaac Lab `TerrainGenerator` / `TerrainImporter` |
| 목표 단계 schedule | `from lib.curriculum import CurriculumManagerCfg, CurriculumParamCfg` |
| 목표 단계의 실제 적용 | spinal env의 `_reset_idx()` 직전 origin 갱신 |
| robot state 초기화 | 기존 AISL-RL reset event와 robot Tensor API |

`AISL-RL/src/lib/curriculum/__init__.py`가 공개하는 API를 사용한다. 기존 `Env`가 manager를 생성하고 매 step 갱신하므로 별도 manager, callback 체계, training loop를 만들 필요가 없다. 해당 라이브러리 파일도 수정하지 않는 안이다.

### 아직 확정하지 않은 실험 조건과 제안값

확정된 것은 행렬 구조와 유형 이동 순서다. 아래는 검토를 위한 **제안**이며, 사용자 결정으로 간주하지 않는다.

1. 현재 AISL-RL API에 맞춰 우선 **시간 기반 schedule**을 사용한다. 성공률 기반 승급이 필요하면 성공의 정의·평가 window·승급/강등 기준을 먼저 정해야 한다. 현재 API는 시간 비율을 입력받으므로 성능 기반이라고 표현하면 안 된다.
2. n은 코드 예시에서 5로 둔다. 모든 유형을 동일한 난이도 축으로 비교하기 위해 **reset 직후의 초기 root 속도 외란 크기**를 행에 따라 증가시키는 초안을 제시한다. 단순 평면만 n개 생성하면 Floor의 실제 난이도는 달라지지 않기 때문이다.
3. 이 초안에서는 벽·테이블의 geometry를 행마다 바꾸지 않는다. 행별 물리 차이는 초기 속도 분포이며, 이후 기존 interval push는 그대로다. 벽 거리·테이블 높이를 난이도 축으로 삼는 대안도 가능하지만, 가까운 지지물이 보호 행동을 오히려 쉽게 만들 수 있어 난이도 증가 방향을 사전 실험으로 확인해야 한다.
4. 벽은 generator가 생성하는 얇은 box mesh, 테이블은 앞서 요청한 공식 USD asset을 유지한다. `TerrainGenerator`의 함수는 mesh와 origin을 반환하므로 USD spawn을 그 함수 안에서 실행하지 않는다. 대신 **생성 완료된 Table+Wall 열의 각 tile origin에 정적 USD를 초기화 시 한 번씩 추가**한다. 따라서 전체 공간은 TerrainGenerator 행렬을 따르고, table도 로봇과 함께 이동하거나 재생성되지 않는다.

## Logic

### 1. 로컬 확인 결과

2026-11-03에 다시 확인했다. Python 3.11.14, Isaac Sim 5.1.0.0, Isaac Lab checkout `VERSION=2.3.2` / HEAD `b4c32102479`, distribution `isaaclab==0.53.1`, PyTorch 2.7.0+cu128, NumPy 1.26.0, trimesh 4.5.1이다. `lib`는 `/home/oksusu/Repos/SPINAL/AISL-RL/src/lib`에서 import된다.

| 위치: 조사 시점 줄 번호 | 확인 내용 / 필요한 변경 |
| --- | --- |
| `G1_spinal_env_cfg.py:136–194` | generator가 1행·6열이며 floor proportion=0이다. `terrain_importer_cfg`라는 이름으로 선언되어 실제 `self.cfg.terrain`과 연결되지 않는다. |
| `G1_spinal_env.py:171–196` | wall과 table을 모든 `/env_*/`에 복제하고 상속한 plane을 생성한다. 공통 지형 위에서 로봇만 이동하는 구조로 바꿔야 한다. |
| `lib/curriculum/__init__.py:1–10` | `CurriculumManager`, `CurriculumManagerCfg`, `CurriculumParamCfg`, `SCHEDULE_REGISTRY`를 공개한다. |
| `curriculum_manager.py:20–40,71–90` | `cfg.total_timesteps`와 시간 비율로 schedule을 계산하고 slash 경로의 속성을 갱신한다. |
| `curriculum_manager.py:93–107` | int 시작·종료값도 보간 결과는 float가 될 수 있다. stage를 읽을 때 `round()` 후 정수로 변환한다. |
| `curriculum/schedules.py:16–31` | `step`은 threshold마다 0, 1/m, …, 1 값을 반환한다. 이 API로 3n개 이산 단계를 만든다. |
| `lib/env/env.py:224–229,425–436` | manager 초기화 및 매 policy step update가 이미 있다. update 이후 done/reward 계산, 그다음 reset 순서다. |
| `G1_spinal_env.py:449–454` | 부모 reset 전에 배치를 변경할 수 있는 지점이다. |
| `randomizer.py:1308` | robot world reset 위치에 **`env.scene.env_origins`**를 더한다. importer origin만 수정하면 로봇이 잘못된 위치로 reset된다. |

### 2. `TerrainGenerator.curriculum=True`의 정확한 역할

이 flag는 생성 시 열별 terrain 종류와 행별 `difficulty`를 정렬하는 기능이다. 학습 중 로봇을 자동 이동시키는 manager가 아니다. 로컬 `terrain_generator.py:234–266`에서 열별 종류 선택과 행별 난이도 생성을 확인했다. `num_cols=3`, 세 sub-terrain의 동일한 양수 proportion, dict 순서 Floor → Wall → Table+Wall을 함께 사용하면 각 유형이 한 열을 차지한다. proportion=0을 사용하지 않는다. [공식 terrain API](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.terrains.html)도 생성 난이도와 origin 구조를 설명한다.

`update_env_origins(env_ids, move_up, move_down)`는 **열을 고정한 채 행만 갱신**하며, 마지막 행을 넘으면 임의 행으로 보낸다. 로컬 `terrain_importer.py:314–329`의 이 동작은 Floor → Wall → Table+Wall 이동과 마지막 단계 유지 요구에 맞지 않는다. 여기서는 `terrain_levels`, `terrain_types`, `env_origins`를 함께 갱신한다.

sub-terrain 함수는 local 좌표에서 mesh를 반환하고, generator가 tile 배치와 전체 map 중심 이동을 수행한다. 벽을 함수 안에서 월드 좌표로 배치하거나 table offset에 tile 중심을 두 번 더하면 안 된다. table world 좌표는 최종 `terrain_origins[row, 2] + table_position`으로 계산한다.

### 3. 시간 schedule과 reset의 역할

예시 n=5이면 s=0…14다. 전체 학습 시간의 80%까지 14회 승급하고 마지막 20%는 최종 단계 연습에 남기는 schedule을 제안한다. threshold는 `0.8 * k / (3n−1)`이며 k=1…3n−1이다. 80%는 제안값이다.

`CurriculumManager`는 `cfg.terrain_stage`라는 **목표 stage 값만** 갱신한다. `_reset_idx(env_ids)`에서 각 env가 실제로 도달한 stage를 기준으로 한 단계씩 따라간다.

```text
target_stage = AISL-RL CurriculumManager의 시간 기반 결과
current_stage[e] = terrain_types[e] * n + terrain_levels[e]
next_stage[e] = min(current_stage[e] + 1, target_stage)
```

이렇게 하면 긴 episode 동안 목표가 여러 번 바뀌어도 중간 단계를 건너뛰지 않는다. 진행 중인 episode를 teleport하지 않고, reset되지 않는 env의 origin도 바꾸지 않는다. 마지막 단계에서는 그대로 유지한다. 반면 종료가 드문 env는 목표보다 늦게 진행할 수 있으므로, 전체 단계 수·stage 유지 시간·최대 episode 길이를 함께 정해야 한다. 모든 env가 최종 단계에 도달했다는 것은 별도로 확인해야 한다.

여기서 시간은 병렬 sample 수 E×step이나 PPO update 횟수가 아니라 `common_step_counter`로 증가하는 **vector environment policy step 수**다. `cfg.total_timesteps`를 같은 단위의 양의 정수로 환경 생성 전에 넣어야 한다. 현재 nominal runner는 `train.py:106`에서 설정하지만 spinal runner/등록 연결은 따로 확인해야 한다. 기존 schedule의 `linear()`에는 계산한 `effective_t` 대신 t를 반환하는 코드가 있어 이 초안은 기존 `step`을 사용하고 해당 함수는 수정하지 않는다.

### 4. n개 행의 실제 난이도: 초기 속도 외란 초안

같은 열의 geometry는 유지하고, 행 r에서 다음 분포로 reset 초기 root 속도를 샘플링한다.

```text
alpha(r) = r / max(n−1, 1)
delta_velocity[j] ~ Uniform(−alpha(r) * limit[j], alpha(r) * limit[j])
limit = (0.5, 0.5, 0.0, 2.5, 2.5, 0.0)
        (vx,  vy,  vz,  wx,  wy,  wz), 단위 m/s 및 rad/s
```

행 0은 추가 외란이 없고 마지막 행은 기존 push 설정과 같은 크기의 상한을 갖는다. 열을 바꿀 때 행은 0으로 돌아가므로 **유형의 복잡도**와 **유형 내부 외란 난이도**는 별개다. 전체 난이도가 수학적으로 단조 증가한다고 주장하지 않는다. 초기 속도 주입은 기존 reset 분포에 대한 명시적인 변경 제안이며 승인 전에는 적용하지 않는다. 기존 주기적 push는 유지되므로 이 효과를 분리해서 평가해야 한다.

### 5. 공통 map과 robot별 collision

벽을 `/World/envs/env_*/Wall`로 복제해 두면 로봇만 다른 tile로 이동했을 때 벽이 따라오지 않는다. 바닥·벽 mesh와 table은 모두 `/World/ground` 아래의 **공통 정적 환경**으로 둔다. robot·sensor만 기존 env 경로에 복제하므로 robot 쪽 `replicate_physics=True`는 유지할 수 있다. 서로 다른 env의 로봇끼리는 collision filtering으로 분리하고 모든 로봇은 공통 map과 충돌하게 한다.

여러 로봇이 같은 tile에서 학습할 수 있다. 이는 tile이 로봇 전용 공간이라는 뜻이 아니다. 특히 동일 위치에서 reset되더라도 로봇 간 충돌이 생기지 않는지 GPU/CPU 작은 배치에서 확인해야 한다. 시각적으로는 로봇이 겹쳐 보일 수 있다.

`self.scene.terrain`을 private 필드로 연결하는 대신, 기존 수동 setup을 유지하면서 **`self.scene.env_origins[env_ids]`를 in-place 동기화**한다. 이는 robot reset event가 읽는 public tensor다. env Xform이나 geometry를 움직이는 동작이 아니며, 이후 robot root pose write가 실제 위치를 변경한다. terrain importer는 계속 `self.terrain`으로 접근한다.

tile 밖으로 걸어 나가 이웃 유형을 경험하는 일을 막기 위해, 경계 안쪽 0.5 m를 넘으면 episode를 **truncation**으로 끝내는 안도 포함한다. 실패 termination이나 새로운 reward penalty로 처리하지 않는다. 이는 기존 종료 동작에 추가되는 명시적인 제안이다.

## Code

아래 코드는 위 초기 속도 외란 난이도 정의까지 포함한 **검토용 적용안**이다. 신규 파일은 1개이며, 나머지는 spinal env 두 파일의 관련 부분만 수정한다. 라이브러리 curriculum API 구현·nominal 코드·Git index는 변경하지 않는다.

### A. 신규 `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_terrain.py:1`부터

Wall과 Table+Wall 열에서 공통으로 쓰는 유한 벽 mesh를 반환한다. Floor는 기존 `MeshPlaneTerrainCfg`를 사용한다.

```python
from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
from isaaclab.utils import configclass


def wall_terrain(difficulty: float, cfg: SpinalWallTerrainCfg):
    """Return floor and wall meshes; cfg defines geometry and difficulty is unused."""
    # Build the floor and its tile-local spawn origin.
    meshes, origin = flat_terrain(difficulty, cfg)
    # Build one finite wall centered above the floor.
    wall = trimesh.creation.box(extents=cfg.wall_size)
    wall.apply_translation(origin + np.array([cfg.wall_distance, 0.0, cfg.wall_size[2] / 2.0]))
    meshes.append(wall)
    return meshes, origin


@configclass
class SpinalWallTerrainCfg(MeshPlaneTerrainCfg):
    """Configure a floor tile with one static wall."""
    # Generate the floor and wall in tile-local coordinates.
    function = wall_terrain
    # Set wall thickness, width, and height in meters.
    wall_size: tuple[float, float, float] = (0.1, 3.0, 2.0)
    # Keep space between the spawn origin, table, and wall.
    wall_distance: float = 2.5
```

`difficulty`를 geometry에 사용하지 않는 것은 의도적이다. 이 초안의 행별 task 난이도는 reset 외란으로 정의한다. 향후 벽 거리 자체를 난이도화하려면 이 함수에 그 관계를 명시적으로 추가한다. 현재 예시 벽은 tile 중심에서 x=2.5 m에 있다.

### B. `G1_spinal_env_cfg.py:9–11` import 교체 및 추가

사용처를 제거하는 기존 `lib.utils.terrain_utils` wildcard import를 없애고 공개 API를 명시적으로 import한다. 다른 import는 유지한다.

```python
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, MeshPlaneTerrainCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from lib.curriculum import CurriculumManagerCfg, CurriculumParamCfg

from .G1_spinal_terrain import SpinalWallTerrainCfg
```

### C. `G1_spinal_env_cfg.py:136–194` terrain / wall / table block 교체

기존 미연결 `terrain_importer_cfg`를 실제 사용하는 `terrain`으로 교체한다. 벽의 `CuboidCfg` spawn은 mesh generator로 대체하므로 제거한다. table은 Table+Wall 열에만 배치한다.

```python
    # Store the target stage written by the AISL-RL curriculum manager.
    terrain_stage: float = 0.0

    # Generate n difficulty rows and three ordered scene columns once.
    terrain_generator_cfg = TerrainGeneratorCfg(
        curriculum=True,
        seed=42,
        size=(10.0, 10.0),
        num_rows=5,
        num_cols=3,
        color_scheme="none",
        sub_terrains={
            "floor": MeshPlaneTerrainCfg(proportion=1.0),
            "wall": SpinalWallTerrainCfg(proportion=1.0),
            "table_wall": SpinalWallTerrainCfg(proportion=1.0),
        },
    )

    # Import the generated mesh with the original floor physics material.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=G1BaseEnvCfg.terrain.physics_material.copy(),
        debug_vis=False,
    )

    # Load the official static table with its authored collision geometry.
    table_spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    )
    # Place the table relative to each Table+Wall tile origin; verify asset bounds.
    table_position = (1.2, 0.0, 1.05)

    # Bound row-dependent initial linear and angular velocity perturbations.
    reset_velocity_limits = (0.5, 0.5, 0.0, 2.5, 2.5, 0.0)
```

table z=1.05는 공식 spawn 예제를 따른 초기값이며 floor 정렬 검증이 끝난 값은 아니다. 실제 USD의 bbox·원점·up-axis·상판/다리 collider·rigid body 유무를 확인하고 table이 벽과 겹치지 않게 확정해야 한다. 공식 경로의 근거는 [USD spawn 예제](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/tutorials/00_sim/spawn_prims.html#spawning-from-another-file)다. generator의 floor/wall mesh와 별도로 USD의 material과 collision 설정은 해당 asset에서 가져온다.

### D. `G1_spinal_env_cfg.py:126–134` 기존 `__post_init__()` 마지막에 추가

`n`의 단일 기준은 importer가 실제 사용하는 `self.terrain.terrain_generator.num_rows`다. 기존 push 설정은 그대로 둔다.

```python
        # Read the number of difficulty rows from the active terrain configuration.
        num_rows = self.terrain.terrain_generator.num_rows
        # Schedule all scene stages through the existing AISL-RL public API.
        self.curriculum = CurriculumManagerCfg(
            params=[
                CurriculumParamCfg(
                    name="terrain_stage",
                    attr_path="cfg/terrain_stage",
                    start_value=0.0,
                    end_value=float(3 * num_rows - 1),
                    schedule="step",
                    schedule_kwargs={
                        "steps": [0.8 * index / (3 * num_rows - 1) for index in range(1, 3 * num_rows)],
                    },
                ),
            ],
        )
```

환경이 중복 manager를 만들거나 manager를 수동으로 두 번 update하지 않는다. `G1SpinalPlayEnvCfg.__post_init__()`의 부모 호출 직후(원본 199–200행)에는 다음을 추가한다. play는 학습 시간 denominator 없이 실행될 수 있어 시간 schedule을 끈다.

```python
        # Keep evaluation on its configured stage without a training-time schedule.
        self.curriculum = None
```

기본 play stage는 0이다. 특정 상위 stage로 바로 평가하는 기능은 아래 한 단계씩 승급 규칙과 별도로 정의해야 하므로 이번 최소안에 추가하지 않는다.

### E. `G1_spinal_env.py:171–196` `_setup_scene()` 교체

초기화 시 생성하는 table 수는 n개이며 robot 수 E와 무관하다. 이 루프는 물리/control loop가 아니다. table path는 tile에 귀속되며 env ID에 귀속되지 않는다.

```python
    # Build one shared terrain map and clone only the robots and sensors.
    def _setup_scene(self):
        super()._setup_scene()
        self.scene.clone_environments(copy_from_source=False)

        # Match the importer batch size to the robot batch size.
        self.cfg.terrain.num_envs = self.scene.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)

        # Cache the table offset in the terrain origin tensor format.
        table_offset = self.terrain.terrain_origins.new_tensor(self.cfg.table_position)
        # Add one table to each row of the Table+Wall column.
        for row in range(self.cfg.terrain.terrain_generator.num_rows):
            # Place each static table in the final terrain coordinate frame.
            table_position = self.terrain.terrain_origins[row, 2] + table_offset
            self.cfg.table_spawn_cfg.func(
                f"{self.cfg.terrain.prim_path}/Tables/row_{row}",
                self.cfg.table_spawn_cfg,
                translation=tuple(table_position.cpu().tolist()),
            )

        # Start every robot at the first Floor stage, overriding importer defaults.
        self.terrain.terrain_levels.zero_()
        self.terrain.terrain_types.zero_()
        self.terrain.env_origins[:] = self.terrain.terrain_origins[0, 0]
        self.scene.env_origins.copy_(self.terrain.env_origins)

        if self.device == "cpu" or not self.scene.cfg.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Illuminate the shared map.
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Match command generation to the robot batch and policy period.
        self.cfg.commands.num_envs = self.scene.num_envs
        self.cfg.commands.step_dt = self.step_dt
```

table spawner가 parent path를 생성하는지 및 공통 collision root 아래 table이 포함되는지는 실제 stage에서 검사한다. geometry를 생성한 뒤 clone하는 방식이 아니므로 heterogeneous asset을 robot clone에 넣지 않는다. GPU에서는 기존 scene의 `filter_collisions=True`를 유지해야 하며, 이 값을 끄고 같은 tile에 로봇들을 배치하면 로봇 간 격리가 보장되지 않는다.

### F. `G1_spinal_env.py:449` `_reset_idx()` 바로 앞에 추가

현재 stage는 importer의 row/type tensor에서 계산하므로 별도의 per-env stage buffer를 만들지 않는다. 목표 stage는 AISL-RL이 갱신한 cfg 속성이다.

```python
    # Advance resetting robots by at most one stage and synchronize reset origins.
    def _assign_terrain(self, env_ids: torch.Tensor):
        # Read the number of rows from the generated origin matrix.
        num_rows = self.terrain.terrain_origins.shape[0]
        # Clamp the scheduled scalar to valid matrix stages.
        target_stage = min(max(round(self.cfg.terrain_stage), 0), 3 * num_rows - 1)
        # Recover each robot's currently applied stage.
        current_stage = self.terrain.terrain_types[env_ids] * num_rows + self.terrain.terrain_levels[env_ids]
        # Preserve intermediate stages if the time schedule has advanced faster.
        next_stage = torch.clamp(current_stage + 1, max=target_stage)
        self.terrain.terrain_levels[env_ids] = next_stage % num_rows
        self.terrain.terrain_types[env_ids] = torch.div(next_stage, num_rows, rounding_mode="floor")
        self.terrain.env_origins[env_ids] = self.terrain.terrain_origins[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
        ]
        self.scene.env_origins[env_ids] = self.terrain.env_origins[env_ids]
```

### G. `G1_spinal_env.py:449–454` reset prefix 변경

기존 `env_ids` 정규화는 유지하고, **부모 reset event가 root pose를 쓰기 전에** `_assign_terrain()`을 실행한다. `super()._reset_idx()` 다음에 초기 외란을 넣는다. 아래 이후의 기존 foot/action/command reset과 `_compute_intermediate_values(env_ids)`는 그대로 유지한다.

```python
    # Reset selected robots on their next curriculum tile.
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._assign_terrain(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Scale the initial perturbation by the robot's applied difficulty row.
        row_scale = self.terrain.terrain_levels[env_ids].float() / max(self.terrain.terrain_origins.shape[0] - 1, 1)
        # Match perturbation limits to the robot state dtype and device.
        limits = self._robot.data.default_root_state.new_tensor(self.cfg.reset_velocity_limits)
        # Sample all resetting robots in one batched operation.
        velocity_noise = 2.0 * torch.rand((len(env_ids), 6), device=self.device, dtype=limits.dtype) - 1.0
        # Add the row-dependent perturbation to the robot's default velocity.
        root_velocity = self._robot.data.default_root_state[env_ids, 7:13] + velocity_noise * limits * row_scale[:, None]
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
```

이 초기 속도 주입은 현재 reset event의 0 속도 분포를 의도적으로 대체한다. 향후 reset event에서 별도 nonzero velocity를 샘플링하게 된다면 두 난수 분포를 어떻게 합칠지 다시 정해야 한다. 이벤트 설정 dictionary를 manager 초기화 후 수정하는 방식은 사용하지 않는다.

### H. `G1_spinal_env.py:430` `_get_dones()`의 기존 time_out 계산 직후 추가

이웃 tile에 들어가는 것을 episode 경계로 처리한다. 기존 실패 조건과 reward 계산은 유지한다.

```python
        # Read the current tile's half extents on the robot device.
        tile_half_size = self.root_pos_w.new_tensor(self.cfg.terrain.terrain_generator.size) / 2.0
        # Measure each robot relative to its currently applied tile origin.
        tile_offset = self.root_pos_w[:, :2] - self.terrain.env_origins[:, :2]
        # Truncate before a robot can enter a neighboring scene tile.
        outside_tile = (torch.abs(tile_offset) >= tile_half_size - 0.5).any(dim=-1)
        time_out = time_out | outside_tile
```

## 검증과 적용 전 확인

### 검증 목표

1. **지형 생성 →** n=2로 먼저 2×3 origin 행렬을 검사하고, floor 2개·wall 2개·table+wall 2개가 정확히 배치되는지 확인한다. generator의 중심 이동까지 반영한 최종 좌표를 기준으로 검사한다.
2. **실제 curriculum API →** 기존 `lib/curriculum` 모듈을 로드해 schedule 경계 전후, 모든 3n 단계, 최종 단계 유지, float→int stage 변환을 검사한다.
3. **reset 이동 →** E=4에서 일부 env만 reset하여 한 단계 승급, 열 전환, 중간 단계 보존, 선택되지 않은 env의 origin 불변을 검사한다. `scene.env_origins`와 `terrain.env_origins`는 `(E,3)`, float32, 같은 device여야 한다. indices는 long이어야 한다.
4. **물리 검사 →** robot 간 collision 격리, global floor/wall/table과의 접촉, table 상판·다리 collider와 벽 간 겹침을 확인한다. root reset 위치가 default position + 새 origin인지 검사한다. 공통 static asset은 runtime에 움직이거나 재생성하지 않는다.
5. **난이도와 경계 →** row 0/최종 row의 초기 속도 범위와 0.5 m 경계 truncation을 검사한다. 기존 contact sensor tensor, action/observation shape와 reward 계산이 유지되는지 확인한다. 초기 외란과 기존 interval push를 분리한 실험으로 실제 난이도 증가 여부를 평가한다.

### 이번 조사에서 확인한 기존 실행 제약

현재 spinal 코드에는 `find_bodies()` 반환 tuple을 body ID처럼 사용하는 부분과 `z_c`, `single_contact_timer`, `support_foot_rot` 등 이전 보고서에 기록한 미완성 초기화가 남아 있다. task 등록·runner 연결도 확인이 필요하다. 전체 RL 실행 성공을 위 설계의 정적 검증으로 대신 보고하지 않는다. 이 문제들은 자동 수정하지 않았다.

curriculum 진행값과 `common_step_counter`의 checkpoint 복구는 별도 검증이 필요하다. 이번 안에 checkpoint 저장 포맷 변경이나 자동 성공률 승급을 추가하지 않는다. 최종 Table+Wall 단계만 반복하면 이전 유형의 성능이 유지되는지도 별도 평가해야 한다. replay 비율은 요구하지 않았으므로 추가하지 않는다.

### 실제 조사 명령과 결과 기록

- sandbox 실행은 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`로 실패하여 승인된 읽기 실행으로 재시도했다.
- `cat /home/oksusu/.codex/AGENTS.md AGENTS.md`, `git status --short`, `rg --files`, 관련 source의 `nl -ba` / `sed -n` / `rg -n`으로 지침·호출 순서·줄 번호를 확인했다. on-disk 지침에 `solution.md` 표기가 남아 있지만 더 최근의 명시적 사용자 지시에 따라 이번 파일은 `consult.md`다.
- `nl -ba AISL-RL/src/lib/curriculum/{__init__,curriculum_cfg,curriculum_manager,schedules}.py`에 해당하는 각 파일을 읽었다. `schedules`는 디렉터리가 아니라 `schedules.py` 파일이다. 처음 디렉터리로 검색한 명령은 실패했고 파일 경로로 확인했다.
- `cat /home/oksusu/Repos/IsaacLab/VERSION` 및 `git -C /home/oksusu/Repos/IsaacLab rev-parse --short HEAD`로 checkout 버전을 확인했다. 설치된 isaaclab Python의 `importlib.metadata.version()`으로 Python distribution 버전을 확인했다.
- `sha256sum consult.md solution.md`로 이전 보고서 보존 기준을 기록했다. 작성 전후 구현·지침 파일 101개를 SHA-256으로 비교한다.
- 오프라인 검증 결과는 아래에 갱신한다. simulator·USD 네트워크 로드·GPU 물리·학습은 이번 설계 보고에서 실행하지 않는다.

<!-- curriculum-validation-results -->

실행한 검증 명령:

```bash
/home/oksusu/Apps/miniconda3/envs/isaaclab/bin/python /tmp/spinal_curriculum_consult_check.py
git status --short
```

임시 검증 스크립트는 repository 구현에 파일을 추가하지 않고 다음 항목을 실행한다. 전체 source를 import하면 Isaac Sim 초기화가 필요하므로, curriculum 패키지는 별도 namespace에서 실제 `__init__.py`와 상대 import를 그대로 실행했다. mesh·generator 함수는 설치된 source의 해당 AST만 추출하여 검사했다.

- **통과:** 제안 Python block 9개와 가상 적용한 source 3개(신규 terrain module 포함)의 문법 검사.
- **통과:** 실제 AISL-RL 공개 curriculum API로 n=1,2,5에서 모든 3n 단계 방문, threshold 전후, 최종 단계 유지 확인.
- **통과:** 일부 env reset 시 선택되지 않은 origin 유지, 열 전환, 한 단계씩 승급, 최종 단계 유지, origin shape/dtype 확인.
- **통과:** 공식 `make_plane` / `flat_terrain`과 제안 wall 함수의 local origin `(5,5,0)`, 벽 bbox `[(7.45,3.5,0),(7.55,6.5,2)]`, 닫힌 box mesh 확인.
- **통과:** 공식 generator의 열 선택 함수에서 동일한 양수 proportion이 Floor / Wall / Table+Wall 순서의 n×3 배치를 선택하는지 확인.
- **통과:** 초기 속도 tensor `(E_reset,6)` / float32, 행 0의 추가 외란 0, 축별 상한과 비활성 z/yaw축, n=1의 분모 처리 확인.
- **통과:** tile 경계 truncation이 기존 time_out을 보존하고 x/y 경계 이탈을 판정하는지 확인.
- **보존 확인:** source·지침 101개 SHA-256 변화 0개. `solution.md` 해시도 동일하다. 이전 `consult.md` 내용은 하단에 보존했다.
- **미실행:** 실제 `TerrainGenerator` 전체 객체 생성·USD spawn·Isaac Sim imports·robot contact·GPU/CPU 물리·학습. 오프라인 계산 통과를 물리 검증 통과로 해석하지 않는다.

---
