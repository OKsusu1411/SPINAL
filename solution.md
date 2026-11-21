# solution.md — G1 spinal cfg에서 실행 로직을 terrain API 클래스로 분리 (R2)

상태: **R2 APPLIED — 사용자 승인 후 적용함. 승인 조건에 따라 `TYPE_CHECKING` import는 모두 뺐다.**
이전 R2 terrain curriculum solution.md는 `/tmp/claude-1000/-home-oksusu-Repos-SPINAL/b2967d2e-5b25-4218-a5bb-7df4204828f6/scratchpad/solution_R2_backup.md`에 백업되어 있음.
라인 번호는 현재 작업 트리 기준임.

## Conclusion

`G1_spinal_env_cfg.py`에는 다음 세 가지만 남긴다.

- 설정값
- `@configclass` 선언 (`SurfaceMaterialRangeCfg`, `SpinalTerrainTileCfg`)
- `__post_init__` (값 대입, 그리고 `G1SpinalTerrain` 객체 생성)

실행 로직은 모두 `G1_spinal_terrain.py`의 `G1SpinalTerrain` 클래스로 옮긴다.

| 현재 위치 (`G1_spinal_env_cfg.py`) | 이동 후 |
|---|---|
| `prepare_terrain_tiles()` (209–236) | `G1SpinalTerrain.prepare_tiles(cfg)` |
| `_sample()` (238–244) | `G1SpinalTerrain._sample()` |
| `_sample_material()` (246–254) | `G1SpinalTerrain._sample_material()` |
| `_validate_tile()` (256–279) | `G1SpinalTerrain._validate_tile(cfg, tile)` |
| `__post_init__`의 `terrain_stage_steps` 검증 (166–168) | `G1SpinalTerrain.prepare_tiles()`의 cfg 검사 구간 |
| `terrain_tiles: list = []` (207) | `terrain_api: G1SpinalTerrain` 필드로 교체. 타일은 `terrain_api.tiles`에 저장 |
| 모듈 함수 `spawn_terrain_tiles()` (`G1_spinal_terrain.py:20`) | `G1SpinalTerrain.spawn_tiles(cfg, device)` |

`SpinalTerrainTileCfg`는 cfg 파일에 그대로 둔다.

동작은 바뀌지 않는다. 샘플링 순서, RNG seed 규칙, 검증 조건, 스폰 경로와 좌표가 모두 기존과 같다.

## Review — R1 사용자 코멘트 반영

| 코멘트 | 반영 |
|---|---|
| `spawn_terrain_tiles()` 통합 → **good** | 유지. `spawn_tiles()` 메서드로 합친다. |
| **`__post_init__`은 유지하라** | 유지. 값 대입만 하고, `raise` 검증만 API로 옮긴다. |
| **Cfg 선언 및 `G1SpinalTerrain` 객체 생성은 cfg에서 수행하라** | `SpinalTerrainTileCfg`를 cfg에 남긴다. `G1SpinalTerrain()`은 cfg의 `__post_init__`에서 만들어 `cfg.terrain_api`에 넣는다. env는 이 객체를 사용하기만 한다. |

## Logic

### 1. 객체가 cfg를 참조하지 않는 이유 (메서드가 `cfg`를 인자로 받음)

객체를 cfg 안에서 만들기 때문에, `G1SpinalTerrain(self)`처럼 cfg를 객체에 저장하면 `cfg → terrain_api → cfg` 순환 참조가 생긴다.

Isaac Lab `class_to_dict()`(`IsaacLab/source/isaaclab/isaaclab/utils/dict.py`)는 `__dict__`가 있는 값을 계속 재귀 순회한다. 순환 참조가 있으면 무한 재귀가 된다.

그래서 객체에는 결과 `tiles`만 저장한다. cfg는 `prepare_tiles(cfg)`와 `spawn_tiles(cfg, device)`의 인자로 받는다. device는 cfg를 만들 때는 아직 정해지지 않으므로 `spawn_tiles()`의 인자로 받는다.

### 2. import 순환을 피하는 방법

`G1_spinal_terrain.py`는 실행 시점에 `SpinalTerrainTileCfg`가 필요하므로 cfg 모듈을 최상단에서 import한다. 반대로 cfg가 terrain 모듈을 최상단에서 import하면 모듈 순환이 생긴다. 그러면 어느 파일을 먼저 import하느냐에 따라 `ImportError`가 날 수 있다.

해결: cfg는 `__post_init__` 안에서만 `G1SpinalTerrain`을 import한다.

- `__post_init__`이 실행될 때는 cfg 모듈이 이미 다 로드된 상태이므로 순환이 생기지 않는다.
- gym 등록은 문자열 entry point를 쓰고, 인스턴스는 train/play에서 `parse_env_cfg`로 만든다. 그래서 이 시점은 항상 앱이 실행된 뒤다.
- `TYPE_CHECKING`은 쓰지 않는다(사용자 지시). 두 파일 모두 `from __future__ import annotations`를 쓰므로 annotation은 평가되지 않는 문자열이다. Isaac Lab `configclass`도 `get_type_hints`를 호출하지 않는다(`configclass.py:182–243`).
  - cfg의 `terrain_api: G1SpinalTerrain` annotation은 import 없이 문자열로만 남는다.
  - terrain 파일은 `G1SpinalEnvCfg`, `SurfaceMaterialRangeCfg`, `SpinalTerrainTileCfg`를 cfg에서 실제로 import한다. cfg가 최상단에서 terrain을 import하지 않으므로 순환이 생기지 않는다.

### 3. stage-step 검증 시점

옮긴 뒤에는 검증이 cfg 생성 시점이 아니라 `_setup_scene()`에서 실행된다. AISL-RL `Env.__init__`은 `_setup_scene()`을 먼저 호출하고(`AISL-RL/src/lib/env/env.py:125`), 그 다음에 `CurriculumManager`를 만든다(`:226`). 따라서 잘못된 steps는 커리큘럼이 쓰기 전에 여전히 `ValueError`로 걸린다.

### 4. 필드 이름

base cfg에 이미 `terrain = TerrainImporterCfg(...)`가 있으므로(`AISL-RL/src/lib/env/G1/base/G1_base_env_cfg.py:84`) 이름이 겹치지 않게 `terrain_api`로 한다.

### 5. 클래스 구조

```
cfg.terrain_api = G1SpinalTerrain()                          # cfg __post_init__에서 생성
  .tiles                                    : list            # 준비된 최종 타일
  .prepare_tiles(cfg)         -> list[SpinalTerrainTileCfg]   # cfg 검사 + 샘플링 + 타일 검증
  .spawn_tiles(cfg, device)   -> torch.Tensor (N, 3, 3)       # USD prim 스폰, 타일 원점 반환
  ._sample / ._sample_material / ._validate_tile              # 내부 보조 함수
```

## Code

### 1. `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_terrain.py` (전체 교체)

```python
"""Prepare and spawn G1 spinal terrain tiles as separate collision prims."""

from __future__ import annotations

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
from isaaclab.terrains.utils import create_prim_from_mesh

from lib.terrain.wall_box_terrain import WallBoxTerrain, WallBoxTerrainCfg

from .G1_spinal_env_cfg import G1SpinalEnvCfg, SpinalTerrainTileCfg, SurfaceMaterialRangeCfg

# Surface names in the order returned by the mesh callbacks
SURFACE_NAMES = ("floor", "wall", "box")
# Fixed display color per surface (visual only)
SURFACE_COLORS = {"floor": (0.3, 0.3, 0.3), "wall": (0.6, 0.45, 0.3), "box": (0.25, 0.4, 0.6)}


class G1SpinalTerrain:
    """Terrain tile API: sample final tiles from the env cfg and spawn them."""

    def __init__(self):
        self.tiles: list[SpinalTerrainTileCfg] = []    # Prepared final tiles

    def prepare_tiles(self, cfg: G1SpinalEnvCfg) -> list[SpinalTerrainTileCfg]:
        """Sample the final terrain tile list once, before the scene is created."""
        # Direct config checks (curriculum steps, finite positive geometry, valid variant count)
        steps = list(cfg.terrain_stage_steps)    # Progress thresholds of stage switches
        if not (len(steps) == 3 and all(math.isfinite(s) for s in steps) and 0.0 < steps[0] < steps[1] < steps[2] < 1.0):
            raise ValueError(f"terrain_stage_steps must be 3 finite, strictly increasing values in (0, 1): {steps}")
        if not (isinstance(cfg.terrain_num_variants, int) and cfg.terrain_num_variants >= 1):
            raise ValueError(f"terrain_num_variants must be an int >= 1: {cfg.terrain_num_variants}")
        if not all(math.isfinite(s) and s > 0.0 for s in (*cfg.terrain_tile_size, *cfg.wall_size)):
            raise ValueError(f"Tile/wall sizes must be finite and positive: {cfg.terrain_tile_size}, {cfg.wall_size}")
        if not (math.isfinite(cfg.terrain_spawn_clearance) and cfg.terrain_spawn_clearance >= 0.0):
            raise ValueError(f"terrain_spawn_clearance must be finite and >= 0: {cfg.terrain_spawn_clearance}")
        seed = cfg.terrain_seed if cfg.terrain_seed is not None else cfg.seed    # Terrain RNG seed
        rng = np.random.default_rng(seed)    # Terrain RNG
        self.tiles = []
        for terrain_type in range(3):
            for variant in range(cfg.terrain_num_variants):
                # Sample every value in a fixed order regardless of tile type
                wall_position = (self._sample(rng, cfg.wall_x_range), self._sample(rng, cfg.wall_y_range),
                                 cfg.wall_size[2] / 2.0)
                edge = self._sample(rng, cfg.box_edge_range)
                box_position = (self._sample(rng, cfg.box_x_range), self._sample(rng, cfg.box_y_range), edge / 2.0)
                materials = [self._sample_material(rng, r) for r in
                             (cfg.floor_material_range, cfg.wall_material_range, cfg.box_material_range)]
                tile = SpinalTerrainTileCfg(terrain_type=terrain_type, variant=variant, floor_material=materials[0])
                if terrain_type >= 1:
                    tile.wall_size, tile.wall_position, tile.wall_material = cfg.wall_size, wall_position, materials[1]
                if terrain_type == 2:
                    tile.box_size, tile.box_position, tile.box_material = (edge, edge, edge), box_position, materials[2]
                self._validate_tile(cfg, tile)
                self.tiles.append(tile)
        return self.tiles

    def spawn_tiles(self, cfg: G1SpinalEnvCfg, device: str) -> torch.Tensor:
        """Spawn every prepared tile and return tile origins shaped (num_variants, 3, 3)."""
        num_variants, tile_size = cfg.terrain_num_variants, cfg.terrain_tile_size    # Grid layout
        origins = torch.zeros((num_variants, 3, 3), dtype=torch.float, device=device)
        wall_box = WallBoxTerrain()    # Stateless mesh callbacks
        for tile in self.tiles:
            # Tile center in world frame (grid centered at the world origin)
            center = np.array([(tile.variant + 0.5 - num_variants / 2.0) * tile_size[0],
                               (tile.terrain_type + 0.5 - 1.5) * tile_size[1], 0.0])
            mesh_cfg = WallBoxTerrainCfg(size=tile_size)    # Per-tile final geometry
            if tile.terrain_type == 0:
                meshes, local_origin = flat_terrain(0.0, mesh_cfg)
            else:
                mesh_cfg.wall_size, mesh_cfg.wall_position = tile.wall_size, tile.wall_position
                if tile.terrain_type == 1:
                    meshes, local_origin = wall_box.wall_generate(0.0, mesh_cfg)
                else:
                    mesh_cfg.box_size, mesh_cfg.box_position = tile.box_size, tile.box_position
                    meshes, local_origin = wall_box.wall_box_generate(0.0, mesh_cfg)
            materials = (tile.floor_material, tile.wall_material, tile.box_material)
            for name, mesh, material in zip(SURFACE_NAMES, meshes, materials):
                # Move from the tile-corner frame to the world tile center
                mesh.apply_translation(center - local_origin)
                create_prim_from_mesh(f"{cfg.terrain_prim_path}/tile_{tile.variant}_{tile.terrain_type}/{name}", mesh,
                                      physics_material=material,
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=SURFACE_COLORS[name]))
            origins[tile.variant, tile.terrain_type] = torch.tensor(center, dtype=torch.float, device=device)
        return origins

    @staticmethod
    def _sample(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
        """Sample a uniform float from a validated (low, high) range."""
        low, high = value_range
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError(f"Invalid terrain range: {value_range}")
        return float(rng.uniform(low, high))

    def _sample_material(self, rng: np.random.Generator, cfg: SurfaceMaterialRangeCfg) -> sim_utils.RigidBodyMaterialCfg:
        """Sample a new physics material object from the surface ranges."""
        return sim_utils.RigidBodyMaterialCfg(
            static_friction=self._sample(rng, cfg.static_friction_range),
            dynamic_friction=self._sample(rng, cfg.dynamic_friction_range),
            restitution=self._sample(rng, cfg.restitution_range),
            friction_combine_mode=cfg.friction_combine_mode,
            restitution_combine_mode=cfg.restitution_combine_mode,
        )

    @staticmethod
    def _validate_tile(cfg: G1SpinalEnvCfg, tile: SpinalTerrainTileCfg):
        """Raise ValueError if a sampled tile is physically invalid."""
        half_tile = (cfg.terrain_tile_size[0] / 2.0, cfg.terrain_tile_size[1] / 2.0)
        aabbs = []    # Obstacle (min, max) corners in tile frame
        for size, pos in ((tile.wall_size, tile.wall_position), (tile.box_size, tile.box_position)):
            if size is None:
                continue
            if not all(math.isfinite(s) and s > 0.0 for s in size):
                raise ValueError(f"Non-positive obstacle size {size} in tile {tile.variant}/{tile.terrain_type}")
            lo = [p - s / 2.0 for p, s in zip(pos, size)]
            hi = [p + s / 2.0 for p, s in zip(pos, size)]
            if any(abs(lo[i]) > half_tile[i] or abs(hi[i]) > half_tile[i] for i in range(2)):
                raise ValueError(f"Obstacle outside tile {tile.variant}/{tile.terrain_type}")
            # Distance from tile origin to obstacle footprint in xy
            dx, dy = max(lo[0], 0.0, -hi[0]), max(lo[1], 0.0, -hi[1])
            if math.hypot(dx, dy) < cfg.terrain_spawn_clearance:
                raise ValueError(f"Obstacle inside spawn clearance in tile {tile.variant}/{tile.terrain_type}")
            aabbs.append((lo, hi))
        if len(aabbs) == 2 and all(aabbs[0][0][i] < aabbs[1][1][i] and aabbs[1][0][i] < aabbs[0][1][i] for i in range(3)):
            raise ValueError(f"Wall and box overlap in tile {tile.variant}/{tile.terrain_type}")
        for material in (tile.floor_material, tile.wall_material, tile.box_material):
            if material is not None and not (0.0 <= material.dynamic_friction <= material.static_friction
                                             and 0.0 <= material.restitution <= 1.0):
                raise ValueError(f"Invalid material {material} in tile {tile.variant}/{tile.terrain_type}")
```

메서드 본문은 기존 코드를 그대로 옮겼다. 바뀐 부분은 세 가지다.

- `self.<cfg 필드>`를 `cfg.<cfg 필드>`로 바꿨다.
- `self.terrain_tiles`를 `self.tiles`로 바꿨다.
- `prim_path`, `num_variants`, `tile_size`를 `cfg`에서 읽는다.

### 2. `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_env_cfg.py`

**import (4–5행)** — 사용하지 않게 되는 `math`, `numpy`를 삭제한다. `sim_utils`는 `SpinalTerrainTileCfg`가 쓰므로 남긴다. 다른 import는 추가하지 않는다.

**`SpinalTerrainTileCfg` (29–41행)** — 변경 없음.

**`__post_init__` (165–180행)** — 검증 3줄(166–168)을 삭제하고 steps를 직접 넘긴다. 그 다음에 terrain API 객체를 만든다.

```python
        # Curriculum: 0=Floor, 1=Wall, 2=Wall+Box, 3=mix of all tiles (env-step progress)
        self.curriculum = CurriculumManagerCfg(
            params=[
                CurriculumParamCfg(
                    name="terrain_stage",
                    attr_path="cfg/terrain_stage",
                    start_value=0.0,
                    end_value=3.0,
                    schedule="step",
                    schedule_kwargs={"steps": list(self.terrain_stage_steps)},
                ),
            ],
        )

        # Terrain tile API (local import avoids the cfg <-> terrain module cycle)
        from .G1_spinal_terrain import G1SpinalTerrain
        self.terrain_api = G1SpinalTerrain()
```

**206–207행** — `terrain_tiles` 필드를 다음으로 바꾼다.

```python
    # Terrain tile API (created in __post_init__)
    terrain_api: G1SpinalTerrain = None                         # Prepares and spawns terrain tiles
```

**209–279행** — `prepare_terrain_tiles`, `_sample`, `_sample_material`, `_validate_tile`을 모두 삭제한다.

`G1SpinalPlayEnvCfg`(282–312행)는 바꾸지 않는다. `super().__post_init__()`을 통해 `terrain_api`를 그대로 물려받는다.

### 3. `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_env.py`

**18행** — `from .G1_spinal_terrain import spawn_terrain_tiles`를 삭제한다. env는 `self.cfg.terrain_api`만 사용한다.

**173–177행**

```python
        # Prepare final tiles from cfg, then spawn them once as global static prims
        self.cfg.terrain_api.prepare_tiles(self.cfg)
        self.terrain_origins = self.cfg.terrain_api.spawn_tiles(self.cfg, self.device)    # (num_variants, 3, 3) tile centers
```

`_update_terrain_origins()`(443–455행)는 바꾸지 않는다.

## Verification

이 저장소에는 테스트 스위트가 없으므로 아래 순서로 확인한다.

1. **문법 검사**: 수정한 세 파일에 `python -m py_compile`을 실행한다.

2. **cfg 순수성**: `grep -nE "def |raise|np\.|math\." G1_spinal_env_cfg.py`의 결과가 `__post_init__` 2개뿐이어야 한다.

3. **cfg와 import 동작** (scratchpad 스크립트, headless `AppLauncher`, 물리 스텝 없음):
   - `G1SpinalEnvCfg()`와 `G1SpinalPlayEnvCfg()`를 만들었을 때 `terrain_api`가 `G1SpinalTerrain` 인스턴스여야 한다.
   - `class_to_dict(cfg)`와 `copy.deepcopy(cfg)`가 재귀 오류 없이 끝나야 한다.
   - terrain 모듈 → cfg 모듈 순서로 import해도, 반대 순서로 import해도 `ImportError`가 나지 않아야 한다(각각 새 프로세스에서 확인).

4. **동작 동일성** (같은 스크립트):
   - 변경 전 `cfg.prepare_terrain_tiles()`의 타일 값(geometry와 material)을 JSON으로 덤프한다.
   - 변경 후 `cfg.terrain_api.prepare_tiles(cfg)`를 같은 seed로 덤프한다.
   - 두 결과가 완전히 같아야 한다.
   - `terrain_stage_steps=(0.6, 0.3, 0.8)`일 때 `prepare_tiles()`가 `ValueError`를 내야 한다.

5. **스모크 실행** (GPU 사용, 별도 승인 필요):
   - `num_envs=4`로 `G1SpinalEnv`를 만들고 zero action으로 몇 스텝을 진행한다.
   - `terrain_origins.shape == (3, 3, 3)`인지, `/World/ground/tile_*` prim이 만들어지는지 확인한다.
   - 참고: `G1-spinal`은 현재 train/play 진입점에 등록되어 있지 않아서, 직접 생성하는 스크립트를 사용한다.

성공 기준: 1–4를 통과하고, 5는 승인된 경우에만 실행하여 예외 없이 끝나야 한다.

## Result (적용 후 검증)

1. `py_compile`: 세 파일 모두 통과.
2. cfg 순수성: `grep -nE "def |raise|np\.|math\."`의 결과가 `150: def __post_init__`, `210: def __post_init__` 두 줄뿐이다. `src/` 아래에 `prepare_terrain_tiles`, `terrain_tiles`, `spawn_terrain_tiles`, `TYPE_CHECKING`을 참조하는 곳이 남아 있지 않다.
3. cfg와 import 동작 (headless `AppLauncher`, 스크립트 `scratchpad/check_terrain.py`): 아래 항목을 모두 통과했다. cfg → terrain 순서와 terrain → cfg 순서 모두 exit 0이다.
   - `G1SpinalEnvCfg`와 `G1SpinalPlayEnvCfg`의 `terrain_api`가 서로 다른 `G1SpinalTerrain` 인스턴스다.
   - `class_to_dict(cfg)`와 `deepcopy(cfg)`가 재귀 오류 없이 끝난다.
   - 잘못된 `terrain_stage_steps`에 `ValueError`가 난다.
4. 동작 동일성: seed 42에서 변경 전 `prepare_terrain_tiles()`와 변경 후 `terrain_api.prepare_tiles(cfg)`의 9개 타일 JSON 덤프가 `cmp`로 바이트 단위까지 같다.
5. 스모크 실행 (`G1SpinalEnv` 생성 + 스텝, `spawn_tiles` 실제 스폰 경로 포함): **실행하지 않음.** 별도 승인이 필요하다.

참고: terrain을 먼저 import한 첫 실행에서 Kit 부팅 중(284ms, Python import 이전)에 `omni.core.ITypeFactory Circular dependency` 로그가 한 번 나왔다. 재실행에서는 나오지 않았다. Kit 플러그인 초기화 로그로 보이며, 이번 변경과는 무관하다.
