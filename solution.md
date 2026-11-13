# G1SpinalEnv terrain spawn 검증 스크립트

## Conclusion

env 전체를 띄우지 않고 **terrain만 단독으로 생성해 검사하는 스크립트** `src/spinal_policy/main/check_terrain.py`를 새로 추가합니다.

- `G1SpinalEnv`는 아직 미정의 속성(`single_contact_timer` 등)과 태스크 미등록 때문에 실행되지 않습니다. 그래서 env를 거치지 않고 `G1SpinalEnvCfg.terrain_importer_cfg`만 꺼내 씁니다. 실제 학습과 같은 cfg를 쓰므로 검증 결과가 그대로 유효합니다.
- 기존 코드는 수정하지 않습니다. 새 파일 1개만 추가합니다.
- 현재 상태로 돌리면 **wall_box column 생성에서 실패할 것으로 예상합니다.** `AISL-RL/src/lib/terrain/wall_box_terrain.py:56`이 없는 필드 `cfg.box_size`를 참조하기 때문입니다 (L57의 `np.array([cfg.box_position])`도 shape이 `(1, 3)`). 이 스크립트가 그 실패를 가장 먼저 드러내며, submodule 쪽 수정 후 다시 돌리면 됩니다.

## Logic

검사는 3단계입니다. 1~2단계는 수치 검사라 headless로 끝나고, 3단계는 GUI에서 눈으로 확인합니다.

1. **Grid 구조**: `TerrainGenerator`를 직접 만들어 `terrain_origins.shape == (num_rows, 3, 3)`인지 확인합니다. column 수가 sub-terrain 수(floor / wall / wall_box)와 같아야 합니다.
2. **칸별 형상**: 전체 terrain mesh에서 각 칸 영역(origin ± size/2) 안의 vertex 중 바닥보다 높은 것(`z > 0.01`)만 골라, origin 기준 상대 좌표의 bounding box를 구합니다.
   - floor: 솟은 vertex가 없어야 합니다.
   - wall: 솟은 vertex의 범위가 cfg 값과 일치해야 합니다. x는 `wall_distance ± wall_size[0]/2`, y는 `± wall_size[1]/2`, 최대 z는 `wall_size[2]`.
   - wall_box: wall 조건에 더해, wall보다 앞쪽(x가 wall 앞면보다 작은 영역)에 솟은 vertex(= box)가 있어야 합니다.
   - 모든 칸의 bounds를 표로 출력하므로 assert가 통과해도 값을 직접 볼 수 있습니다.
3. **Stage → origin 매핑 + 실제 spawn**: `SimulationContext`와 `TerrainImporter`로 stage에 실제로 spawn한 뒤, env의 `_update_terrain_origins`와 같은 식(`level = stage % num_rows`, `type = stage // num_rows`)으로 stage 0~8의 origin을 출력합니다. `--headless`가 아니면 창을 유지해 육안으로 확인할 수 있습니다.

`TerrainImporter`는 내부 generator를 보관하지 않아 mesh를 꺼낼 수 없습니다. 그래서 2단계용 generator를 따로 한 번 더 만듭니다. 형상은 seed와 무관하게 결정적이므로 두 결과는 같습니다.

## Code

### 새 파일: `src/spinal_policy/main/check_terrain.py`

```python
"""Verify that the G1-spinal terrain grid (floor / wall / wall_box) is generated and spawned correctly."""

import argparse

from isaaclab.app import AppLauncher

# CLI arguments (AppLauncher adds --headless, --device, ...)
parser = argparse.ArgumentParser(description="Check the G1-spinal terrain grid.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator before importing any isaaclab module
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGenerator, TerrainImporter

from spinal_policy.envs.G1.G1_spinal.G1_spinal_env_cfg import G1SpinalEnvCfg

# Height above which a vertex counts as an obstacle (not floor)
RAISED_Z = 0.01
# Tolerance for geometry comparison [m]
TOL = 1e-3


def raised_bounds(vertices: np.ndarray, origin: np.ndarray, size: tuple[float, float]) -> np.ndarray | None:
    """Return origin-relative [min, max] bounds of the raised vertices inside one tile, or None if flat."""
    # Vertices relative to the tile origin
    rel = vertices - origin
    # Mask of raised vertices inside the tile footprint
    mask = (np.abs(rel[:, 0]) < size[0] / 2) & (np.abs(rel[:, 1]) < size[1] / 2) & (rel[:, 2] > RAISED_Z)
    if not mask.any():
        return None
    return np.stack([rel[mask].min(axis=0), rel[mask].max(axis=0)])


def check_geometry(gen_cfg) -> None:
    """Check the grid shape and the per-tile geometry of the generated terrain mesh."""
    # Stand-alone generator (the importer does not keep its mesh)
    generator = TerrainGenerator(cfg=gen_cfg, device="cpu")
    # Tile origins, shape (num_rows, num_cols, 3)
    origins = generator.terrain_origins
    # All vertices of the merged terrain mesh
    vertices = np.asarray(generator.terrain_mesh.vertices)
    # Sub-terrain names in column order
    names = list(gen_cfg.sub_terrains.keys())

    # 1. Grid structure: one column per sub-terrain type
    assert origins.shape == (gen_cfg.num_rows, len(names), 3), f"unexpected terrain_origins shape {origins.shape}"
    print(f"[OK] terrain_origins shape = {origins.shape}")

    # 2. Per-tile geometry
    for col, name in enumerate(names):
        # Sub-terrain config of this column
        sub_cfg = gen_cfg.sub_terrains[name]
        for row in range(gen_cfg.num_rows):
            bounds = raised_bounds(vertices, origins[row, col], gen_cfg.size)
            print(f"  [{name:8s}] row={row} origin={np.round(origins[row, col], 2)} raised bounds=\n{bounds}")

            if name == "floor":
                assert bounds is None, f"floor tile (row {row}) has raised geometry"
                continue

            assert bounds is not None, f"{name} tile (row {row}) has no raised geometry"
            # Expected wall extents relative to the tile origin
            wall_x_near = sub_cfg.wall_distance - sub_cfg.wall_size[0] / 2
            wall_x_far = sub_cfg.wall_distance + sub_cfg.wall_size[0] / 2
            assert abs(bounds[1, 0] - wall_x_far) < TOL, f"{name}: wall far face at x={bounds[1, 0]:.3f}"
            assert abs(bounds[1, 2] - sub_cfg.wall_size[2]) < TOL or bounds[1, 2] > sub_cfg.wall_size[2], \
                f"{name}: wall top at z={bounds[1, 2]:.3f}"

            if name == "wall":
                # Only the wall is raised
                assert abs(bounds[0, 0] - wall_x_near) < TOL, f"wall near face at x={bounds[0, 0]:.3f}"
                assert abs(bounds[0, 1] + sub_cfg.wall_size[1] / 2) < TOL, f"wall y-min at {bounds[0, 1]:.3f}"
                assert abs(bounds[1, 1] - sub_cfg.wall_size[1] / 2) < TOL, f"wall y-max at {bounds[1, 1]:.3f}"
            else:
                # The box must add raised geometry in front of the wall
                assert bounds[0, 0] < wall_x_near - TOL, "wall_box tile has no box in front of the wall"
    print("[OK] per-tile geometry matches the config")


def check_spawn(importer_cfg) -> None:
    """Spawn the terrain on the stage and print the origin selected by each curriculum stage."""
    # Simulation context required by the importer
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    # Spawn the terrain mesh under /World/ground
    terrain = TerrainImporter(importer_cfg)
    # Light for visual inspection
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 3. Stage -> origin mapping (same decoding as G1SpinalEnv._update_terrain_origins)
    num_levels, num_types = terrain.terrain_origins.shape[:2]
    # Sub-terrain names in column order
    names = list(importer_cfg.terrain_generator.sub_terrains.keys())
    for stage in range(num_levels * num_types):
        level, terrain_type = stage % num_levels, stage // num_levels
        origin = terrain.terrain_origins[level, terrain_type].cpu().numpy()
        print(f"  stage {stage}: {names[terrain_type]:8s} level={level} origin={np.round(origin, 2)}")
    print("[OK] terrain spawned at", importer_cfg.prim_path)

    # Keep the window open for visual inspection
    if not args_cli.headless:
        sim.set_camera_view(eye=(25.0, 25.0, 20.0), target=(0.0, 0.0, 0.0))
        sim.reset()
        while simulation_app.is_running():
            sim.step()


def main():
    # Same terrain config as the training environment
    importer_cfg = G1SpinalEnvCfg().terrain_importer_cfg
    check_geometry(importer_cfg.terrain_generator)
    check_spawn(importer_cfg)


if __name__ == "__main__":
    main()
    simulation_app.close()
```

### 실행

```bash
# 수치 검사만 (1~3단계 출력 후 종료)
python src/spinal_policy/main/check_terrain.py --headless
# GUI에서 육안 확인까지
python src/spinal_policy/main/check_terrain.py
```

## 판정 기준

| 출력 | 의미 |
|---|---|
| `[OK] terrain_origins shape = (3, 3, 3)` | 3 type × 3 level grid가 생성됨 |
| `[OK] per-tile geometry matches the config` | floor는 평지, wall은 cfg 위치·크기, wall_box는 wall + box |
| stage 0~8 표 | stage 0~2 floor, 3~5 wall, 6~8 wall_box 순으로 서로 다른 origin |
| `AttributeError: ... 'box_size'` | 위 Conclusion의 submodule 버그. `wall_box_terrain.py` 수정 필요 |

GUI에서는 y 방향으로 floor → wall → wall_box 세 줄이, x 방향으로 level 3칸이 보여야 하고, wall은 각 칸 중심에서 +x로 2.5 m 떨어져 있어야 합니다.

## 한계

- 이 스크립트는 **terrain 생성과 stage→origin 식**만 검증합니다. 로봇이 reset 시 실제로 그 origin에 놓이는지(`_reset_idx` 순서, `scene.env_origins` 갱신)는 env가 실행 가능해진 뒤 `root_pos_w[:, :2]`와 `terrain_origins`를 비교해 따로 확인해야 합니다.
- wall top 검사는 wall_box 칸에서 box가 wall보다 높을 경우도 통과하도록 `>=`로 두었습니다.
