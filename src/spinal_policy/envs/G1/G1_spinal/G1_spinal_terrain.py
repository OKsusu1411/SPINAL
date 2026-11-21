from __future__ import annotations

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
from isaaclab.terrains.utils import create_prim_from_mesh

from lib.terrain.wall_box_terrain import WallBoxTerrain, WallBoxTerrainCfg
from .G1_spinal_env_cfg import G1SpinalEnvCfg

@configclass
class SurfaceMaterialRangeCfg:
    """Sampling ranges of one terrain surface physics material."""

    static_friction_range: tuple[float, float] = (0.8, 1.2)    # Static friction bounds
    dynamic_friction_range: tuple[float, float] = (0.6, 0.8)   # Dynamic friction bounds
    restitution_range: tuple[float, float] = (0.0, 0.3)        # Restitution bounds
    friction_combine_mode: str = "multiply"                    # PhysX friction combine mode
    restitution_combine_mode: str = "average"                  # PhysX restitution combine mode


@configclass
class SpinalTerrainTileCfg:
    """Final (already sampled) values of one terrain tile."""

    terrain_type_num: int = 0                                    # 0=Floor, 1=Wall, 2=Wall+Box
    variant: int = 0                                             # Variant row index
    wall_size: tuple[float, float, float] | None = None          # Wall extents, None if absent
    wall_position: tuple[float, float, float] | None = None      # Wall center in tile frame
    box_size: tuple[float, float, float] | None = None           # Box extents, None if absent
    box_position: tuple[float, float, float] | None = None       # Box center in tile frame
    floor_material: sim_utils.RigidBodyMaterialCfg = None        # Floor physics material
    wall_material: sim_utils.RigidBodyMaterialCfg | None = None  # Wall physics material
    box_material: sim_utils.RigidBodyMaterialCfg | None = None   # Box physics material

class G1SpinalTerrain:
    """Terrain tile API: sample final tiles from the env cfg and spawn them."""

    def __init__(self):
        self.tiles: list[SpinalTerrainTileCfg] = []    # Prepared final tiles

    # def prepare_tiles(self, cfg: G1SpinalEnvCfg) -> list[SpinalTerrainTileCfg]:
    #     """Sample the final terrain tile list once, before the scene is created."""
    #     # Direct config checks (curriculum steps, finite positive geometry, valid variant count)
    #     steps = list(cfg.terrain_stage_steps)    # Progress thresholds of stage switches

    #     # ================ Debugging ================ #
    #     if not (len(steps) == 3 and all(math.isfinite(s) for s in steps) and 0.0 < steps[0] < steps[1] < steps[2] < 1.0):
    #         raise ValueError(f"terrain_stage_steps must be 3 finite, strictly increasing values in (0, 1): {steps}")
    #     if not (isinstance(cfg.terrain_num_variants, int) and cfg.terrain_num_variants >= 1):
    #         raise ValueError(f"terrain_num_variants must be an int >= 1: {cfg.terrain_num_variants}")
    #     if not all(math.isfinite(s) and s > 0.0 for s in (*cfg.terrain_tile_size, *cfg.wall_size)):
    #         raise ValueError(f"Tile/wall sizes must be finite and positive: {cfg.terrain_tile_size}, {cfg.wall_size}")
    #     if not (math.isfinite(cfg.terrain_spawn_clearance) and cfg.terrain_spawn_clearance >= 0.0):
    #         raise ValueError(f"terrain_spawn_clearance must be finite and >= 0: {cfg.terrain_spawn_clearance}")
    #     # ================ Debugging ================ #

    #     seed = cfg.terrain_seed if cfg.terrain_seed is not None else cfg.seed    # Terrain RNG seed
    #     rng = np.random.default_rng(seed)    # Terrain RNG
    #     self.tiles = []
    #     for terrain_type in range(3):
    #         for variant in range(cfg.terrain_num_variants):
    #             # Sample every value in a fixed order regardless of tile type
    #             wall_position = (self._sample(rng, cfg.wall_x_range), self._sample(rng, cfg.wall_y_range),
    #                              cfg.wall_size[2] / 2.0)
    #             edge = self._sample(rng, cfg.box_edge_range)
    #             box_position = (self._sample(rng, cfg.box_x_range), self._sample(rng, cfg.box_y_range), edge / 2.0)
    #             materials = [self._sample_material(rng, r) for r in
    #                          (cfg.floor_material_range, cfg.wall_material_range, cfg.box_material_range)]
    #             tile = SpinalTerrainTileCfg(terrain_type=terrain_type, variant=variant, floor_material=materials[0])
    #             if terrain_type >= 1:
    #                 tile.wall_size, tile.wall_position, tile.wall_material = cfg.wall_size, wall_position, materials[1]
    #             if terrain_type == 2:
    #                 tile.box_size, tile.box_position, tile.box_material = (edge, edge, edge), box_position, materials[2]
                
    #             # ================ Debugging ================ #
    #             self._validate_tile(cfg, tile)
    #             # ================ Debugging ================ #

    #             self.tiles.append(tile)
    #     return self.tiles

    def spawn_tiles(self, cfg: G1SpinalEnvCfg, device: str) -> torch.Tensor:
        """Spawn every prepared tile and return tile origins shaped (num_variants, 3, 3)."""
        # Terrain config
        terrain_type = cfg.terrain_type
        terrain_color = {"floor": (0.3, 0.3, 0.3), "wall": (0.6, 0.45, 0.3), "box": (0.25, 0.4, 0.6)}

        seed = cfg.terrain_seed if cfg.terrain_seed is not None else cfg.seed    # Terrain RNG seed
        self.tiles = []
        for terrain_type_num in range(len(terrain_type)):
            for variant in range(cfg.terrain_num_variants):
                # Sample every value in a fixed order regardless of tile type
                wall_position = (self._sample(cfg.wall_x_range, seed), self._sample(cfg.wall_y_range, seed),
                                cfg.wall_size[2] / 2.0)
                edge = self._sample(cfg.box_edge_range, seed)
                box_position = (self._sample(cfg.box_x_range, seed), self._sample(cfg.box_y_range, seed), edge / 2.0)
                materials = [self._sample_material(r, seed) for r in
                            (cfg.floor_material_range, cfg.wall_material_range, cfg.box_material_range)]
                tile = SpinalTerrainTileCfg(terrain_type_num=terrain_type_num, variant=variant, floor_material=materials[0])
                if terrain_type_num >= 1:
                    tile.wall_size, tile.wall_position, tile.wall_material = cfg.wall_size, wall_position, materials[1]
                if terrain_type_num == 2:
                    tile.box_size, tile.box_position, tile.box_material = (edge, edge, edge), box_position, materials[2]
                
                # ================ Debugging ================ #
                self._validate_tile(cfg, tile)
                # ================ Debugging ================ #

                self.tiles.append(tile)

        num_variants, tile_size = cfg.terrain_num_variants, cfg.terrain_tile_size    # Grid layout
        origins = torch.zeros((num_variants, 3, 3), dtype=torch.float, device=device)
        wall_box = WallBoxTerrain()    # Stateless mesh callbacks
        for tile in self.tiles:
            # Tile center in world frame (grid centered at the world origin)
            center = np.array([(tile.variant + 0.5 - num_variants / 2.0) * tile_size[0],
                               (tile.terrain_type_num + 0.5 - 1.5) * tile_size[1], 0.0])
            mesh_cfg = WallBoxTerrainCfg(size=tile_size)    # Per-tile final geometry
            if tile.terrain_type_num == 0:
                meshes, local_origin = flat_terrain(0.0, mesh_cfg)
            else:
                mesh_cfg.wall_size, mesh_cfg.wall_position = tile.wall_size, tile.wall_position
                if tile.terrain_type_num == 1:
                    meshes, local_origin = wall_box.wall_generate(0.0, mesh_cfg)
                else:
                    mesh_cfg.box_size, mesh_cfg.box_position = tile.box_size, tile.box_position
                    meshes, local_origin = wall_box.wall_box_generate(0.0, mesh_cfg)
            materials = (tile.floor_material, tile.wall_material, tile.box_material)
            for name, mesh, material in zip(terrain_type, meshes, materials):
                # Move from the tile-corner frame to the world tile center
                mesh.apply_translation(center - local_origin)
                create_prim_from_mesh(f"{cfg.terrain_prim_path}/tile_{tile.variant}_{tile.terrain_type_num}/{name}", mesh,
                                      physics_material=material,
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=terrain_color[name]))
            origins[tile.variant, tile.terrain_type_num] = torch.tensor(center, dtype=torch.float, device=device)
        return origins

    # ================ Auxillary ================ #
    @staticmethod
    def _sample(value_range: tuple[float, float], seed: int) -> float:
        """Sample a uniform float from a validated (low, high) range."""
        low, high = value_range
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError(f"Invalid terrain range: {value_range}")
        randomizer = np.random.default_rng(seed)    # Terrain Randomizer
        return float(randomizer.uniform(low, high))

    def _sample_material(self, cfg: SurfaceMaterialRangeCfg, seed: int) -> sim_utils.RigidBodyMaterialCfg:
        """Sample a new physics material object from the surface ranges."""
        return sim_utils.RigidBodyMaterialCfg(
            static_friction=self._sample(cfg.static_friction_range, seed),
            dynamic_friction=self._sample(cfg.dynamic_friction_range, seed),
            restitution=self._sample(cfg.restitution_range, seed),
            friction_combine_mode=cfg.friction_combine_mode,
            restitution_combine_mode=cfg.restitution_combine_mode,)

    # ================ Debugging ================ #
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
