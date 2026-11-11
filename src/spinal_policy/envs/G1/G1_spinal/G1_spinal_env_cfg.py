
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from lib.utils.terrain_utils import *

from lib.curriculum import CurriculumManagerCfg, CurriculumParamCfg
from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg
from lib.terrain.wall_box_terrain import *
from lib.utils.plot_utils import PNGSavePlotter

# Training Environment
@configclass
class G1SpinalEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 1/200
    decimation = 4         

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 17, "leg": 12}                         
    observation_space = {"arm": 63, "leg": 48}                
    state_space = {"arm": 100, "leg": 100}
    num_agents = 2
    action_scale_factor = {"arm": [0.5, ()], 
                           "leg": [0.5, ()]}

    ## ========== Single Agent Setting ========== ##  
    # action_space = 29                     
    # observation_space = 101                 
    # num_agents = 1
    # action_scale_factor = 0.5

    ## ==================== Reward Shaping ==================== ##
    # Common rewards
    w_upright:           float = 2.0
    w_track_height : float = 1.0
    w_ang_vel_xy:         float = 0.1
    w_lin_vel_z:          float = 2.0
    w_termination: float = 200

    # Arm rewards
    w_deviation_arm:     float = 2.0
    w_limits:            float = 10.0
    w_joint_torque_limit: float = 1.0e-5
    w_joint_torque:       float = 1.0e-7
    w_joint_vel:          float = 5.0e-4
    w_deviation_torso:   float = 2.0
    w_action_rate:       float = 1.0e-3

    # Leg rewards
    w_track_lin_vel: float = 4.5
    w_support_xy:     float = 1.0
    w_deviation_hip:     float = 2.0
    w_track_heading: float = 4.0
    # w_limits
    # w_joint_torque_limit
    # w_joint_torque
    # w_joint_vel
    w_single_contact:     float = 1.0
    w_feet_air_time:      float = 1.0
    # w_ation_rate
    
    termination_height: float = 0.35
    termination_gravity: float = 0.8
    termination_ang_vel: float = 20.0

    ## ============== Self collision =============== ##
    allowed_collision_bodies = ["left_ankle_pitch_link",
                                "left_ankle_roll_link",
                                "right_ankle_pitch_link",
                                "right_ankle_roll_link", 
                                "waist_yaw_link",
                                "waist_roll_link"]
    
    forbidden_collision_bodies = ["head_link",
                                  "left_rubber_hand",
                                  "right_rubber_hand",
                                  "torso_link",
                                  "left_shoulder_pitch_link",
                                  "right_shoulder_pitch_link",
                                  "pelvis_contour_link",        # hip relative links
                                  "pelvis",
                                  "left_hip_pitch_link",
                                  "right_hip_pitch_link"]

    # Commander
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=0.0,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.0, 0.0), heading=(0.0, 0.0)
        ),
    )

    # visualization
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_ang_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angular_velocity_goal"
    )
    current_ang_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angular_velocity_current"
    )

    # Set the scale of the visualization markers
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
    goal_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    def __post_init__(self):
        super().__post_init__()
        
        # Push event
        self.events.push_robot.interval_range_s = (4.0, 5.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), 
            "y": (-0.5, 0.5), 
            "roll": (-2.5, 2.5), 
            "pitch": (-2.5, 2.5)
        }

        # Curriculum
        num_rows = self.terrain.terrain_generator.num_rows      # Level
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

    ## =============== Terrain =============== ##
    flat = FlatTerrain()
    
    wall_box = WallBoxTerrain(wall_distance = 2.5,
                              box_size_range = (0.5, 0.8),
                              box_position = (0.6, 0.6, 0.6))
    
    # pyramid = PyramidStairsTerrain(border_width=0.1,
    #                                 step_height_range=(0.1, 0.3),
    #                                 step_width=0.2,
    #                                 platform_width=0.3,
    #                                 holes=False)
    # inverted_pyramid = InvertedPyramidStairsTerrain(border_width=0.1,
    #                                                 step_height_range=(0.1, 0.3),
    #                                                 step_width=0.2,
    #                                                 platform_width=0.3,
    #                                                 holes=False)

    terrain_generator_cfg = TerrainGeneratorCfg(
        curriculum = True,
        size=[10.0, 10.0],
        num_rows = 3,                               # curriculum level for each terrain
        num_cols = 1,                               # variation of terrain
        color_scheme="random",
        sub_terrains = {
            "floor": SubTerrainBaseCfg(function=flat.generate, proportion=1.0),
            "wall": WallBoxTerrainCfg(function=wall_box.wall_generate(),proportion=1.0),
            "wall_box": WallBoxTerrainCfg(function=wall_box.wall_box_generate() ,proportion=1.0)
        }
    )

    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.3,                 # Collision
        ),
        debug_vis=False
    )

# Eveluation Envrironment
@configclass
class G1SpinalPlayEnvCfg(G1SpinalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
    
        # viewer
        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0)
        )

        self.curriculum = None
        
        self.scene.num_envs = 1

        self.plotter = PNGSavePlotter

        self.viz_data = {
            "body_angular_momentum": 0.0,
            "left_shoulder_joint_pos": 0.0,
            "right_shoulder_joint_pos": 0.0,}

        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-4.0, 4.0),
            "pitch": (-4.0, 4.0)
        }
