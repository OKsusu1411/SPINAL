
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg
from lib.utils.plot_utils import PNGSavePlotter

# Training Environment
@configclass
class G1GaitEnvCfg(G1BaseEnvCfg):
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
    w_track_lin_vel: float = 4.0
    w_track_heading: float = 4.0
    w_track_height : float = 1.0
    w_feet_gait:      float = 6.0
    w_support_xy:     float = 1.0
    w_upright:           float = 2.0
    w_single_contact:     float = 1.0
    w_feet_air_time:      float = 1.0

    w_lin_vel_z:          float = 2.0
    w_ang_vel_xy:         float = 0.1
    w_joint_torque:       float = 1.0e-7
    w_joint_torque_limit: float = 1.0e-5
    w_joint_vel:          float = 1.0e-4


    w_limits:            float = 10.0
    w_deviation_swing:   float = 2.0
    w_deviation_hip:     float = 2.0
    w_deviation_torso:   float = 2.0
    w_deviation_arm:     float = 2.0
    w_action_rate:       float = 1.0e-3

    w_termination: float = 200
    termination_height: float = 0.35
    termination_gravity: float = 0.8
    termination_ang_vel: float = 20.0

    # ===== Gait guidance ===== #
    time_period = 0.35
    z_c = 0.75
    single_contact_window = 0.2     # [s] look-back window for single-foot contact
    air_time_threshold = 0.4        # [s] target swing duration

    ## ============== Self collision =============== ##
    allowed_collision_bodies = ["left_ankle_pitch_link",
                                "left_ankle_roll_link",
                                "right_ankle_pitch_link",
                                "right_ankle_roll_link", 
                                "waist_yaw_link",
                                "waist_roll_link"]

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
        self.events.push_robot.interval_range_s = (4.0, 5.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), 
            "y": (-0.5, 0.5), 
            "roll": (-2.5, 2.5), 
            "pitch": (-2.5, 2.5)
        }

# Eveluation Envrironment
@configclass
class G1GaitPlayEnvCfg(G1GaitEnvCfg):
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
