import gymnasium as gym

gym.register(
    id="G1-spinal", 
    entry_point=f"{__name__}.G1_spinal_env:G1SpinalEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_spinal_env_cfg:G1SpinalEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)

gym.register(
    id="G1-spinal-play", 
    entry_point=f"{__name__}.G1_spinal_env:G1SpinalEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_spinal_env_cfg:G1SpinalEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)