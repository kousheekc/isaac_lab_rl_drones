"""
Drone base environment.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DroneBase-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_base_env_cfg:DroneBaseEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)