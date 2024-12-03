import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test on running the drone basic RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

from envs.drone_base.drone_base_env_cfg import DroneBaseEnvCfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = DroneBaseEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 1.0
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    env.reset()

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            thrusts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(thrusts)
            # print observations
            # print(obs)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
