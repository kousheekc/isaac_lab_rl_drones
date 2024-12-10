import argparse
import torch

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script is for test purposes only.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

from assets.cf2x import CF2X_CFG


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = CF2X_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

    robot = Articulation(robot_cfg)

    sim.reset()

    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 2000 == 0:
            # reset counters
            count = 0
            # reset drone state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            # reset command
            print(">>>>>>>> Reset!")

        print(robot.data.root_state_w.tolist())
        print()
        # apply action to the robot (make the robot float in place)
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = robot_mass * gravity / 4
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
