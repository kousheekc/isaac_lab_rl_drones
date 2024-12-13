from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from controllers.nmpc_free_flight.controller import NMPCFreeFlightController

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class NMPCControlAction(ActionTerm):
    r"""MPC based control action term.

    This action term applies a reference trajectory to the body frame.
    The raw actions correspond to reference trajectory (pos, ori, lin_vel, ang_vel x N) in the body frame of the drone.

    """

    cfg: NMPCControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: NMPCControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self.nmpc = NMPCFreeFlightController(
            n=self.num_envs,
            control_frequency=int(1/self._env.step_dt),
            mass=float(self._robot.root_physx_view.get_masses().sum()/self.num_envs),
            moment_of_inertia=cfg.inertia,
            thrust_to_weight=cfg.thrust_weight_ratio,
            limit_min=cfg.limit_min,
            limit_max=cfg.limit_max,
            gravity=float(env.sim._gravity_tensor.norm()),
            prediction_horizon=cfg.prediction_horizon,
            control_horizon=cfg.control_horizon
        )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self.nmpc.state_dim*(self.nmpc.n_p+1)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        self._processed_actions = self._raw_actions.reshape(self.num_envs, self.nmpc.state_dim, self.nmpc.n_p+1)

        # TODO: vectorise
        for i in range(self.num_envs):        
            self._processed_actions[i, 0, :] += self._robot.data.root_state_w[i, 0]
            self._processed_actions[i, 1, :] += self._robot.data.root_state_w[i, 1]
            self._processed_actions[i, 2, :] += self._robot.data.root_state_w[i, 2]

    def apply_actions(self):
        self.nmpc.set_current(self._robot.data.root_state_w[:, :-3].detach().cpu().numpy())
        self.nmpc.set_reference(self._processed_actions.detach().cpu().numpy())
        output = self.nmpc.compute_control()

        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        torques = torch.zeros_like(forces)
        
        torques[:, 0, :] = torch.from_numpy(output[:, 1:4])
        forces[:, 0, 2] = torch.from_numpy(output[:, 0])

        self._robot.set_external_force_and_torque(forces, torques, body_ids=self._body_id)
        self._robot.update(self._env.physics_dt)
        self._robot.write_data_to_sim()

    def reset(self, env_ids):
        joint_pos, joint_vel = self._robot.data.default_joint_pos, self._robot.data.default_joint_vel
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # self._robot.write_root_pose_to_sim(self._robot.data.default_root_state[:, :7])
        self._robot.write_root_velocity_to_sim(self._robot.data.default_root_state[:, 7:])
        self._robot.reset()


@configclass
class NMPCControlActionCfg(ActionTermCfg):
    """
    See :class:`NMPCControlAction` for more details.
    """

    class_type: type[ActionTerm] = NMPCControlAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    thrust_weight_ratio: float = 2.0
    """Thrust to weight ratio of the drone, actions get clipped based on this"""
    inertia: dict = {'Ixx': 5.2, 'Iyy': 5.2, 'Izz': 3.2}
    """Inertia of drone around x, y, and z axis of body frame"""
    limit_min: list = [2.0, -6.0, -6.0, -6.0]
    """Minimum limit for outputs of NMPC controller"""
    limit_max: list = [20.0, 6.0, 6.0, 6.0]
    """Maximum limit for outputs of NMPC controller"""
    prediction_horizon: int = 20
    """Number of steps into the future NMPC will look at for optimization"""
    control_horizon: int = 1
    """Number of control steps that NMPC will return to be executed"""

