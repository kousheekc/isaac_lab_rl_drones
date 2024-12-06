from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class MPCVelocityControlAction(ActionTerm):
    r"""MPC based velocity control action term.

    This action term applies a velocity to the body frame.
    The raw actions correspond to vx, vy, vz, w in the body frame of the drone.

    """

    cfg: MPCVelocityControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: MPCVelocityControlActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]
        self._prop_body_ids = self._robot.find_bodies("m.*_prop")[0]

        # Mass of one drone * gravity / 4 * thrust to weight ratio
        self._upper_limit = (self._robot.root_physx_view.get_masses().sum()/self.num_envs) * env.sim._gravity_tensor.norm() / 4.0 * cfg.thrust_weight_ratio

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

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
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions.clamp(0.0, self._upper_limit)

    def apply_actions(self):
        forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = self._processed_actions

        self._robot.set_external_force_and_torque(forces, torques, body_ids=self._prop_body_ids)
        self._robot.update(self._env.physics_dt)
        self._robot.write_data_to_sim()

    def reset(self, env_ids):
        joint_pos, joint_vel = self._robot.data.default_joint_pos, self._robot.data.default_joint_vel
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # self._robot.write_root_pose_to_sim(self._robot.data.default_root_state[:, :7])
        self._robot.write_root_velocity_to_sim(self._robot.data.default_root_state[:, 7:])
        self._robot.reset()


@configclass
class MPCVelocityControlActionCfg(ActionTermCfg):
    """
    See :class:`MPCVelocityControlAction` for more details.
    """

    class_type: type[ActionTerm] = MPCVelocityControlAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    thrust_weight_ratio: float = 2.0
    """Thrust to weight ratio of the drone, actions get clipped based on this"""

