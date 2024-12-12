import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass

import envs.drone_base.mdp as mdp

from assets.cf2x import CF2X_CFG

@configclass
class DroneBaseSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )

    # drone
    robot: ArticulationCfg = CF2X_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # thrust_control_action: mdp.ThrustControlActionCfg = mdp.ThrustControlActionCfg(asset_name="robot")
    body_torque_control_action: mdp.BodyTorqueControlActionCfg = mdp.BodyTorqueControlActionCfg(asset_name="robot")
    # nmpc_control_action: mdp.NMPCControlActionCfg = mdp.NMPCControlActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_position = ObsTerm(func=mdp.root_pos_w)
        base_quaternion = ObsTerm(func=mdp.root_quat_w)
        base_linear_vel = ObsTerm(func=mdp.root_lin_vel_w)
        base_angular_vel = ObsTerm(func=mdp.root_ang_vel_w)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Penalty for large changes action commands
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # (4) Penalty for not remaining flat
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    # (5) Penatly for moving away from target height
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 1.0})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Crash with ground
    crash = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.05})


@configclass
class DroneBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the drone environment."""

    # Scene settings
    scene: DroneBaseSceneCfg = DroneBaseSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
