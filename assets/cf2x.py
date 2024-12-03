import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg


CF2X_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/usds/cf2x.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "m1_joint": 0.0,
            "m2_joint": 0.0,
            "m3_joint": 0.0,
            "m4_joint": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "actuator": ImplicitActuatorCfg(
            joint_names_expr=["m1_joint", "m2_joint", "m3_joint", "m4_joint"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)