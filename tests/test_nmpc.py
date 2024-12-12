import time
import numpy as np
from controllers.nmpc.model.body_rate import NMPCBodyRateController


nmpc = NMPCBodyRateController(
    n=1,
    control_frequency=100,
    mass=2.5,
    moment_of_inertia={'Ixx': 3, 'Iyy': 3, 'Izz': 7},
    thrust_to_weight=2.0,
    limit_min=[2.0, -6.0, -6.0, -6.0],
    limit_max=[20.0, 6.0, 6.0, 6.0],
    gravity=9.81,
    prediction_horizon=200,
    control_horizon=1
)


current_state = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# start = 0.0
# end = 0.1
# step = step = (end - start) / (nmpc.n_p)
# reference_trajectory = np.zeros((nmpc.state_dim, nmpc.n_p+1))
# reference_trajectory[0] = np.arange(start, end+step, step)
# reference_trajectory[3] = 1.0
# reference_trajectory[:3] += current_state[:3, np.newaxis]

theta = np.linspace(0, 2 * np.pi, nmpc.n_p+1)
reference_trajectory = np.vstack((np.cos(theta), np.sin(theta), np.ones_like(theta), np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta)))

print(reference_trajectory)

nmpc.set_current(np.array([
    current_state,
]))

nmpc.set_reference(np.array([
    reference_trajectory,
]))

init_time = time.time()

u = nmpc.compute_control()

print(time.time() - init_time)