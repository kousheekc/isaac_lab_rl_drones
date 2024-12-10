import time
import numpy as np
from controllers.nmpc.model.body_rate import NMPCBodyRateController

theta = np.linspace(0, 0.25 * np.pi, 10+1)
reference_trajectory = np.vstack((np.cos(theta), np.sin(theta), np.ones_like(theta), np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta)))

nmpc = NMPCBodyRateController(4, 100, 2, {'Ixx': 7, 'Iyy': 7, 'Izz': 3}, 2.0, [2.0, -6.0, -6.0, -6.0], [20.0, 6.0, 6.0, 6.0], 9.81, 10, 1)

init_time = time.time()

nmpc.set_current(np.array([
    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]))

nmpc.set_reference(np.array([
    reference_trajectory,
    reference_trajectory,
    reference_trajectory,
    reference_trajectory,
]))

print(time.time() - init_time)
print(nmpc.compute_control())
print(nmpc.compute_control())
print(nmpc.compute_control())