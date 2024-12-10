import time
import numpy as np
from controllers.nmpc.model.body_rate import NMPCBodyRateController

# theta = np.linspace(0, 0.5 * np.pi, 20+1)
# reference_trajectory = np.vstack((np.cos(theta), np.sin(theta), np.ones_like(theta), np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta)))

nmpc = NMPCBodyRateController(1, 60, 2, {'Ixx': 7, 'Iyy': 7, 'Izz': 3}, 2.0, [0.0, -60.0, -60.0, -60.0], [20.0, 60.0, 60.0, 60.0], 9.81, 200, 1)

init_time = time.time()

nmpc.set_current(np.array([
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]))

a = np.zeros((10, 201))
a[0] = np.arange(0.0, 0.201, 0.001)
a[2] = 1.0
a[3] = 1.0

nmpc.set_reference(np.array([
    a,
]))

u, sol = nmpc.compute_control()

print(time.time() - init_time)