from controllers.base_controller import BaseController
from casadi import *
import numpy as np

class NMPCBodyRateController(BaseController):
    def __init__(self, control_frequency: float, mass: float, moment_of_inertia: dict, thrust_to_weight: float, limit_min: list, limit_max: list, gravity: float, prediction_horizon: int, control_horizon: int):
        super().__init__(control_frequency, mass, moment_of_inertia, thrust_to_weight, limit_min, limit_max, gravity)
        
        assert control_horizon < prediction_horizon - 1, f"Control horizon: {control_horizon} muct be smaller than prediction horizon - 1: {prediction_horizon - 1}"  # This will pass

        self.dt = 1/self._control_frequency
        self.n_p = prediction_horizon
        self.n_u = control_horizon

        self.opti = Opti()

        self.x = self.opti.variable(10, self.n_p+1) # state trajectory
        self.u = self.opti.variable(4, self.n_p)   # control trajectory

        self.x_ref = self.opti.parameter(10, self.n_p+1)  # Reference trajectory
        self.u_ref = self.opti.parameter(4, self.n_p) # Reference controls

        f = lambda x,u: vertcat(
            x[7],
            x[8],
            x[9],
            0.5 * ( -u[1]*x[4] - u[2]*x[5] - u[3]*x[6] ),
            0.5 * (  u[1]*x[3] + u[3]*x[5] - u[2]*x[6] ),
            0.5 * (  u[2]*x[3] - u[3]*x[4] + u[1]*x[6] ),
            0.5 * (  u[3]*x[3] + u[2]*x[4] - u[1]*x[5] ),
            2 * ( x[3]*x[5] + x[4]*x[6] ) * u[0],
            2 * ( x[5]*x[6] - x[3]*x[4] ) * u[0], 
            (x[3]*x[3] - x[4]*x[4] -x[5]*x[5] + x[6]*x[6]) * u[0] - self.gravity
        ) # dx/dt = f(x,u)

        for k in range(self.n_p):
            # Runge-Kutta 4 integration
            # Discretise
            k1 = f(self.x[:,k],         self.u[:,k])
            k2 = f(self.x[:,k]+self.dt/2*k1, self.u[:,k])
            k3 = f(self.x[:,k]+self.dt/2*k2, self.u[:,k])
            k4 = f(self.x[:,k]+self.dt*k3,   self.u[:,k])
            x_next = self.x[:,k] + self.dt/6*(k1+2*k2+2*k3+k4) 
            self.opti.subject_to(self.x[:,k+1]==x_next) # ensure dynamics

        self.opti.subject_to(self.opti.bounded(self.limit_min[0], self.u[0, :], self.limit_max[0])) # control is limited
        self.opti.subject_to(self.opti.bounded(self.limit_min[1], self.u[1, :], self.limit_max[1])) # angular velocity is limited
        self.opti.subject_to(self.opti.bounded(self.limit_min[2], self.u[2, :], self.limit_max[2])) # angular velocity is limited
        self.opti.subject_to(self.opti.bounded(self.limit_min[3], self.u[3, :], self.limit_max[3])) # angular velocity is limited
        self.opti.subject_to(self.x[:,0] == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])     # Initial state
        self.opti.subject_to(self.u[:,0] == [9.81, 0.0, 0.0, 0.0])                                  # Initial state
        
        x_weight = diag(MX([100, 100, 100, 10, 10, 10, 10, 10, 10, 10]))  # Weights for state error
        u_weight = diag(MX([0.1, 0.1, 0.1, 0.1]))                         # Weights for control effort

        # Cost function calculations
        cost = 0 # Cost function init
        for k in range(self.n_p):
            state_error = self.x[:, k] - self.x_ref[:, k]
            cost += mtimes(state_error.T, mtimes(x_weight, state_error))
            control_error = self.u[:, k] - self.u_ref[:, k]
            cost += mtimes(control_error.T, mtimes(u_weight, control_error))
        self.opti.minimize(cost)

        # reference controls (dont change)
        reference_controls = np.tile(np.array([9.81, 0.0, 0.0, 0.0]), (self.n_p, 1)).T # Nominal control for hovering
        self.opti.set_value(self.u_ref, reference_controls)

        self.opti.solver("ipopt") # set numerical backend

    def set_current(self, current_state):
        super().set_current(current_state)
        self.opti.subject_to(self.x[:,0] == current_state)     # Current state

    def set_reference(self, reference_state):
        super().set_reference(reference_state)
        self.opti.set_value(self.x_ref, self.reference_state)

    def compute_control(self, timestamp = None):
        self.opti.set_initial(self.x, 0.0)  # Initial guess
        sol = self.opti.solve()             # actual solve

        self.opti.subject_to(self.u[:,0] == sol.value(self.u[:, self.n_u-1]))
        self.control_output = sol.value(self.u[:, self.n_u-1])

        return sol.value(self.u[:, 0:self.n_u])


        