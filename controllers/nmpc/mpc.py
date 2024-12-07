import numpy as np
import casadi as ca

class MPC:
    def __init__(self, prediction_horizon, dt, solver_path):
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.solver_path = solver_path

        self.n = int(self.prediction_horizon/self.dt)

        # x = state = position (px, py, pz), quaternion (qw, qx, qy, qz), linear velocity (vx, vy, vz), angular velociy (wx, wy, wz)
        self.x_dim = 13
        self.u_dim = 4

        self.x0 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.u0 = [0.0, 0.0, 0.0, 0.0]

        self.x_cost = np.diag([0, 0, 0, 
                               10, 10, 10, 10, 
                               100, 100, 100,
                               10, 10, 10])
        
        self.u_cost = np.diag([0.1, 0.1, 0.1, 0.1])

        self.g = 9.81 # gravity
        self.dx = [0.1, 0.1, 0.1, 0.1] # distances to each motor from body frame
        self.dy = [0.1, 0.1, 0.1, 0.1]
        self.c = 10 # rotor drag torque constant
        self.m = 0.250 # mass

        self.init()

    def init(self):
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        wx, wy, wz = ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')

        f0, f1, f2, f3 = ca.SX.sym('f1'), ca.SX.sym('f2'), ca.SX.sym('f3'), ca.SX.sym('f4')

        self.x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz)
        self.u = ca.vertcat(f0, f1, f2, f3)

        x_dot = ca.vertcat(
            vx,
            vy,
            vz,
            0.5 * (-wx*qx - wy*qy - wz*qz),
            0.5 * ( wx*qw + wz*qy - wy*qz),
            0.5 * ( wy*qw - wz*qx + wx*qz),
            0.5 * ( wz*qw + wy*qx - wx*qy),
            1/self.j[0][0] * (-self.dx[0]*f0 - self.dx[1]*f1 + self.dx[2]*f2 + self.dx[3]*f3 + wy*wz*(self.j[1][1] - self.j[2][2])),
            1/self.j[1][1] * ( self.dy[0]*f0 - self.dy[1]*f1 - self.dy[2]*f2 + self.dx[3]*f3 + wx*wz*(self.j[2][2] - self.j[0][0])),
            1/self.j[2][2] * (-self.c*f0     + self.c*f1     - self.c*f2     + self.c*f3     + wx*wy*(self.j[0][0] - self.j[1][1])),
            (2 * (f0+f1+f2+f3) / self.m) * (qy*qw + qz*qx),
            (2 * (f0+f1+f2+f3) / self.m) * (qy*qz - qw*qx),
            ((f0+f1+f2+f3) / self.m) * (qw*qw - qx*qx - qy*qy + qz*qz) - self.g
        )

        f = ca.Function('f', [self.x, self.u], [x_dot], ['x', 'u'], ['ode'])
        F = self.runge_kutta_4(f, self.dt)
        fMap = F.map(self.n, "openmp") # parallel

        # # # # # # # # # # # # # # # 
        # ---- loss function --------
        # # # # # # # # # # # # # # # 
        delta_x = ca.SX.sym("delta_x", self.x_dim)
        delta_u = ca.SX.sym("delta_u", self.u_dim)        
        
        cost_x = delta_x.T @ self.x_cost @ delta_x 
        cost_u = delta_u.T @ self.u_cost @ delta_u

        f_cost_x = ca.Function('cost_x', [delta_x], [cost_x])
        f_cost_u = ca.Function('cost_u', [delta_u], [cost_u])

    def runge_kutta_4(self, f, dt):
        M = 4
        DT = dt/M
        X0 = ca.SX.sym("X", self.x_dim)
        U = ca.SX.sym("U", self.u_dim)

        X = X0
        # --------- RK4------------
        for _ in range(M):
            k1 =DT*f(X, U)
            k2 =DT*f(X+0.5*k1, U)
            k3 =DT*f(X+0.5*k2, U)
            k4 =DT*f(X+k3, U)
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6        

        F = ca.Function('F', [X0, U], [X])
        return F











