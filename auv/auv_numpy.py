import numpy as np
import numpy.linalg as LA
import yaml
from numpy import cos, fabs, sin, tan

from utils.utils import skew


class AUV(object):
    def __init__(self, vehicle_dynamics):
        """Initializes the AUV object.

        Args:
            vehicle_dynamics: Dictionary containing the vehicle dynamics parameters.
        """
        self.vehicle_mass = vehicle_dynamics["vehicle_mass"]
        self.rb_mass = vehicle_dynamics["rb_mass"]
        self.added_mass = vehicle_dynamics["added_mass"]
        self.lin_damp = vehicle_dynamics["lin_damp"]
        self.quad_damp = vehicle_dynamics["quad_damp"]
        self.tam = vehicle_dynamics["tam"]
        self.inertial_terms = vehicle_dynamics["inertial_terms"]
        self.inertial_skew = vehicle_dynamics["inertial_skew"]
        self.r_gb_skew = vehicle_dynamics["r_gb_skew"]
        self.W = vehicle_dynamics["W"]
        self.B = vehicle_dynamics["B"]
        self.cog = vehicle_dynamics["cog"]
        self.cob = vehicle_dynamics["cob"]
        self.cog_to_cob = self.cog - self.cob
        self.neutral_bouy = vehicle_dynamics["neutral_bouy"]
        self.curr_timestep = 0.0
        self.ocean_current_data = []

        # Precompute the total mass matrix (w/added mass) inverted for future dynamic calls
        self.mass_inv = LA.inv(self.rb_mass + self.added_mass)
        self.rb_mass_inv = LA.inv(self.rb_mass)
        self.added_mass_inv = LA.inv(self.added_mass)

    @classmethod
    def load_params(cls, filename):
        f = open(filename, "r")
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)

        m = params["m"]
        Ixx = params["Ixx"]
        Iyy = params["Iyy"]
        Izz = params["Izz"]
        I_b = np.eye(3) * [Ixx, Iyy, Izz]

        skew_r_gb = skew(params["cog"])
        skew_I = skew([Ixx, Iyy, Izz])

        rb_mass = np.diag([m, m, m, Ixx, Iyy, Izz])
        rb_mass[0:3, 3:6] = -m * skew_r_gb
        rb_mass[3:6, 0:3] = m * skew_r_gb

        vehicle_dynamics = {}
        vehicle_dynamics["vehicle_mass"] = m
        vehicle_dynamics["rb_mass"] = rb_mass
        vehicle_dynamics["added_mass"] = -np.diag(params["m_added"])
        vehicle_dynamics["lin_damp"] = -np.diag(params["d_lin"])
        vehicle_dynamics["quad_damp"] = -np.diag(params["d_quad"])
        vehicle_dynamics["tam"] = np.array(params["tam"])
        vehicle_dynamics["r_gb_skew"] = skew_r_gb
        vehicle_dynamics["inertial_skew"] = skew_I
        vehicle_dynamics["inertial_terms"] = I_b
        vehicle_dynamics["W"] = params["W"]
        vehicle_dynamics["B"] = params["B"]
        vehicle_dynamics["cog"] = np.array(params["cog"])
        vehicle_dynamics["cob"] = np.array(params["cob"])
        vehicle_dynamics["neutral_bouy"] = params["neutral_bouy"]

        return cls(vehicle_dynamics)

    def compute_transformation_matrix(self, x):
        """Computes the transformation matrix from Body to NED frame.

        Args:
            x: State vector.

        Returns:
            tf_mtx: Transformation matrix.
        """
        rot_mtx = np.eye(3)
        rot_mtx[0, 0] = cos(x[5]) * cos(x[4])
        rot_mtx[0, 1] = -sin(x[5]) * cos(x[3]) + cos(x[5]) * sin(x[4]) * sin(x[3])
        rot_mtx[0, 2] = sin(x[5]) * sin(x[3]) + cos(x[5]) * cos(x[3]) * sin(x[4])
        rot_mtx[1, 0] = sin(x[5]) * cos(x[4])
        rot_mtx[1, 1] = cos(x[5]) * cos(x[3]) + sin(x[3]) * sin(x[4]) * sin(x[5])
        rot_mtx[1, 2] = -cos(x[5]) * sin(x[3]) + sin(x[4]) * sin(x[5]) * cos(x[3])
        rot_mtx[2, 0] = -sin(x[4])
        rot_mtx[2, 1] = cos(x[4]) * sin(x[3])
        rot_mtx[2, 2] = cos(x[4]) * cos(x[3])

        t_mtx = np.eye(3)
        t_mtx[0, 1] = sin(x[3]) * tan(x[4])
        t_mtx[0, 2] = cos(x[3]) * tan(x[4])
        t_mtx[1, 1] = cos(x[3])
        t_mtx[1, 2] = -sin(x[3])
        t_mtx[2, 1] = sin(x[3]) / cos(x[4])
        t_mtx[2, 2] = cos(x[3]) / cos(x[4])

        tf_mtx = np.eye(6)
        tf_mtx[0:3, 0:3] = rot_mtx
        tf_mtx[3:, 3:] = t_mtx

        return tf_mtx

    def compute_C_RB_force(self, v):
        """Computes the Rigid-body Coriolis force matrix.

        Args:
            v: Velocity vector.

        Returns:
            coriolis_force: Rigid-body Coriolis force matrix.
        """
        v1 = v[0:3, 0]
        v2 = v[3:6, 0]

        skew(v1)
        skew_v2 = skew(v2)
        skew_I_v2 = skew((self.inertial_terms @ v2))

        coriolis_force = np.zeros((6, 6))
        # coriolis_force[0:3, 3:6] = -self.vehicle_mass * (skew_v1 + mtimes(skew_v2, self.r_gb_skew))
        # coriolis_force[3:6, 0:3] = -self.vehicle_mass * (skew_v1 - mtimes(self.r_gb_skew, skew_v2))
        coriolis_force[0:3, 0:3] = self.vehicle_mass * skew_v2
        coriolis_force[0:3, 3:6] = -self.vehicle_mass * (skew_v2 @ self.r_gb_skew)
        coriolis_force[3:6, 0:3] = self.vehicle_mass * (self.r_gb_skew @ skew_v2)
        coriolis_force[3:6, 3:6] = -skew_I_v2
        return coriolis_force

    def compute_C_A_force(self, v):
        """Computes the Added-mass Coriolis force matrix.

        Args:
            v: Velocity vector.

        Returns:
            coriolis_force: Added-mass Coriolis force matrix.
        """
        v1 = v[0:3, 0]
        v2 = v[3:6, 0]

        np.diag(self.added_mass)

        A_11 = self.added_mass[0:3, 0:3]
        A_12 = self.added_mass[0:3, 3:6]
        A_21 = self.added_mass[3:6, 0:3]
        A_22 = self.added_mass[3:6, 3:6]

        coriolis_force = np.zeros((6, 6))
        # coriolis_force[0:3, 3:6] = -skew(MA_diag[0:3] @ v1)
        # coriolis_force[3:6, 0:3] = -skew(MA_diag[0:3] @ v1)
        # coriolis_force[3:6, 3:6] = -skew(MA_diag[3:6] @ v2)
        coriolis_force[0:3, 3:6] = -skew((A_11 @ v1) + (A_12 @ v2))
        coriolis_force[3:6, 0:3] = -skew((A_11 @ v1) + (A_12 @ v2))
        coriolis_force[3:6, 3:6] = -skew((A_21 @ v1) + (A_22 @ v2))
        return coriolis_force

    def compute_damping_force(self, v):
        """Computes the damping force.

        Args:
            v: velocity vector.

        Returns:
            damping_force: Damping force.
        """
        damping_force = (self.quad_damp * fabs(v)) + self.lin_damp
        return damping_force

    def compute_restorive_force(self, x):
        """Computes the restorive force.

        Args:
            x: State vector.

        Returns:
            restorive_force: Restorive force.
        """
        if self.neutral_bouy:
            restorive_force = (self.cog_to_cob * self.W) * np.vstack(
                (0.0, 0.0, 0.0, cos(x[4]) * sin(x[3]), sin(x[4]), 0.0)
            )
        else:
            restorive_force = np.vstack(
                (
                    (self.W - self.B) * sin(x[4]),
                    -(self.W - self.B) * cos(x[4]) * sin(x[3]),
                    -(self.W - self.B) * cos(x[4]) * cos(x[3]),
                    -(self.cog[1] * self.W - self.cob[1] * self.B)
                    * cos(x[4])
                    * cos(x[3])
                    + (self.cog[2] * self.W - self.cob[2] * self.B)
                    * cos(x[4])
                    * sin(x[3]),
                    (self.cog[2] * self.W - self.cob[2] * self.B) * sin(x[4])
                    + (self.cog[0] * self.W - self.cob[0] * self.B)
                    * cos(x[4])
                    * cos(x[3]),
                    -(self.cog[0] * self.W - self.cob[0] * self.B)
                    * cos(x[4])
                    * sin(x[3])
                    - (self.cog[1] * self.W - self.cob[1] * self.B) * sin(x[4]),
                )
            )
        return restorive_force

    def compute_nonlinear_dynamics(
        self,
        x,
        u,
        f_B=np.zeros((3, 1)),
        f_B_dot=np.zeros((3, 1)),
        f_est=False,
        complete_model=False,
    ):
        """Computes the nonlinear dynamics of the AUV.

        Args:
            x: State vector.
            u: Control input.
            f_B: Flow state. Defaults to np.zeros((3, 1)).
            f_B_dot: Acceleration of the flow. Defaults to np.zeros((3, 1)).
            f_est: Flag to use if flow state estimation is underway. Defaults to False.
            complete_model: Flag to use the complete model. Defaults to False.

        Returns:
            chi_dot: Time derivative of the state vector.
        """
        x = x.reshape(-1, 1)
        u = u.reshape(-1, 1)
        f_B = f_B.reshape(-1, 1)
        f_B_dot = f_B_dot.reshape(-1, 1)
        eta = x[0:6, :]
        nu_r = x[6:12, :]

        # Gets the transformation matrix to convert from Body to NED frame
        tf_mtx = self.compute_transformation_matrix(eta)
        tf_mtx_inv = LA.inv(tf_mtx)

        # nu_c = SX.zeros(6,1)
        nu_c = np.vstack((f_B, np.zeros((3, 1))))

        # Converts ocean current disturbances to Body frame
        # nu_c = mtimes(tf_mtx_inv, nu_c_ned)

        # Computes total vehicle velocity
        nu = nu_r + nu_c

        # Computes the ocean current acceleration in Body frame
        skew_mtx = np.eye(6)
        skew_mtx[0:3, 0:3] = -skew(nu_r[3:6, 0])

        if f_est:
            nu_c_dot = skew_mtx @ nu_c
        else:
            nu_c_dot = np.vstack((f_B_dot, np.zeros((3, 1))))

        # Kinematic Equation
        # Convert the relative velocity from Body to NED and add it with the ocean current velocity in NED to get the total velocity of the vehicle in NED
        eta_dot = tf_mtx @ (nu_r + nu_c)

        # Force computation
        # thruster_force = (self.tam @ u)
        thruster_force = u
        restorive_force = self.compute_restorive_force(eta)
        damping_force = self.compute_damping_force(nu_r)
        coriolis_force_rb = self.compute_C_RB_force(nu)
        coriolis_force_added = self.compute_C_A_force(nu_r)
        coriolis_force_RB_A = self.compute_C_RB_force(nu_r) + self.compute_C_A_force(
            nu_r
        )

        if complete_model:
            nu_r_dot = self.mass_inv @ (
                thruster_force
                - (self.rb_mass @ nu_c_dot)
                - (coriolis_force_rb @ nu)
                - (coriolis_force_added @ nu_r)
                - (damping_force @ nu_r)
                - restorive_force
            )
        else:
            nu_r_dot = self.mass_inv @ (
                thruster_force
                - (coriolis_force_RB_A @ nu_r)
                - (damping_force @ nu_r)
                - restorive_force
            )

        chi_dot = np.vstack((eta_dot, nu_r_dot, nu_c_dot))
        return chi_dot
