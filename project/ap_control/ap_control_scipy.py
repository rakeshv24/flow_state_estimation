import numpy as np
import yaml
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize

from auv.auv_numpy import AUV
from utils.utils import skew


class APC:
    def __init__(self, auv, apc_params):
        self.auv = auv
        self.dt = apc_params["dt"]
        self.log_quiet = apc_params["quiet"]
        self.window_size = apc_params["window_size"]
        self.thrusters = apc_params["thrusters"]
        self.dt_int = str(apc_params["dt_int"])

        self.Q_eta = float(apc_params["noises"]["Q_eta"]) * np.eye(6)
        self.Q_nu_r = float(apc_params["noises"]["Q_nu_r"]) * np.eye(6)
        self.Q_f = float(apc_params["noises"]["Q_f"]) * np.eye(3)
        self.Q_auv_state = block_diag(self.Q_eta, self.Q_nu_r)

        self.R_att = float(apc_params["noises"]["R_att"]) * np.eye(3)
        self.R_linvel = float(apc_params["noises"]["R_linvel"]) * np.eye(3)
        self.R_angvel = float(apc_params["noises"]["R_angvel"]) * np.eye(3)
        self.R_linacc = float(apc_params["noises"]["R_linacc"]) * np.eye(3)
        self.R_xy = float(apc_params["noises"]["R_xy"]) * np.eye(2)
        self.R_z = float(apc_params["noises"]["R_z"])
        self.R_dr = float(apc_params["noises"]["R_dr"])
        self.R_auv_meas = block_diag(self.R_z, self.R_att, self.R_linvel, self.R_angvel, self.R_linacc)
        self.R_f_meas = block_diag(self.R_linacc, self.R_dr)

        self.umin = float(apc_params["bounds"]["umin"])
        self.umax = float(apc_params["bounds"]["umax"])

    @classmethod
    def load_params(cls, auv_filename, apc_filename):
        auv = AUV.load_params(auv_filename)

        f = open(apc_filename, "r")
        apc_params = yaml.load(f.read(), Loader=yaml.SafeLoader)

        return cls(auv, apc_params)

    def cost_fn(self, u, chi, f_B, ctrl_obj):
        chi_dot = self.auv.compute_nonlinear_dynamics(
            chi, u, f_B, f_est=True, complete_model=False
        )
        chi_next = chi + chi_dot[0:12, :] * self.dt

        S_kk_next = skew(chi_next[9:12, :])
        tf_B2I_est = self.auv.compute_transformation_matrix(chi_next[0:6, :])
        R_B2I_est = tf_B2I_est[0:3, 0:3]
        H_kk_next = np.vstack(
            (-S_kk_next, np.array([[0, 0, 1]], dtype=np.float64) @ R_B2I_est)
        )

        gram = H_kk_next.T @ np.linalg.inv(self.R_f_meas) @ H_kk_next + ctrl_obj["G"]
        cov = np.cov(gram)
        min_eig = -np.min(np.linalg.eigvals(gram))
        max_var = np.max(np.diag(cov))
        mean_var = np.mean(np.diag(cov))

        # cost = max_var
        # cost = mean_var
        cost = min_eig

        return cost

    def ap_control_cg(self, idx, chi, f_B, ctrl_obj):
        eta = chi[0:6, :]
        nu_r = chi[6:12, :]
        nu_r2 = nu_r[3:6, :]
        f_B = f_B
        tf_B2I = self.auv.compute_transformation_matrix(eta)
        R_B2I = tf_B2I[0:3, 0:3]

        ctrl_obj["S_kk"][:, :, idx] = skew(nu_r2)
        ctrl_obj["F_kk"][:, :, idx] = np.eye(3) - ctrl_obj["S_kk"][:, :, idx] * self.dt
        ctrl_obj["H_kk"][:, :, idx] = np.vstack(
            (-ctrl_obj["S_kk"][:, :, idx], np.array([[0, 0, 1]]) @ R_B2I)
        )

        if idx < self.window_size - 1:
            N = idx
        else:
            N = self.window_size - 1

        uco_idx = [i for i in range(idx - N + 1, idx + 1)]
        uco_idx.reverse()

        Phi = np.eye(3)
        for kk in uco_idx:
            Phi = Phi @ ctrl_obj["F_kk"][:, :, kk]
        Phi_inv = np.linalg.inv(Phi)

        if idx == 0:
            ctrl_obj["G"] = Phi_inv.T @ ctrl_obj["G0"] @ Phi_inv
        else:
            ctrl_obj["G"] = Phi_inv.T @ ctrl_obj["G_kk"][:, :, idx - 1] @ Phi_inv

        lb = self.umin
        ub = self.umax
        bnds = tuple([(lb, ub) for _ in range(6)])

        sol = minimize(
            self.cost_fn,
            np.random.uniform(lb, ub, size=(self.thrusters, 1)),
            args=(chi, f_B, ctrl_obj),
            bounds=bnds,
            method="SLSQP",
        )

        u_next = sol.x

        chi_dot = self.auv.compute_nonlinear_dynamics(
            chi, u_next, f_B, f_est=True, complete_model=False
        )
        chi_next = chi + chi_dot[0:12, :] * self.dt

        S_kk_next = skew(chi_next[9:12, :])
        tf_B2I_est = self.auv.compute_transformation_matrix(chi_next[0:6, :])
        R_B2I_est = tf_B2I_est[0:3, 0:3]
        H_kk_next = np.vstack(
            (-S_kk_next, np.array([[0, 0, 1]], dtype=np.float64) @ R_B2I_est)
        )

        ctrl_obj["G_kk"][:, :, idx] = (
            H_kk_next.T @ np.linalg.inv(self.R_f_meas) @ H_kk_next + ctrl_obj["G"]
        )

        return u_next, ctrl_obj
