import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ap_control.ap_control_scipy import APC
from auv.auv_numpy import AUV
from env.env_numpy import EnvironmentManager
from kf.kalman import KalmanFilter
from pf.particle_filter import ParticleFilter
from rbpf.rbpf import RBPF
from utils.utils import skew

from multiprocessing import Pool, cpu_count
from tqdm import tqdm



class APControl:
    def __init__(self):
        self.auv_yaml = os.path.join("config", "auv_bluerov2_heavy.yaml")
        self.apc_yaml = os.path.join("config", "apc.yaml")
        self.test_params_yaml = os.path.join("config", "test_params.yaml")

        self.apc = APC.load_params(self.auv_yaml, self.apc_yaml)
        self.auv = AUV.load_params(self.auv_yaml)
        self.test_params = self.load_params(self.test_params_yaml)
        self.num_trials = self.test_params["num_trials"]
        self.envs = self.test_params["environments"]
        self.rbpf_flag = self.test_params["rbpf"]

    def load_params(self, filename):
        f = open(filename, "r")
        test_params = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return test_params

    def initialize_params(self):
        self.x_ini = np.array([self.test_params["x0"]]).T
        self.env_man = EnvironmentManager(self.auv)
        self.env = self.env_man.init_env(self.env_name, self.env_type, self.env_val)
        self.env = self.env_man.handle_env(self.env, 0.0)
        self.env_flow_ini = self.env_man.get_env(
            self.x_ini[0:6, :], self.x_ini[6:12, :], self.env, 0
        )
        self.f_B_ini = np.vstack((self.env_flow_ini["f_B"], np.zeros((3, 1))))
        self.f1_ini = self.f_B_ini[0:3, :]
        self.x_ini[6:9, :] = -self.f1_ini

        self.dt = self.apc.dt
        self.t_f = self.test_params["t_len"]
        self.t_span = np.arange(0.0, self.t_f, self.dt)

        n_f_state = self.f1_ini.shape[0]
        n_f_meas = len(self.apc.R_f_meas)
        n_auv_meas = len(self.apc.R_auv_meas)

        n_particles = self.test_params["pf"]["particles"]
        pf_init_cov = float(self.test_params["pf"]["cov"])

        self.kf = KalmanFilter(
            self.f1_ini,
            self.apc.Q_f,
            self.apc.R_f_meas,
            self.f_state_trans_model,
            self.f_meas_model,
        )

        self.pf = ParticleFilter(
            self.apc.Q_auv_state,
            self.apc.R_auv_meas,
            self.auv_state_model,
            self.auv_meas_model,
        )
        self.pf.initialize(
            n_particles, self.x_ini, pf_init_cov * np.eye(self.x_ini.shape[0])
        )

        self.rbpf = RBPF(
            self.apc.Q_auv_state,
            self.apc.R_auv_meas,
            self.auv_state_model,
            self.auv_meas_model,
            self.apc.Q_f,
            self.apc.R_f_meas,
            self.f_state_trans_model,
            self.f_meas_model,
            self.f1_ini,
        )
        self.rbpf.initialize(
            n_particles, self.x_ini, pf_init_cov * np.eye(self.x_ini.shape[0])
        )

        self.ctrl_obj = {}
        self.ctrl_obj["F_kk"] = np.zeros((3, 3, len(self.t_span)))
        self.ctrl_obj["S_kk"] = np.zeros((3, 3, len(self.t_span)))
        self.ctrl_obj["H_kk"] = np.zeros((n_f_meas, n_f_state, len(self.t_span)))
        self.ctrl_obj["G_kk"] = np.zeros((3, 3, len(self.t_span)))
        self.ctrl_obj["G0"] = np.linalg.inv(np.eye(n_f_state))

        self.nav_data = {"state": {}, "control": {}, "analysis": {}}
        self.nav_data["state"]["t"] = self.t_span
        self.nav_data["state"]["eta"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["nu"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["nu_r"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["f_B"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["f_I"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["nu_I"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["control"]["u"] = np.zeros((self.apc.thrusters, len(self.t_span)))
        self.nav_data["analysis"]["eta_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["nu_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["nu_r_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["f_B_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["f_I_dot"] = np.zeros((3, len(self.t_span)))
        self.nav_data["analysis"]["nu_I_dot"] = np.zeros((3, len(self.t_span)))

        self.nav_est_data = {"state": {}, "meas": {}, "analysis": {}}
        self.nav_est_data["state"]["t"] = self.t_span[0:-1]
        self.nav_est_data["state"]["eta"] = np.zeros((6, len(self.t_span)))
        self.nav_est_data["state"]["nu_r"] = np.zeros((6, len(self.t_span)))
        self.nav_est_data["state"]["f_B"] = np.zeros((3, len(self.t_span)))
        self.nav_est_data["state"]["f_I"] = np.zeros((3, len(self.t_span)))
        self.nav_est_data["state"]["t_pred"] = self.t_span
        self.nav_est_data["state"]["eta_pred"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_est_data["state"]["nu_r_pred"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_est_data["state"]["f_B_pred"] = np.zeros((3, len(self.t_span) + 1))
        self.nav_est_data["state"]["f_I_pred"] = np.zeros((3, len(self.t_span) + 1))
        self.nav_est_data["meas"]["zeta_f"] = np.zeros((n_f_meas, len(self.t_span)))
        self.nav_est_data["meas"]["zeta_auv"] = np.zeros((n_auv_meas, len(self.t_span)))
        self.nav_est_data["analysis"]["eta_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_est_data["analysis"]["nu_r_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_est_data["analysis"]["f_B_dot"] = np.zeros((3, len(self.t_span)))
        self.nav_est_data["analysis"]["f_I_dot"] = np.zeros((3, len(self.t_span)))

        self.env_data = {"state": {}, "analysis": {}}
        self.env_data["state"]["t"] = self.t_span[0:-1]
        self.env_data["state"]["f_I"] = np.zeros((3, len(self.t_span)))
        self.env_data["state"]["f_B"] = np.zeros((3, len(self.t_span)))
        self.env_data["analysis"]["f_I_dot"] = np.zeros((3, len(self.t_span)))
        self.env_data["analysis"]["f_B_dot"] = np.zeros((3, len(self.t_span)))

        self.env_est_data = {"state": {}, "analysis": {}}
        self.env_est_data["state"]["t"] = self.t_span[0:-1]
        self.env_est_data["state"]["f_I"] = np.zeros((3, len(self.t_span)))
        self.env_est_data["state"]["f_B"] = np.zeros((3, len(self.t_span)))
        self.env_est_data["analysis"]["f_I_dot"] = np.zeros((3, len(self.t_span)))
        self.env_est_data["analysis"]["f_B_dot"] = np.zeros((3, len(self.t_span)))

        self.est_data = {"analysis": {}}
        self.est_data["P_f_pred"] = np.zeros((3, 3, len(self.t_span) + 1))
        self.est_data["P_f_est"] = np.zeros((3, 3, len(self.t_span) + 1))
        self.est_data["P_auv_pred"] = np.zeros((12, 12, len(self.t_span) + 1))
        self.est_data["P_auv_est"] = np.zeros((12, 12, len(self.t_span) + 1))
        self.est_data["analysis"]["est_eta_diff"] = np.zeros((6, len(self.t_span)))
        self.est_data["analysis"]["est_eta_sqdiff"] = np.zeros((1, len(self.t_span)))
        self.est_data["analysis"]["est_nu_r_diff"] = np.zeros((6, len(self.t_span)))
        self.est_data["analysis"]["est_nu_r_sqdiff"] = np.zeros((1, len(self.t_span)))
        self.est_data["analysis"]["est_f_B_diff"] = np.zeros((3, len(self.t_span)))
        self.est_data["analysis"]["est_f_B_dot_diff"] = np.zeros((3, len(self.t_span)))
        self.est_data["analysis"]["est_f_B_sqdiff"] = np.zeros((1, len(self.t_span)))
        self.est_data["analysis"]["est_f_B_dot_sqdiff"] = np.zeros(
            (1, len(self.t_span))
        )

    def f_state_trans_model(self, x):
        nu_r2 = x[9:12, :]
        S = skew(nu_r2)
        A = np.eye(3) - S * self.dt
        A = np.array(A)
        return A

    def f_meas_model(self, x):
        eta = x[0:6, :]
        nu_r2 = x[9:12, :]
        S = skew(nu_r2)
        tf_B2I = self.auv.compute_transformation_matrix(eta)
        R_B2I = tf_B2I[0:3, 0:3]
        H = np.vstack((-S, np.array([[0, 0, 1]], dtype=np.float64) @ R_B2I))
        H = np.array(H)
        return H

    def flow_model_est(self, chi, chi_f):
        nu_r = chi[6:12, :]
        f1_B = chi_f
        S = skew(nu_r[3:6, :])
        f1_B_dot = -(S @ f1_B)
        f1_B_dot = np.array(f1_B_dot)
        return f1_B_dot

    def auv_state_model(self, chi, u, f_B):
        chi = chi.reshape(-1, 1)
        chi_dot = self.auv.compute_nonlinear_dynamics(
            chi, u, f_B, f_est=True, complete_model=True
        )
        chi_next = chi + chi_dot[0:12, :] * self.dt
        chi_next[3:6, :] = self.wrap_pi2negpi(chi_next[3:6, :])
        return chi_next[0:12, :]

    def auv_meas_model(self, chi, u, f_B):
        chi = chi.reshape(-1, 1)
        eta = chi[0:6, :]
        nu_r = chi[6:12, :]
        chi_dot = self.auv.compute_nonlinear_dynamics(
            chi, u, f_B, f_est=True, complete_model=True
        )
        f1_B_dot = chi_dot[12:15, 0]
        nu_r1_dot = chi_dot[6:9, 0]
        acc = f1_B_dot + nu_r1_dot

        zeta = np.zeros((13, 1))
        zeta[0, 0] = eta[2, 0]
        zeta[1:4, 0] = eta[3:6, 0]
        zeta[4:7, 0] = nu_r[0:3, 0]
        zeta[7:10, 0] = nu_r[3:6, 0]
        zeta[10:13, 0] = acc
        return zeta

    def wrap_pi2negpi(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def run_apc(self, e):
        for _, e in self.envs.items():
            self.env_val = np.array(e["current"]["value"], dtype=np.float64)
            self.env_name = str(e["current"]["name"])
            self.env_type = str(e["current"]["type"])

            for t_id in range(self.num_trials):
                self.initialize_params()
                x_true = self.x_ini
                f_B_true = self.f_B_ini
                chi_true = np.vstack((x_true, f_B_true))
                chi_true = np.array(chi_true)

                self.nav_data["state"]["eta"][:, 0] = chi_true[0:6, :].flatten()
                self.nav_data["state"]["nu_r"][:, 0] = chi_true[6:12, :].flatten()
                self.nav_data["state"]["f_B"][:, 0] = chi_true[12:18, :].flatten()
                self.nav_data["state"]["nu"][:, 0] = (
                    chi_true[6:12, :] + chi_true[12:18, :]
                ).flatten()

                if self.rbpf_flag:
                    self.nav_est_data["state"]["eta_pred"][:, 0] = self.rbpf.X[0:6].flatten()
                    self.nav_est_data["state"]["nu_r_pred"][:, 0] = self.rbpf.X[6:12].flatten()
                    self.nav_est_data["state"]["f_B_pred"][:, 0] = self.rbpf.f_B.flatten()
                    self.est_data["P_auv_pred"][:, :, 0] = self.rbpf.P
                    self.est_data["P_auv_est"][:, :, 0] = self.rbpf.P
                else:
                    self.nav_est_data["state"]["eta_pred"][:, 0] = self.pf.X[0:6].flatten()
                    self.nav_est_data["state"]["nu_r_pred"][:, 0] = self.pf.X[6:12].flatten()
                    self.nav_est_data["state"]["f_B_pred"][:, 0] = self.kf.x.flatten()
                    self.est_data["P_f_pred"][:, :, 0] = self.kf.P
                    self.est_data["P_f_est"][:, :, 0] = self.kf.P
                    self.est_data["P_auv_pred"][:, :, 0] = self.pf.P
                    self.est_data["P_auv_est"][:, :, 0] = self.pf.P

                for i in range(len(self.t_span)):
                    eta_pred = self.nav_est_data["state"]["eta_pred"][:, i].reshape(-1, 1)
                    nu_r_pred = self.nav_est_data["state"]["nu_r_pred"][:, i].reshape(-1, 1)
                    f_B_pred = self.nav_est_data["state"]["f_B_pred"][:, i].reshape(-1, 1)
                    chi_auv_pred = np.vstack((eta_pred, nu_r_pred))
                    P_auv_pred = self.est_data["P_auv_pred"][:, :, i]

                    u, ctrl_obj = self.apc.ap_control_cg(
                        i, chi_auv_pred, f_B_pred, self.ctrl_obj
                    )
                    # u = np.zeros((6,))
                    self.ctrl_obj = ctrl_obj

                    self.env = self.env_man.handle_env(self.env, self.t_span[i])
                    self.env_flow = self.env_man.get_env(
                        chi_true[0:6, :], chi_true[6:12, :], self.env, self.t_span[i]
                    )

                    if self.apc.dt_int == "euler":
                        chi_dot = self.auv.compute_nonlinear_dynamics(
                            chi_true,
                            u,
                            self.env_flow["f_B"],
                            self.env_flow["f_B_dot"],
                            f_est=True,
                            complete_model=True,
                        )
                    elif self.apc.dt_int == "rk4":
                        pass

                    chi_dot = np.array(chi_dot)

                    chi_true += chi_dot * self.dt
                    chi_true[3:6, :] = self.wrap_pi2negpi(chi_true[3:6, :])

                    self.nav_data["state"]["eta"][:, i + 1] = chi_true[0:6, :].flatten()
                    self.nav_data["state"]["nu_r"][:, i + 1] = chi_true[6:12, :].flatten()
                    self.nav_data["state"]["f_B"][:, i + 1] = chi_true[12:18, :].flatten()
                    self.nav_data["state"]["nu"][:, i + 1] = (
                        chi_true[6:12, :] + chi_true[12:18, :]
                    ).flatten()
                    self.nav_data["control"]["u"][:, i] = u
                    self.nav_data["analysis"]["eta_dot"][:, i] = chi_dot[0:6, :].flatten()
                    self.nav_data["analysis"]["nu_r_dot"][:, i] = chi_dot[6:12, :].flatten()
                    self.nav_data["analysis"]["f_B_dot"][:, i] = chi_dot[12:18, :].flatten()
                    self.nav_data["analysis"]["nu_dot"][:, i] = (
                        chi_dot[6:12, :] + chi_dot[12:18, :]
                    ).flatten()

                    self.env_data["state"]["f_B"][:, i] = (self.env_flow["f_B"]).flatten()
                    self.env_data["analysis"]["f_B_dot"][:, i] = (
                        self.env_flow["f_B_dot"]
                    ).flatten()

                    eta_true = self.nav_data["state"]["eta"][:, i].reshape(-1, 1)
                    nu_r_true = self.nav_data["state"]["nu_r"][:, i].reshape(-1, 1)
                    x_true = np.vstack((eta_true, nu_r_true))
                    nu_true = self.nav_data["state"]["nu"][:, i]
                    f_B_true = self.env_data["state"]["f_B"][:, i]
                    eta_dot_true = self.nav_data["analysis"]["eta_dot"][:, i].reshape(-1, 1)
                    nu_r_dot_true = self.nav_data["analysis"]["nu_r_dot"][:, i].reshape(-1, 1)
                    nu_dot_true = self.nav_data["analysis"]["nu_dot"][:, i]
                    f_B_dot_true = self.env_data["analysis"]["f_B_dot"][:, i]
                    chi_auv_true = np.vstack((eta_true, nu_r_true))
                    np.vstack((eta_dot_true, nu_r_dot_true))

                    chi_pred_dot = self.auv.compute_nonlinear_dynamics(
                        chi_auv_pred, u, f_B_pred, f_est=True, complete_model=True
                    )
                    chi_auv_pred_dot = chi_pred_dot[0:12, :]

                    nu_r_pred_dot = chi_auv_pred_dot[6:12, :]
                    nu_r_pred = chi_auv_pred[6:12]

                    tf_B2I_est = self.auv.compute_transformation_matrix(chi_auv_pred[0:6, :])
                    R_B2I_est = tf_B2I_est[0:3, 0:3]

                    eta_z_meas = eta_true[2] + np.sqrt(self.apc.R_z) * np.random.randn()
                    eta_z_dot_meas = (
                        eta_dot_true[2] + np.sqrt(self.apc.R_dr) * np.random.randn()
                    )
                    ang_vel_meas = (
                        nu_true[3:6] + np.diag(np.sqrt(self.apc.R_angvel)) * np.random.randn()
                    )
                    eta_att_meas = (
                        eta_true[3:6, 0] + np.diag(np.sqrt(self.apc.R_att)) * np.random.randn()
                    )
                    lin_acc_meas = (
                        nu_dot_true[0:3]
                        + np.diag(np.sqrt(self.apc.R_linacc)) * np.random.randn()
                    )
                    xy_meas = (
                        eta_true[0:2, 0]
                        + np.diag(np.sqrt(self.apc.R_xy)) * np.random.randn()
                    )
                    lin_vel_meas = (
                        nu_r_true[0:3, 0]
                        + np.diag(np.sqrt(self.apc.R_linvel)) * np.random.randn()
                    )

                    xy_meas = xy_meas.reshape(2, 1)
                    lin_vel_meas = lin_vel_meas.reshape(3, 1)
                    ang_vel_meas = ang_vel_meas.reshape(3, 1)
                    eta_att_meas = eta_att_meas.reshape(3, 1)
                    lin_acc_meas = lin_acc_meas.reshape(3, 1)

                    zeta_auv = np.vstack((eta_z_meas, eta_att_meas, lin_vel_meas, ang_vel_meas, lin_acc_meas))

                    zeta_f = np.vstack(
                        (
                            lin_acc_meas - nu_r_pred_dot[0:3, :],
                            eta_z_dot_meas
                            - np.array([[0, 0, 1]]) @ (R_B2I_est @ nu_r_pred[0:3, :]),
                        )
                    )

                    if self.rbpf_flag:
                        chi_auv_est, chi_f_est, P_auv_est = self.rbpf.update(zeta_auv, zeta_f, u)
                        chi_auv_est = chi_auv_est.reshape(-1, 1)
                        chi_auv_est[3:6, :] = self.wrap_pi2negpi(chi_auv_est[3:6, :])

                        chi_est_dot = self.auv.compute_nonlinear_dynamics(
                            chi_auv_est, u, chi_f_est, f_est=True, complete_model=True
                        )
                        chi_auv_est_dot = chi_est_dot[0:12, :]
                        chi_f_est_dot = self.flow_model_est(chi_auv_est, chi_f_est)

                        chi_auv_pred, chi_f_pred, P_auv_pred = self.rbpf.predict(u)
                        chi_auv_pred = chi_auv_pred.reshape(-1, 1)
                        chi_auv_pred[3:6, :] = self.wrap_pi2negpi(chi_auv_pred[3:6, :])

                    else:
                        chi_f_est, P_f_est = self.kf.update(chi_auv_pred, zeta_f)
                        chi_auv_est, P_auv_est = self.pf.update(zeta_auv, u, chi_f_est)
                        chi_auv_est = chi_auv_est.reshape(-1, 1)
                        chi_auv_est[3:6, :] = self.wrap_pi2negpi(chi_auv_est[3:6, :])

                        chi_est_dot = self.auv.compute_nonlinear_dynamics(
                            chi_auv_est, u, chi_f_est, f_est=True, complete_model=True
                        )
                        chi_auv_est_dot = chi_est_dot[0:12, :]
                        chi_f_est_dot = self.flow_model_est(chi_auv_est, chi_f_est)

                        chi_auv_pred, P_auv_pred = self.pf.predict(u, chi_f_est)
                        chi_auv_pred[3:6, :] = self.wrap_pi2negpi(chi_auv_pred[3:6, :])

                        chi_f_pred, P_f_pred = self.kf.predict(chi_auv_est)

                        self.est_data["P_f_pred"][:, :, i + 1] = P_f_pred
                        self.est_data["P_f_est"][:, :, i + 1] = P_f_est

                    self.nav_est_data["meas"]["zeta_auv"][:, i] = zeta_auv.flatten()
                    self.nav_est_data["meas"]["zeta_f"][:, i] = zeta_f.flatten()
                    self.nav_est_data["state"]["eta"][:, i] = chi_auv_est[0:6].flatten()
                    self.nav_est_data["state"]["nu_r"][:, i] = chi_auv_est[6:12].flatten()
                    self.nav_est_data["state"]["f_B"][:, i] = chi_f_est.flatten()
                    self.nav_est_data["analysis"]["eta_dot"][:, i] = chi_auv_est_dot[
                        0:6
                    ].flatten()
                    self.nav_est_data["analysis"]["nu_r_dot"][:, i] = chi_auv_est_dot[
                        6:12
                    ].flatten()
                    self.nav_est_data["analysis"]["f_B_dot"][:, i] = chi_f_est_dot.flatten()
                    self.nav_est_data["state"]["eta_pred"][:, i + 1] = chi_auv_pred[
                        0:6
                    ].flatten()
                    self.nav_est_data["state"]["nu_r_pred"][:, i + 1] = chi_auv_pred[
                        6:12
                    ].flatten()
                    self.nav_est_data["state"]["f_B_pred"][:, i + 1] = chi_f_pred.flatten()

                    self.env_est_data["analysis"]["f_B_dot"][:, i] = chi_f_est_dot.flatten()

                    self.est_data["P_auv_pred"][:, :, i + 1] = P_auv_pred
                    self.est_data["P_auv_est"][:, :, i + 1] = P_auv_est
                    self.est_data["analysis"]["est_eta_diff"][:, i] = (
                        chi_auv_true[0:6, :] - chi_auv_est[0:6, :]
                    ).flatten()
                    self.est_data["analysis"]["est_eta_sqdiff"][:, i] = np.linalg.norm(
                        chi_auv_true[0:6, :] - chi_auv_est[0:6, :]
                    )
                    self.est_data["analysis"]["est_nu_r_diff"][:, i] = (
                        chi_auv_true[6:12, :] - chi_auv_est[6:12, :]
                    ).flatten()
                    self.est_data["analysis"]["est_nu_r_sqdiff"][:, i] = np.linalg.norm(
                        chi_auv_true[6:12, :] - chi_auv_est[6:12, :]
                    )
                    self.est_data["analysis"]["est_f_B_diff"][:, i] = (
                        f_B_true - chi_f_est.flatten()
                    )
                    self.est_data["analysis"]["est_f_B_sqdiff"][:, i] = np.linalg.norm(
                        f_B_true - chi_f_est.flatten()
                    )
                    self.est_data["analysis"]["est_f_B_dot_diff"][:, i] = (
                        f_B_dot_true - chi_f_est_dot.flatten()
                    )
                    self.est_data["analysis"]["est_f_B_dot_sqdiff"][:, i] = np.linalg.norm(
                        f_B_dot_true - chi_f_est_dot.flatten()
                    )

                    print(f"T = {round(self.t_span[i],3)}s, Time Index = {i}")
                    print("----------------------------------------------")
                    print(f"AP Control Input: {u.T}")
                    print("----------------------------------------------")
                    print(f"True Vehicle Pose: {np.round(chi_auv_true[0:6], 3).T}")
                    print(f"Est Vehicle Pose: {np.round(chi_auv_est[0:6], 3).T}")
                    print(
                        "Diff: ", np.round(self.est_data["analysis"]["est_eta_diff"][:, i], 6)
                    )
                    print(
                        "Sq Diff: ",
                        np.round(self.est_data["analysis"]["est_eta_sqdiff"][:, i], 6),
                    )
                    print("----------------------------------------------")
                    print(f"True Vehicle Velocity: {np.round(chi_auv_true[6:12], 3).T}")
                    print(f"Est Vehicle Velocity: {np.round(chi_auv_est[6:12], 3).T}")
                    print(
                        "Diff: ", np.round(self.est_data["analysis"]["est_nu_r_diff"][:, i], 6)
                    )
                    print(
                        "Sq Diff: ",
                        np.round(self.est_data["analysis"]["est_nu_r_sqdiff"][:, i], 6),
                    )
                    print("----------------------------------------------")
                    print(f"True Flow Velocity (f1): {f_B_true}")
                    print(f"Estimated Flow Velocity (f1'): {chi_f_est.T}")
                    print("Diff: ", self.est_data["analysis"]["est_f_B_diff"][:, i])
                    print("Sq Diff: ", self.est_data["analysis"]["est_f_B_sqdiff"][:, i])
                    print("----------------------------------------------")
                    print(f"True Flow Acc (f1_dot): {f_B_dot_true}")
                    print(f"Estimated Flow Acc (f1'_dot): {chi_f_est_dot.T}")
                    print("Diff: ", self.est_data["analysis"]["est_f_B_dot_diff"][:, i])
                    print("Sq Diff: ", self.est_data["analysis"]["est_f_B_dot_sqdiff"][:, i])
                    print("----------------------------------------------")
                    print("")

                self.plot_graphs()

    def plot_graphs(self):
        plt.figure(dpi=100)
        plt.plot(
            self.nav_data["state"]["t"],
            self.est_data["analysis"]["est_f_B_diff"][0, :],
            "c--",
            label="Error - u",
        )
        plt.plot(
            self.nav_data["state"]["t"],
            self.est_data["analysis"]["est_f_B_diff"][1, :],
            "m--",
            label="Error - v",
        )
        plt.plot(
            self.nav_data["state"]["t"],
            self.est_data["analysis"]["est_f_B_diff"][2, :],
            "k--",
            label="Error - w",
        )
        plt.xlabel("Timestep [s]")
        plt.ylabel("Flow Estimation Error [m/s]")
        plt.legend()

        plt.figure(dpi=100)
        plt.plot(
            self.nav_data["state"]["t"],
            self.est_data["analysis"]["est_f_B_dot_diff"][0, :],
            "c--",
            label="Error - u_dot",
        )
        plt.plot(
            self.nav_data["state"]["t"],
            self.est_data["analysis"]["est_f_B_dot_diff"][1, :],
            "m--",
            label="Error - v_dot",
        )
        plt.plot(
            self.nav_data["state"]["t"],
            self.est_data["analysis"]["est_f_B_dot_diff"][2, :],
            "k--",
            label="Error - w_dot",
        )
        plt.xlabel("Timestep [s]")
        plt.ylabel("Flow Acc Estimation Error [m/s^2]")
        plt.legend()

        plt.show()

    def main(self):
        test_cases = []
        for _, e in self.envs.items():
            env_val = np.array(e["current"]["value"], dtype=np.float64)
            env_name = str(e["current"]["name"])
            env_type = str(e["current"]["type"])
            case = [env_val, env_name, env_type]
            test_cases.append(case)

        with Pool(cpu_count()-6) as p:
            successes = list(tqdm(p.imap_unordered(self.run_apc, test_cases), total=len(test_cases)))

        print("Successful:")
        print(successes)


if __name__ == "__main__":
    apc = APControl()
    apc.main()
