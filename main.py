import numpy as np
import samples_selection as select
from paramspace_computation import get_coefficients, get_error
from paramspace_visualization import error_contour, response_surf, response_contour, force_surfaces
from import_data import load_rmac, load_cfd
from weighting import radial_weighting
from project_set import project_set
from l2error import l2error
from lin_extrapolation import lin_extrapolation
from disk_plot import disk_plot_comparison
import force_computation as fc
import scipy.io


# Number of important snapshots to select
n = 0
force_dir = 'L'

# -------- START ----------------------------------------------------

# Load LF snapshots
U_LF_0, params = load_rmac(force_dir)

# Weight snapshots matrix with radial values
U_LF = radial_weighting(U_LF_0, params)

# New LF set - project onto some basis snaps
# projection_snaps = np.array([14, 16, 18, 42, 44, 46, 70, 72, 74])
projection_snaps = np.array([14, 18, 70, 74])

U_LF_proj = project_set(U_LF, projection_snaps)

# Compute the n most important snapshots
_, selected_snaps = select.qr_pivoted(U_LF_proj, n)

# Basis snapshots indices
basis_snaps = np.concatenate((projection_snaps, selected_snaps)).astype(int)

U_LF_n = U_LF_0[:, basis_snaps]

# Computation of the coefficients C of for the approximation
C = get_coefficients(U_LF_0, U_LF_n)

# Approximation of HF snapshots set
U_LF_hat = np.matmul(U_LF_n, C)

# Plot Response Surface
# response_surf(C, params)

# Computation of the error distribution
_, Error = get_error(U_LF, U_LF[:, basis_snaps], C)
# error_contour(Error, basis_snaps, params, force_dir, save=True)

# Load HF snapshots
U_HF, params_HF = load_cfd(force_dir)
U_HF_copy = U_HF
U_HF = U_HF[:, [0, 2, 6, 8]]

# Lifting Procedure of HF snapshots set
U_HF_hat = np.matmul(U_HF, C)

# Export data from MATLAB

rmac_snaps = np.array([16, 42, 44, 46, 72])
n_r, n_theta = params_HF['n_r'], params_HF['n_theta']

if force_dir == 'L':

    for i in range(5):
        ind = rmac_snaps[i]
        u = U_HF_hat[:, ind].reshape(-1, 1)
        n_row = np.shape(u)[0]

        front_rotor = u[0:n_row // 2]
        rear_rotor = u[n_row // 2:]

        front_rotor = np.reshape(front_rotor, (n_r, n_theta), 'F')
        rear_rotor = np.reshape(rear_rotor, (n_r, n_theta), 'F')

        name = ("lift_MF_{}.mat").format(str(i+1))

        #scipy.io.savemat(name, dict(lift_front_MF=front_rotor, lift_rear_MF=rear_rotor))

if force_dir == 'D':

    for i in range(5):
        ind = rmac_snaps[i]
        u = U_HF_hat[:, ind].reshape(-1, 1)
        n_row = np.shape(u)[0]

        front_rotor = u[0:n_row // 2]
        rear_rotor = u[n_row // 2:]

        front_rotor = np.reshape(front_rotor, (n_r, n_theta), 'F')
        rear_rotor = np.reshape(rear_rotor, (n_r, n_theta), 'F')

        name = ("drag_MF_{}.mat").format(str(i+1))

        #scipy.io.savemat(name, dict(drag_front_MF=front_rotor, drag_rear_MF=rear_rotor))


# -------- COMPARISON QR vs RMAC------------------------------------- #
# u_HF = U_HF_copy[:, 4].reshape(-1, 1)
# u_QR = U_HF_hat[:, 44].reshape(-1, 1)
# u_RMAC = U_LF_0[:, 44].reshape(-1, 1)

test_snaps = np.array([1, 3, 4, 5, 7])
rmac_snaps = np.array([16, 42, 44, 46, 72])

#U_HF_copy = radial_weighting(U_HF_copy, params_HF)
#U_HF_hat = radial_weighting(U_HF_hat, params_HF)

for i in range(5):
     test_ind = test_snaps[i]
     rmac_ind = rmac_snaps[i]
     u_test = U_HF_copy[:, test_ind].reshape(-1, 1)
     u_rmac = U_LF_0[:, rmac_ind].reshape(-1, 1)
     u_qr = U_HF_hat[:, rmac_ind].reshape(-1, 1)

     #l2error(u_test, params_HF, 'cfd', u_rmac, params, 'rmac')
     #l2error(u_test, params_HF, 'cfd', u_qr, params_HF, 'cfd')

#disk_plot_comparison(u_HF, params_HF, 'cfd', u_QR, params_HF, 'cfd', force_dir, 4, save=True)
#disk_plot_comparison(u_HF, params_HF, 'cfd', u_RMAC, params, 'rmac', force_dir, 4, save=True)

# -------- THRUST AND TORQUE ---------------------------------------- #
snaps_train = np.array([0, 2, 6, 8])
snaps_val = np.array([1, 3, 4, 5, 7])

force_surfaces(U_LF_0, params, U_HF_copy, U_HF_hat, params_HF, force_dir, snaps_train, snaps_val, front=True, save=True)

# -------- INTERPOLATION -------------------------------------------- #

# U_HF_test = U_HF[:, 4].reshape(-1, 1)

# U_qr = U_HF_hat[:, 72].reshape(-1, 1)
# U_int = 0.5*(U_HF[:, 2] + U_HF[:, 6]).reshape(-1, 1)

# l2error(U_HF_test, params_HF, 'cfd', U_qr, params_HF, 'cfd')
# l2error(U_HF_test, params_HF, 'cfd', U_int, params_HF, 'cfd')

# -------- EXTRAPOLATION -------------------------------------------- #


# class DiskPlot:
#
#     def __init__(self, u, theta1, theta2):
#         self.u = u
#         self.theta1 = theta1
#         self.theta2 = theta2
#
#
# diskplot1 = DiskPlot(U_HF[:, 0], 3.0, 0.0)
# diskplot2 = DiskPlot(U_HF[:, 1], 3.0, 0.5)
# diskplot3 = DiskPlot(U_HF[:, 2], 3.5, 0.5)

# U_ext = lin_extrapolation(diskplot1, diskplot2, diskplot3, 2.5, 0.25).reshape(-1, 1)
# U_qr = U_HF_hat[:, 72].reshape(-1, 1)

# l2error(U_HF_test, params_HF, 'cfd', U_qr, params_HF, 'cfd')
# l2error(U_HF_test, params_HF, 'cfd', U_ext, params_HF, 'cfd')

# disk_plot_comparison(U_HF_test, U_qr, params_HF, force_dir, location=8, cfd=True, save=False)
# disk_plot_comparison(U_HF_test, U_ext, params_HF, force_dir, location=8, cfd=True, save=False)