import numpy as np
import scipy.io
from weighting import weighting


# ATTENTION
# The snapshots should be created from a matrix N_r x N_theta reshaped with the columns puts one under the OTHER

# Parameter space size and boundaries
Ndx, Ndy = 15, 7
dx_min, dx_max = 2.25, 4
dy_min, dy_max = 0, 0.75

R = 0.8382
n_b = 2

# Parameters dict
params = {'R': R, 'n_b': n_b, 'Ndx': Ndx, 'Ndy': Ndy,
          'dx_min': dx_min, 'dx_max': dx_max, 'dy_min': dy_min, 'dy_max': dy_max}


def load_rmac(force_dir):
    """ Loads snapshots data under 2D matrix form of (snapshot_dim)x(N_snapshots)"""

    # Import data set
    data = scipy.io.loadmat('snapshots_data.mat')

    # Extract matrices
    u_fz = np.array(data['U_Fz'])
    u_fx = np.array(data['U_Fx'])

    params['RadialGrid'] = np.array(data['RadialGrid'])
    params['AzimuthalGrid'] = np.array(data['AzimuthalGrid'])
    params['Dimension'] = np.shape(u_fz)[0]
    params['n_r'] = np.size(params['RadialGrid'])
    params['n_theta'] = np.size(params['AzimuthalGrid'])

    # Select the snapshots matrix and weight it with square root of radial values
    if force_dir == 'L':
        u = u_fz

    elif force_dir == 'D':
        u = u_fx

    elif force_dir == 'LD':
        u_fz_norm = u_fz / np.max(u_fz, 0)
        u_fx_norm = u_fx / np.max(u_fx, 0)

        u = np.vstack((u_fz_norm, u_fx_norm))

    else:
        u = np.zeros([])
        print('Error in loading data')

    # cols = np.array([14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39,
    #                 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67,
    #                 70, 71, 72, 73, 74])

    # cols = np.array([42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67,
    #                  70, 71, 72, 73, 74])
    # u = u[:, cols]

    return u, params


# Parameter space size and boundaries
Ndx, Ndy = 3, 3
dx_min, dx_max = 2.5, 3.5
dy_min, dy_max = 0, 0.5

# Parameters dict
params_cfd = {'n_b': n_b, 'Ndx': Ndx, 'Ndy': Ndy, 'dx_min': dx_min, 'dx_max': dx_max, 'dy_min': dy_min, 'dy_max': dy_max}


def load_cfd(force_dir):

    data = scipy.io.loadmat('cfd_data.mat')

    widths = scipy.io.loadmat('localspan.mat')
    widths = np.array(widths['localspan'])

    u_l_hf = np.array(data['U_L_HF'])
    u_d_hf = np.array(data['U_D_HF'])

    params_cfd['RadialGrid'] = np.array(data['RadialGrid'])
    params_cfd['AzimuthalGrid'] = np.array(data['AzimuthalGrid'])
    params_cfd['Dimension'] = np.shape(u_l_hf)[0]
    params_cfd['R'] = np.max(params_cfd['RadialGrid'])
    params_cfd['n_r'] = np.size(params_cfd['RadialGrid'])
    params_cfd['n_theta'] = np.size(params_cfd['AzimuthalGrid'])

    if force_dir == 'L':
        u = u_l_hf

    elif force_dir == 'D':
        u = u_d_hf

    elif force_dir == 'LD':
        u_l_hf_norm = u_l_hf / np.max(u_l_hf, 0)
        u_d_hf_norm = u_d_hf / np.max(u_d_hf, 0)

        u = np.vstack((u_l_hf_norm, u_d_hf_norm))

    else:
        u = np.zeros([])
        print('Error in loading data')

    return u, params_cfd
