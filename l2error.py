from scipy import interpolate
import numpy as np
from weighting import weighting
from decimal import Decimal


def l2error(u1, params1, type1, u2, params2, type2):

    if type1 == 'cfd':
        radial_grid = params1['RadialGrid'].reshape(-1, 1)
        dr = np.diff(radial_grid, axis=0)
        dr = np.vstack((radial_grid[1], dr))
        u1 = weighting(u1, 1 / dr, mode='repmat')

    if type2 == 'cfd':
        radial_grid = params2['RadialGrid'].reshape(-1, 1)
        dr = np.diff(radial_grid, axis=0)
        dr = np.vstack((radial_grid[1], dr))
        u2 = weighting(u2, 1 / dr, mode='repmat')

    u1_front = u1[:u1.size // 2, :]
    u1_rear = u1[u1.size // 2:, :]

    u2_front = u2[:u2.size // 2, :]
    u2_rear = u2[u2.size // 2:, :]

    # If the disk plots are differently sized, interpolation is used
    if type1 != type2:

        # The rmac snapshot radial grid must be scaled w.r.t. the radius R
        if type1 == 'cfd':
            ax1_1 = params1['RadialGrid']
            ax2_1 = params1['AzimuthalGrid']

            ax1_2 = params2['RadialGrid'] * params2['R']
            ax2_2 = params2['AzimuthalGrid']

        else:
            ax1_1 = params1['RadialGrid'] * params1['R']
            ax2_1 = params1['AzimuthalGrid']

            ax1_2 = params2['RadialGrid']
            ax2_2 = params2['AzimuthalGrid']

        # Interpolation to get same dimension disk plots
        u1_front, u2_front, _, _ = vector_interpol(u1_front, ax1_1, ax2_1, u2_front, ax1_2, ax2_2)
        u1_rear, u2_rear, _, _ = vector_interpol(u1_rear, ax1_1, ax2_1, u2_rear, ax1_2, ax2_2)

    # L2 Relative Errors
    err_rel_front = np.linalg.norm(u2_front - u1_front) / np.linalg.norm(u2_front)
    err_rel_rear = np.linalg.norm(u2_rear - u1_rear) / np.linalg.norm(u2_rear)

    err_rel_front = Decimal(err_rel_front)
    err_rel_rear = Decimal(err_rel_rear)

    err_rel_front = round(100*err_rel_front, 2)
    err_rel_rear = round(100*err_rel_rear, 2)

    print('')
    print('L2 Relative Error for Front Rotor in %:')
    print(err_rel_front)
    print('L2 Relative Error for Rear Rotor in %:')
    print(err_rel_rear)
    print('')

    return err_rel_front, err_rel_rear


def matrix_interpol(u1, ax1_1, ax2_1, u2, ax1_2, ax2_2):

    u1 = u1.reshape((ax1_1.size, ax2_1.size), order='F')
    u2 = u2.reshape((ax1_2.size, ax2_2.size), order='F')

    f1 = interpolate.interp2d(ax1_1, ax2_1, u1.T)
    f2 = interpolate.interp2d(ax1_2, ax2_2, u2.T)

    if ax1_1.size > ax1_2.size:
        ax1 = ax1_1.reshape(ax1_1.size)
    else:
        ax1 = ax1_2.reshape(ax1_2.size)

    if ax2_1.size > ax2_2.size:
        ax2 = ax2_1.reshape(ax2_1.size)
    else:
        ax2 = ax2_2.reshape(ax2_2.size)

    u1 = f1(ax1, ax2)
    u2 = f2(ax1, ax2)

    u1 = u1.T
    u2 = u2.T

    return u1, u2, ax1, ax2


def vector_interpol(u1, ax1_1, ax2_1, u2, ax1_2, ax2_2):

    u1 = u1.reshape((ax1_1.size, ax2_1.size), order='F')
    u2 = u2.reshape((ax1_2.size, ax2_2.size), order='F')

    f1 = interpolate.interp2d(ax1_1, ax2_1, u1.T)
    f2 = interpolate.interp2d(ax1_2, ax2_2, u2.T)

    if ax1_1.size > ax1_2.size:
        ax1 = ax1_1.reshape(ax1_1.size)
    else:
        ax1 = ax1_2.reshape(ax1_2.size)

    if ax2_1.size > ax2_2.size:
        ax2 = ax2_1.reshape(ax2_1.size)
    else:
        ax2 = ax2_2.reshape(ax2_2.size)

    u1 = f1(ax1, ax2)
    u2 = f2(ax1, ax2)

    u1 = u1.T
    u2 = u2.T

    new_shape = (ax1.size*ax2.size, 1)

    u1 = u1.reshape(new_shape, order='F')
    u2 = u2.reshape(new_shape, order='F')

    return u1, u2, ax1, ax2


