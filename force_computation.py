import numpy as np
from weighting import weighting, radial_weighting


def compute_thrust_rmac(u, params):

    dim_u = np.shape(u)[0]

    front_rotors = u[:dim_u // 2, :]
    rear_rotors = u[dim_u // 2:, :]

    R = params['R']

    radial_grid = R*params['RadialGrid'].reshape(-1, 1)
    azimuthal_grid = params['AzimuthalGrid'].reshape(-1, 1)
    azimuthal_grid = (np.pi/180)*azimuthal_grid

    dr = np.diff(radial_grid, axis=0)
    dr = np.vstack((dr[1], dr))

    d_theta = np.diff(azimuthal_grid, axis=0)
    d_theta = np.vstack((d_theta[1], d_theta))

    front_rotors = weighting(front_rotors, dr, mode='repmat')
    front_rotors = weighting(front_rotors, d_theta, mode='repelem')/(2 * np.pi)

    rear_rotors = weighting(rear_rotors, dr, mode='repmat')
    rear_rotors = weighting(rear_rotors, d_theta, mode='repelem')/(2 * np.pi)

    force_front = params['n_b']*np.sum(front_rotors, axis=0)
    force_rear = params['n_b']*np.sum(rear_rotors, axis=0)

    force_front = np.around(force_front, decimals=2)
    force_rear = np.around(force_rear, decimals=2)

    force_front = force_front.reshape(1, -1)
    force_rear = force_rear.reshape(1, -1)

    return force_front, force_rear


def compute_thrust_cfd(u, params):

    dim_u = np.shape(u)[0]

    front_rotors = u[:dim_u // 2, :]
    rear_rotors = u[dim_u // 2:, :]

    azimuthal_grid = params['AzimuthalGrid'].reshape(-1, 1)
    azimuthal_grid = (np.pi/180)*azimuthal_grid

    d_theta = np.diff(azimuthal_grid, axis=0)
    d_theta = np.vstack((d_theta, d_theta[1]))

    front_rotors = weighting(front_rotors, d_theta, mode='repelem')/(2 * np.pi)
    force_front = params['n_b']*np.sum(front_rotors, axis=0)
    force_front = np.around(force_front, decimals=2)

    rear_rotors = weighting(rear_rotors, d_theta, mode='repelem')/(2 * np.pi)
    force_rear = params['n_b']*np.sum(rear_rotors, axis=0)
    force_rear = np.around(force_rear, decimals=2)

    force_front = force_front.reshape(1, -1)
    force_rear = force_rear.reshape(1, -1)

    return force_front, force_rear


def compute_torque_rmac(u, params):

    dim_u = np.shape(u)[0]

    front_rotors = u[:dim_u // 2, :]
    rear_rotors = u[dim_u // 2:, :]

    R = params['R']

    radial_grid = R * params['RadialGrid'].reshape(-1, 1)
    azimuthal_grid = params['AzimuthalGrid'].reshape(-1, 1)
    azimuthal_grid = (np.pi / 180) * azimuthal_grid

    dr = np.diff(radial_grid, axis=0)
    dr = np.vstack((dr[1], dr))

    d_theta = np.diff(azimuthal_grid, axis=0)
    d_theta = np.vstack((d_theta[1], d_theta))

    front_rotors = weighting(front_rotors, radial_grid, mode='repmat')
    front_rotors = weighting(front_rotors, dr, mode='repmat')
    front_rotors = weighting(front_rotors, d_theta, mode='repelem') / (2 * np.pi)

    rear_rotors = weighting(rear_rotors, radial_grid, mode='repmat')
    rear_rotors = weighting(rear_rotors, dr, mode='repmat')
    rear_rotors = weighting(rear_rotors, d_theta, mode='repelem') / (2 * np.pi)

    force_front = params['n_b'] * np.sum(front_rotors, axis=0)
    force_rear = params['n_b'] * np.sum(rear_rotors, axis=0)

    force_front = np.around(force_front, decimals=2)
    force_rear = np.around(force_rear, decimals=2)

    force_front = force_front.reshape(1, -1)
    force_rear = force_rear.reshape(1, -1)

    return force_front, force_rear


def compute_torque_cfd(u, params):

    dim_u = np.shape(u)[0]

    front_rotors = u[:dim_u // 2, :]
    rear_rotors = u[dim_u // 2:, :]

    radial_grid = params['RadialGrid'].reshape(-1, 1)

    azimuthal_grid = params['AzimuthalGrid'].reshape(-1, 1)
    azimuthal_grid = (np.pi/180)*azimuthal_grid

    d_theta = np.diff(azimuthal_grid, axis=0)
    d_theta = np.vstack((d_theta, d_theta[1]))

    front_rotors = weighting(front_rotors, radial_grid, mode='repmat')
    front_rotors = weighting(front_rotors, d_theta, mode='repelem')/(2 * np.pi)
    force_front = params['n_b']*np.sum(front_rotors, axis=0)
    force_front = np.around(force_front, decimals=2)

    rear_rotors = weighting(rear_rotors, radial_grid, mode='repmat')
    rear_rotors = weighting(rear_rotors, d_theta, mode='repelem')/(2 * np.pi)
    force_rear = params['n_b']*np.sum(rear_rotors, axis=0)
    force_rear = np.around(force_rear, decimals=2)

    force_front = force_front.reshape(1, -1)
    force_rear = force_rear.reshape(1, -1)

    return force_front, force_rear


