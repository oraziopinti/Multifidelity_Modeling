import numpy as np
import matplotlib.pyplot as plt
from paramspace_computation import get_dx_dy
from force_computation import compute_thrust_rmac, compute_thrust_cfd, compute_torque_rmac, compute_torque_cfd
from weighting import weighting
from l2error import l2error, matrix_interpol


def disk_plot(u, location, params, force_dir, save=False, cfd=False):
    """ Returns the disk plots of u seen as front and rear rotor r-theta matrix """

    if cfd:
        radial_grid = params['RadialGrid'].reshape(-1, 1)
        dr = np.diff(radial_grid, axis=0)
        dr = np.vstack((radial_grid[1], dr))
        u = weighting(u, 1 / dr, mode='repmat')

    n_row = np.shape(u)[0]

    dx, dy = get_dx_dy(location, params)

    n_r = params['n_r']
    n_theta = params['n_theta']

    azimuths = np.radians(np.linspace(0, 360, n_theta))
    zeniths = params['RadialGrid']

    theta, r = np.meshgrid(azimuths, zeniths)

    front_rotor = u[0:n_row//2]
    rear_rotor = u[n_row//2:]

    front_rotor = np.reshape(front_rotor, (n_r, n_theta), 'F')
    rear_rotor = np.reshape(rear_rotor, (n_r, n_theta), 'F')

    plt.rc('grid', linestyle="-", color='black', lw=0.8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), subplot_kw=dict(projection='polar'))

    force = "Lift" if force_dir == "L" else "Drag"

    # config = "$(d_x, d_y)$ = (" + str(dx) + ", " + str(dy) + ")"
    # title = fig.suptitle(("Disk plots of {} [N/m] at" + config).format(force), fontsize=16)
    # title.set_position([.5, 0.9])

    # type_snap = 'cfd' if cfd else 'rmac'
    # err_front, err_rear = l2error(u1, params, type_snap, u2, params, type_snap)

    # ax1.set_title('Front L2 Error = ' + str(err_front) + '%', fontsize=20, pad=20)
    # ax2.set_title('Rear L2 Error = ' + str(err_rear) + '%', fontsize=20, pad=20)

    ax1.set_title('Front Rotor', fontsize=20, pad=20)
    ax2.set_title('Rear Rotor', fontsize=20, pad=20)

    im1 = ax1.contourf(theta, r, front_rotor, vmin=0, levels=100, cmap='rainbow')
    ax1.set_theta_direction(-1)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.grid()

    im2 = ax2.contourf(theta, r, rear_rotor, vmin=0, levels=100, cmap='rainbow')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    plt.grid()
    plt.show()

    if save:
        fig_name = ("Disk_plot_{}_dx=" + str(dx) + "dy=" + str(dy) + ".png").format(force)
        plt.savefig(fig_name, bbox_inches='tight')


def disk_plot_comparison_old(u1, u2, params, force_dir, location, save=False, cfd=False):

    force = "Lift" if force_dir == "L" else "Drag"
    type_snap = 'cfd' if cfd else 'rmac'
    err_front, err_rear = l2error(u1, params, type_snap, u2, params, type_snap)

    if cfd:
        radial_grid = params['RadialGrid'].reshape(-1, 1)
        dr = np.diff(radial_grid, axis=0)
        dr = np.vstack((radial_grid[1], dr))
        u1 = weighting(u1, 1 / dr, mode='repmat')
        u2 = weighting(u2, 1 / dr, mode='repmat')

    if np.shape(u1)[0] == np.shape(u2)[0]:
        n_row = np.shape(u1)[0]
    else:
        print('Dimensions of Snapshot to compare must be the same')
        return

    dx, dy = get_dx_dy(location, params)

    n_r = params['n_r']
    n_theta = params['n_theta']

    azimuths = np.radians(params['AzimuthalGrid'])
    zeniths = params['RadialGrid']

    theta, r = np.meshgrid(azimuths, zeniths)

    front_rotor = u1[0:n_row // 2]
    rear_rotor = u1[n_row // 2:]

    front_rotor1 = np.reshape(front_rotor, (n_r, n_theta), 'F')
    rear_rotor1 = np.reshape(rear_rotor, (n_r, n_theta), 'F')

    front_rotor = u2[0:n_row // 2]
    rear_rotor = u2[n_row // 2:]

    front_rotor2 = np.reshape(front_rotor, (n_r, n_theta), 'F')
    rear_rotor2 = np.reshape(rear_rotor, (n_r, n_theta), 'F')

    plt.rc('grid', linestyle="-", color='black', lw=0.8)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(wspace=0.3, hspace=0.35)

    # config = "$(d_x, d_y)$ = (" + str(dx) + ", " + str(dy) + ")"
    # title = fig.suptitle(("Disk Plots of {} [N/m] at " + config).format(force), fontsize=14)
    # title.set_position([0.5, 0.95])

    axes[0, 0].set_title('Front Rotor Snapshot', fontsize=12, pad=15)
    axes[0, 1].set_title('Rear Rotor Snapshot', fontsize=12, pad=15)
    axes[1, 0].set_title('Front Rotor Reconstruction', fontsize=12, pad=15)
    axes[1, 1].set_title('Rear Rotor Reconstruction', fontsize=12, pad=15)
    axes[2, 0].set_title('Difference, L2 error =' + str(err_front) + '%', fontsize=12, pad=15)
    axes[2, 1].set_title('Difference, L2 error =' + str(err_rear) + '%', fontsize=12, pad=15)

    im1 = axes[0, 0].contourf(theta, r, front_rotor1, vmax=2700, levels=150, cmap='rainbow')
    axes[0, 0].set_theta_direction(-1)
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    plt.grid()

    im2 = axes[0, 1].contourf(theta, r, rear_rotor1, vmax=2700, levels=150, cmap='rainbow')
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    plt.grid()

    im3 = axes[1, 0].contourf(theta, r, front_rotor2, vmax=2700, levels=150, cmap='rainbow')
    axes[1, 0].set_theta_direction(-1)
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    plt.grid()

    im4 = axes[1, 1].contourf(theta, r, rear_rotor2, vmax=2700, levels=150, cmap='rainbow')
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    plt.grid()

    im5 = axes[2, 0].contourf(theta, r, front_rotor1-front_rotor2, vmax=500, vmin=-500, levels=100, cmap='rainbow')
    axes[2, 0].set_theta_direction(-1)
    fig.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)
    plt.grid()

    im6 = axes[2, 1].contourf(theta, r, rear_rotor1-rear_rotor2, vmax=500, vmin=-500, levels=100, cmap='rainbow')
    fig.colorbar(im6, ax=axes[2, 1], fraction=0.046, pad=0.04)
    plt.grid()

    if save:
        fig_name = ("Disk_Plot_{}_Comparison_dx=" + str(dx) + "dy =" + str(dy) + ".png").format(force)
        plt.savefig(fig_name, bbox_inches='tight')

    plt.show()


def disk_plot_comparison(u1, params1, type1, u2, params2, type2, force_dir, location, save=False):

    force = "Lift" if force_dir == "L" else "Drag"
    err_front, err_rear = l2error(u1, params1, type1, u2, params2, type2)
    dx, dy = get_dx_dy(location, params1)

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

    azimuths1 = np.radians(params1['AzimuthalGrid'])
    zeniths1 = params1['RadialGrid']
    theta1, r1 = np.meshgrid(azimuths1, zeniths1)

    n_row = np.shape(u1)[0]
    n_r, n_theta = params1['n_r'], params1['n_theta']

    front_rotor = u1[0:n_row // 2]
    rear_rotor = u1[n_row // 2:]

    front_rotor1 = np.reshape(front_rotor, (n_r, n_theta), 'F')
    rear_rotor1 = np.reshape(rear_rotor, (n_r, n_theta), 'F')

    azimuths2 = np.radians(params2['AzimuthalGrid'])
    zeniths2 = params2['RadialGrid']
    theta2, r2 = np.meshgrid(azimuths2, zeniths2)

    n_row = np.shape(u2)[0]
    n_r, n_theta = params2['n_r'], params2['n_theta']

    front_rotor = u2[0:n_row // 2]
    rear_rotor = u2[n_row // 2:]

    front_rotor2 = np.reshape(front_rotor, (n_r, n_theta), 'F')
    rear_rotor2 = np.reshape(rear_rotor, (n_r, n_theta), 'F')

    plt.rc('grid', linestyle="-", color='black', lw=0.8)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(wspace=0.3, hspace=0.35)

    config = "$(d_x, d_y)$ = (" + str(dx) + ", " + str(dy) + ")"
    title = fig.suptitle(("Disk Plots of {} [N/m] at " + config).format(force), fontsize=16)
    title.set_position([0.55, 1])

    axes[0, 0].set_title('Front Rotor \nCFD', fontsize=14, pad=15)
    axes[0, 1].set_title('Rear Rotor \nCFD', fontsize=14, pad=15)
    axes[1, 0].set_title('RMAC', fontsize=12, pad=15)
    axes[1, 1].set_title('RMAC', fontsize=12, pad=15)
    axes[2, 0].set_title('Difference $L_2$ Error =' + str(err_front) + '%', fontsize=12, pad=15)
    axes[2, 1].set_title('Difference $L_2$ Error =' + str(err_rear) + '%', fontsize=12, pad=15)

    im1 = axes[0, 0].contourf(theta1, r1, front_rotor1, vmin=0, vmax=450, levels=150, cmap='rainbow')
    axes[0, 0].set_theta_direction(-1)
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    plt.grid()

    im2 = axes[0, 1].contourf(theta1, r1, rear_rotor1, vmin=0, vmax=450, levels=150, cmap='rainbow')
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    plt.grid()

    im3 = axes[1, 0].contourf(theta2, r2, front_rotor2, vmin=0, vmax=450, levels=150, cmap='rainbow')
    axes[1, 0].set_theta_direction(-1)
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    plt.grid()

    im4 = axes[1, 1].contourf(theta2, r2, rear_rotor2, vmin=0, vmax=450, levels=150, cmap='rainbow')
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    plt.grid()

    [u1, u2, ax1, ax2] = matrix_interpol(front_rotor1, zeniths1, azimuths1, front_rotor2, zeniths2, azimuths2)

    im5 = axes[2, 0].contourf(ax2, ax1, u1-u2, vmin=-150, vmax=150, levels=100, cmap='rainbow')
    axes[2, 0].set_theta_direction(-1)
    fig.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)
    plt.grid()

    [u1, u2, ax1, ax2] = matrix_interpol(rear_rotor1, zeniths1, azimuths1, rear_rotor2, zeniths2, azimuths2)

    im6 = axes[2, 1].contourf(ax2, ax1, u1-u2, vmin=-150, vmax=150, levels=100, cmap='rainbow')
    fig.colorbar(im6, ax=axes[2, 1], fraction=0.046, pad=0.04)
    plt.grid()

    if save:
        model = 'RMAC' if type2 == 'rmac' else 'CFD'
        fig_name = ("{}_{}_dx=" + str(dx) + "dy =" + str(dy) + ".png").format(model, force)
        plt.savefig(fig_name, bbox_inches='tight')

    plt.show()

