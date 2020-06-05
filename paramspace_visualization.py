import numpy as np
from import_data import load_rmac
from paramspace_computation import get_dx_dy
import matplotlib.pyplot as plt
import force_computation
from mpl_toolkits.mplot3d import axes3d, Axes3D


def visualize_snapshots_sets(params, snaps_rmac, params_cfd, snaps_cfd, save=False):

    dx_max, dx_min = params['dx_max'], params['dx_min']
    dy_max, dy_min = params['dy_max'], params['dy_min']

    Ndx, Ndy = params['Ndx'], params['Ndy']

    dx = np.linspace(dx_min, dx_max, Ndx)
    dy = np.linspace(dy_min, dy_max, Ndy)
    dX, dY = np.meshgrid(dx, dy)

    dx_rmac, dy_rmac = get_dx_dy(snaps_rmac, params)
    dx_cfd, dy_cfd = get_dx_dy(snaps_cfd, params_cfd)

    # Plot a grid
    plt.rc('grid', linestyle="-", color='black', lw=0.5)
    fig = plt.figure(figsize=(20, 12))
    plt.grid()
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Background
    ax = fig.gca()
    # Ones = dX / dX
    # contour_set = ax.contourf(dX, dY, Ones, cmap='Blues')
    ax.set_xticks(dx)
    ax.set_yticks(dy)
    ax.tick_params(labelsize=24)

    # Plot the RMAC snaps
    basis = plt.scatter(dx_rmac, dy_rmac, s=200, alpha=.8, c='b', marker='o', edgecolors='k', linewidths=1.5)
    basis.set_label('RMAC Snapshots')

    # for i, txt in enumerate(snaps_rmac):
    #      ax.annotate(txt, (dx_rmac[i], dy_rmac[i]), fontsize=22)

    # Plot the CFD snpas
    basis = plt.scatter(dx_cfd, dy_cfd, s=800, alpha=1, c='k', marker='X')
    basis.set_label('CFD Snapshots')

    ax.legend(prop={'size': 20}, bbox_to_anchor=(0.5, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=2)
    ax.set_xlabel('$d_x$', fontsize=32)
    ax.set_ylabel('$d_y$', fontsize=32, rotation=0)

    if save:
        plt.savefig('snapshots_sets.png', bbox_inches='tight')

    plt.show()


def visualize_cfd_snapshots(params_cfd, pivot_samples_cfd, reconstructed_samples, force_dir, save=False):

    _, params = load_rmac(force_dir)

    dx_max, dx_min = params['dx_max'], params['dx_min']
    dy_max, dy_min = params['dy_max'], params['dy_min']
    Ndx, Ndy = params['Ndx'], params['Ndy']

    dx = np.linspace(dx_min, dx_max, Ndx)
    dy = np.linspace(dy_min, dy_max, Ndy)
    dX, dY = np.meshgrid(dx, dy)

    Ones = dX / dX

    dx_selected, dy_selected = get_dx_dy(pivot_samples_cfd, params_cfd)

    # Plot a grid
    plt.rc('grid', linestyle="-", color='black', lw=0.8)
    fig = plt.figure(figsize=(20, 12))
    plt.grid()
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Plot the background
    ax = fig.gca()
    contour_set = ax.contourf(dX, dY, Ones, cmap='Blues', alpha=0.5)

    ax.set_xticks(dx)
    ax.set_yticks(dy)
    ax.tick_params(labelsize=18)

    # Plot the selected samples
    basis = plt.scatter(dx_selected, dy_selected, s=150, alpha=1, c='r')
    basis.set_label('Snapshots used as a Basis for the Reconstruction')

    # Enumerate the selected samples in the plot figure
    # n = np.shape(pivot_samples_cfd)[0]
    for i, txt in enumerate(pivot_samples_cfd+1):
        ax.annotate(txt, (dx_selected[i], dy_selected[i]), fontsize=35)

    dx_recon, dy_recon = get_dx_dy(reconstructed_samples, params_cfd)

    test = plt.scatter(dx_recon, dy_recon, s=700, alpha=1, c='g', marker='^')
    test.set_label('Reconstructed Snapshots')

    # n = np.shape(reconstructed_samples)[0]
    for i, txt in enumerate(reconstructed_samples+1):
        ax.annotate(txt, (dx_recon[i], dy_recon[i]), fontsize=35)

    ax.legend(prop={'size': 22}, bbox_to_anchor=(0., 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=1)
    ax.set_xlabel('$d_x$', fontsize=26)
    ax.set_ylabel('$d_y$', fontsize=26, rotation=0)

    if save:
        plt.savefig('CFD_snapshots.png', bbox_inches='tight')

    plt.show()


def error_contour(Error, selected_samples, params, force_dir, other_samples=None, save=False):
    """ Plots the error distribution in the parameter space """

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'mathtext.default': 'regular'})

    dx_max, dx_min = params['dx_max'], params['dx_min']
    dy_max, dy_min = params['dy_max'], params['dy_min']
    Ndx, Ndy = params['Ndx'], params['Ndy']

    dx = np.linspace(dx_min, dx_max, Ndx)
    dy = np.linspace(dy_min, dy_max, Ndy)
    dX, dY = np.meshgrid(dx, dy)

    Error = Error.reshape(Ndx, Ndy)

    dx_selected, dy_selected = get_dx_dy(selected_samples, params)

    print('')
    print('The most important snapshots are:')
    print(np.array(list(zip(dx_selected, dy_selected))))

    # Plot a grid
    plt.rc('grid', linestyle="-", color='black', lw=0.8)
    fig = plt.figure(figsize=(20, 12))
    plt.grid()
    plt.xticks(rotation=45, fontsize=40)
    plt.yticks(rotation=45, fontsize=40)
    plt.tick_params(labelsize=22)

    # Plot the contour of the error
    ax = fig.gca()
    contour_set = ax.contourf(dX, dY, Error.T, cmap='coolwarm')
    ax.set_xticks(dx)
    ax.set_yticks(dy)

    cbar = fig.colorbar(contour_set)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel('% Error', rotation=90, fontsize=32)

    # Plot the selected samples
    plt.scatter(dx_selected, dy_selected, s=100, alpha=1, c='b')

    # Enumerate the selected samples in the plot figure
    n = np.shape(selected_samples)[0]
    for i, txt in enumerate(np.arange(1, n + 1)):
        ax.annotate(txt, (dx_selected[i], dy_selected[i]), fontsize=70, c='k')

    # Plot other optional sample set
    if other_samples is not None:
        other_samples = np.array(other_samples)
        dx_other, dy_other = get_dx_dy(other_samples, params)

        # Plot the other samples
        plt.scatter(dx_other, dy_other, s=100, alpha=1, c='k')

        # Enumerate the selected samples in the plot figure
        n_other = np.shape(other_samples)[0]
        for i, txt in enumerate(np.arange(1, n_other + 1)):
            ax.annotate(txt, (dx_other[i], dy_other[i]), fontsize=20)

    ax.set_xlabel('$d_x/R$', fontsize=40)
    ax.set_ylabel('$d_y/R$', fontsize=40, rotation=0)

    if force_dir == 'L':
        plt.title('Relative Error Distribution using Lift Snapshots', fontsize=40)
    elif force_dir == 'D':
        plt.title('Relative Error Distribution using Drag Snapshots', fontsize=40)

    if save:
        plt.savefig('Error_dist_n=' + str(n) + '_{}.png'.format(force_dir), bbox_inches='tight')

    plt.show()


def response_surf(c, params, title=False, save=False):

    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'mathtext.default': 'regular'})

    dx_max, dx_min = params['dx_max'], params['dx_min']
    dy_max, dy_min = params['dy_max'], params['dy_min']
    Ndx, Ndy = params['Ndx'], params['Ndy']

    dx = np.linspace(dx_min, dx_max, Ndx)
    dy = np.linspace(dy_min, dy_max, Ndy)
    dX, dY = np.meshgrid(dx, dy)

    for i in range(np.shape(c)[0]):

        surf = c[i, :]
        surf = surf.reshape(Ndx, Ndy)

        # Plot a grid
        plt.rc('grid', linestyle="-", color='black', lw=0.8)
        fig = plt.figure(figsize=(12, 8))
        plt.grid()

        # Plot the surf
        ax = fig.gca(projection='3d')
        ax.plot_surface(dX, dY, surf.T, cmap='coolwarm')
        ax.set_xticks(dx)
        ax.set_yticks(dy)

        ax.set_xlabel('$d_x$', fontsize=20)
        ax.set_ylabel('$d_y$', fontsize=20)

        if title:
            ax.set_title(title)
            if save:
                plt.savefig(title, bbox_inches='tight')

    plt.show()


def response_contour(c, params, title=False, save=False):

    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'mathtext.default': 'regular'})

    dx_max, dx_min = params['dx_max'], params['dx_min']
    dy_max, dy_min = params['dy_max'], params['dy_min']
    Ndx, Ndy = params['Ndx'], params['Ndy']

    dx = np.linspace(dx_min, dx_max, Ndx)
    dy = np.linspace(dy_min, dy_max, Ndy)
    dX, dY = np.meshgrid(dx, dy)

    for i in range(np.shape(c)[0]):

        surf = c[i, :]
        surf = surf.reshape(Ndx, Ndy)

        # Plot a grid
        plt.rc('grid', linestyle="-", color='black', lw=0.8)
        fig = plt.figure(figsize=(15, 10))
        plt.grid()
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tick_params(labelsize=22)

        # Plot the contour of the error
        ax = fig.gca()
        contour_set = ax.contourf(dX, dY, surf.T, vmin = 118, vmax=127,cmap='coolwarm')
        cbar = fig.colorbar(contour_set)
        cbar.ax.get_yaxis().labelpad = 20

        ax.set_xticks(dx)
        ax.set_yticks(dy)
        ax.set_xlabel('$d_x$', fontsize=26)
        ax.set_ylabel('$d_y$', fontsize=26, rotation=0)

        if title:
            ax.set_title(title)
            if save:
                plt.savefig(title, bbox_inches='tight')

    plt.show()


def force_surfaces(U_LF_0, params, U_HF, U_HF_hat, params_HF, force_dir, snaps_train, snaps_val,front, save=False):

    # Plot a grid
    plt.rc('grid', linestyle="-", color='black', lw=0.4)
    figure = plt.figure(figsize=(12, 8))
    ax = figure.gca(projection='3d')

    plt.grid()
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'mathtext.default': 'regular'})

    # Mesh Grid
    dx_max, dx_min = params['dx_max'], params['dx_min']
    dy_max, dy_min = params['dy_max'], params['dy_min']
    Ndx, Ndy = params['Ndx'], params['Ndy']
    dx = np.linspace(dx_min, dx_max, Ndx)
    dy = np.linspace(dy_min, dy_max, Ndy)
    dX, dY = np.meshgrid(dx, dy)

    QoI = 'Thrust [N]' if force_dir == 'L' else 'Torque [Nm]'
    Rotor = 'Front Rotor ' if front else 'Rear Rotor '
    title = Rotor + QoI

    # LF surf
    if force_dir == 'L':
        [Q_LF_front, Q_LF_rear] = force_computation.compute_thrust_rmac(U_LF_0, params)
        if front:
            surf_LF = Q_LF_front.reshape(Ndx, Ndy)
        else:
            surf_LF = Q_LF_rear.reshape(Ndx, Ndy)
    else:
        [Q_LF_front, Q_LF_rear] = force_computation.compute_torque_rmac(U_LF_0, params)
        if front:
            surf_LF = Q_LF_front.reshape(Ndx, Ndy)
        else:
            surf_LF = Q_LF_rear.reshape(Ndx, Ndy)

    # HF Lifted surf
    if force_dir == 'L':
        [Q_HF_front_hat, Q_HF_rear_hat] = force_computation.compute_thrust_cfd(U_HF_hat, params_HF)
        if front:
            surf_HF_hat = Q_HF_front_hat.reshape(Ndx, Ndy)
        else:
            surf_HF_hat = Q_HF_rear_hat.reshape(Ndx, Ndy)
    else:
        [Q_HF_front_hat, Q_HF_rear_hat] = force_computation.compute_torque_cfd(U_HF_hat, params_HF)
        if front:
            surf_HF_hat = Q_HF_front_hat.reshape(Ndx, Ndy)
        else:
            surf_HF_hat = Q_HF_rear_hat.reshape(Ndx, Ndy)

    # HF points
    if force_dir == 'L':
        [Q_HF_front_hat, Q_HF_rear_hat] = force_computation.compute_thrust_cfd(U_HF, params_HF)
        if front:
            z_train = Q_HF_front_hat[:, snaps_train]
            z_val = Q_HF_front_hat[:, snaps_val]
        else:
            z_train = Q_HF_rear_hat[:, snaps_train]
            z_val = Q_HF_rear_hat[:, snaps_val]
    else:
        [Q_HF_front_hat, Q_HF_rear_hat] = force_computation.compute_torque_cfd(U_HF, params_HF)
        if front:
            z_train = Q_HF_front_hat[:, snaps_train]
            z_val = Q_HF_front_hat[:, snaps_val]
        else:
            z_train = Q_HF_rear_hat[:, snaps_train]
            z_val = Q_HF_rear_hat[:, snaps_val]

    # Plot Surf
    # ax.plot_surface(dX1, dY1, surf_LF.T, edgecolor='k', cmap='Reds', alpha=0.8, label='LF')
    # ax.plot_surface(dX1, dY1, surf_HF.T, edgecolor='k', cmap='Greens', alpha=0.6, label="HF lifted")

    ax.plot_wireframe(dX, dY, surf_LF.T, color='b', linewidth=2, label='RMAC')
    ax.plot_wireframe(dX, dY, surf_HF_hat.T, edgecolor='k', linewidth=2, label="CFD lifted")

    # Plot points
    x_train, y_train = get_dx_dy(snaps_train, params_HF)
    ax.scatter(x_train, y_train, z_train, s=150, color='k', alpha=1, label='CFD training')

    x_val, y_val = get_dx_dy(snaps_val, params_HF)
    ax.scatter(x_val, y_val, z_val, s=150, color='r', alpha=1, label='CFD validation')

    # ax.plot([2.5, 2.5], [0.25, 0.25], [1350, z_val[:, 0]], color='r')
    # ax.plot([3.0, 3.0], [0.00, 0.00], [1350, z_val[:, 1]], color='r')
    # ax.plot([3.0, 3.0], [0.25, 0.25], [1350, z_val[:, 2]], color='r')
    # ax.plot([3.0, 3.0], [0.50, 0.50], [1350, z_val[:, 3]], color='r')
    # ax.plot([3.5, 3.5], [0.25, 0.25], [1350, z_val[:, 4]], color='r')

    ax.set_xticks(dx[::2])
    plt.xticks(rotation=30, fontsize=14)
    ax.set_yticks(dy[::2])
    plt.yticks(fontsize=14)

    ax.set_zlim(1350, 1420)
    ax.view_init(20, -70)

    ax.set_xlabel('\n\n$d_x/R$', fontsize=22)
    ax.set_ylabel('\n\n$d_y/R$', fontsize=22)

    ax.set_title(title)
    plt.legend()

    if save:
        plt.savefig(title, bbox_inches='tight')

    plt.show()
