from import_data import load_cfd, load_rmac
from disk_plot import disk_plot, disk_plot_comparison
from paramspace_visualization import visualize_cfd_snapshots, visualize_snapshots_sets
import numpy as np


force_dir = 'D'


U_HF, params_cfd = load_cfd(force_dir)
U, params = load_rmac(force_dir)

# Disk Plots
for loc in range(0, 9):
    u1 = U_HF[:, loc].reshape(-1, 1)
    u2 = U_HF[:, 4].reshape(-1, 1)
    u = u1 - u2
    # disk_plot(u, u1, u2, loc, params_cfd, force_dir, save=True, cfd=True, show=False)

# visualize_cfd_snapshots(params_cfd, np.array([2, 6, 8]), np.array([0]), force_dir=force_dir, save=True)

snaps_cfd = np.array([0, 2, 6, 8])
snaps_test = np.array([1, 3, 4, 5, 7])
snaps_rmac = np.arange(105)

from paramspace_computation import get_dx_dy
import matplotlib.pyplot as plt

dx_max, dx_min = params['dx_max'], params['dx_min']
dy_max, dy_min = params['dy_max'], params['dy_min']

Ndx, Ndy = params['Ndx'], params['Ndy']

dx = np.linspace(dx_min, dx_max, Ndx)
dy = np.linspace(dy_min, dy_max, Ndy)
dX, dY = np.meshgrid(dx, dy)

dx_rmac, dy_rmac = get_dx_dy(snaps_rmac, params)
dx_test, dy_test = get_dx_dy(snaps_test, params_cfd)
dx_cfd, dy_cfd = get_dx_dy(snaps_cfd, params_cfd)

# Plot a grid
plt.rc('grid', linestyle="-", color='black', lw=0.4)
fig = plt.figure(figsize=(18, 10))
plt.grid()
plt.xticks(rotation=45)
plt.yticks(rotation=20)


# Background
ax = fig.gca()
ax.set_xticks(dx)
ax.set_yticks(dy)
ax.tick_params(labelsize=24)

some_inds = np.array([0, 39, 48, 59, 99])
print(some_inds)
basis = plt.scatter(dx_rmac[some_inds], dy_rmac[some_inds], s=200, alpha=.8, c='k', marker='X', edgecolors='k', linewidths=1.5)
basis.set_label('CFD Snapshots')

#for i, txt in enumerate(np.arange(5)+1):
#    ax.annotate(txt, (dx_test[i]+0.02, dy_test[i]+0.02), fontsize=30)

# Plot the CFD snpas
#basis = plt.scatter(dx_cfd, dy_cfd, s=800, alpha=1, c='g', marker='X')
#basis.set_label('CFD Training Snapshots')

#basis = plt.scatter(dx_test, dy_test, s=500, alpha=1, c='r', marker='s')
#basis.set_label('CFD Validation Snapshots')

ax.legend(prop={'size': 20}, bbox_to_anchor=(0.75, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=3)
ax.set_xlabel('$d_x/R$', fontsize=32)
ax.set_ylabel('$d_y/R  $', fontsize=32, rotation=0)

plt.savefig('snapshots_sets.png', bbox_inches='tight')

plt.show()
