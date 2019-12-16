import numpy as np
import samples_selection as select
from paramspace_computation import error_computation
from paramspace_visualization import error_contour
from import_data import load_rmac, load_cfd
from disk_plot import disk_plot, disk_plot_comparison
from weighting import radial_weighting

# Number of important snapshots to select
n = 6

# Load snapshots
force_dir = 'D'
U_LF, params = load_rmac(force_dir)

# Weight snapshots matrix with radial values
U_LFw = radial_weighting(U_LF, params)

# Compute the n most important snapshots of U with a previous weighting due to polar coordinates
U_LFw_n, selected_samples = select.qr_pivoted(U_LFw, n)

# Computation of the error distribution and the coefficients C of for the approximation
Error, C = error_computation(U_LFw, U_LFw_n)
error_contour(Error, pivot_samples_rmac, params, force_dir, save=True)

# Load CFD snapshots
U_HF, params_HF = load_cfd(force_dir)

# Approximation of HF snapshots set
U_HF_hat = np.matmul(U_HF, C)




