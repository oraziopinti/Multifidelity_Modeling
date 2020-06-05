import numpy as np


def get_coefficients(u, u_n):
    """ Returns the error between the cols of U and their projection U_hat onto cols(U_n)
        and the coefficients C of such projection, such that U_hat = U_n*C """

    # Linear system G*C = U_n.T*U finds the coefficients C
    # to approximate the cols of U using the span of U_n cols

    # Gramian matrix computation
    G = np.matmul(u_n.T, u_n)

    # RHS term f = U_n.T*U
    f = np.matmul(u_n.T, u)

    # Coefficients computation
    c = np.linalg.solve(G, f)

    return c


def get_error(u, u_n, c):

    # Approximation of u
    u_hat = np.matmul(u_n, c)

    # Approximation error vector
    error = 100 * np.linalg.norm(u - u_hat, axis=0) / np.linalg.norm(u, axis=0)

    return u_hat, error


def get_dx_dy(selected_samples, params):
    """Returns the values of dx and dy of selected samples"""

    dx_max, dx_min = params['dx_max'], params['dx_min']
    dy_max, dy_min = params['dy_max'], params['dy_min']
    Ndx, Ndy = params['Ndx'], params['Ndy']

    dx_step = (dx_max - dx_min) / (Ndx - 1)
    dy_step = (dy_max - dy_min) / (Ndy - 1)

    # Indices of the selected samples
    dx_selected_index = selected_samples // Ndy
    dy_selected_index = selected_samples - Ndy * dx_selected_index

    # Coordinates of the selected samples
    dx_selected = dx_min + dx_selected_index * dx_step
    dy_selected = dy_min + dy_selected_index * dy_step

    return dx_selected, dy_selected
