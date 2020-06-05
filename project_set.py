import numpy as np
from paramspace_computation import get_coefficients, get_error


def project_set(u, snaps_indices):

    if snaps_indices.size == 0:

        return u

    else:

        basis = u[:, snaps_indices]

        c = get_coefficients(u, basis)

        u_hat, _ = get_error(u, basis, c)

        u = u - u_hat

        u[:, snaps_indices] = 0

        # u = np.delete(u, snaps_indices, 1)

        return u
