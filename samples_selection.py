import numpy as np
from scipy import linalg
from paramspace_computation import get_coefficients, get_error
import time


def qr_pivoted(u, n):
    """Return a 2D array with the n most independent columns of u using QR pivoted decomposition"""

    _, _, p = linalg.qr(u, pivoting=True, mode='economic')

    selected_cols = p[0:n]

    print('')
    print('The most important samples indices are:')
    print(selected_cols)

    # Matrix with important snapshots
    u_n = u[:, selected_cols]

    return u_n, selected_cols


def iterative(U, n):
    """Return a 2D array with the n most independent columns of U using an iterative enrichment"""

    start_time = time.time()

    # Compute the n most important snapshots of U, stored in the matrix U_n
    # selected_samples is the array that contains their indices
    # The matrix V contains the remaining samples of U

    # Dimension and number of the snapshots
    dL, N = np.shape(U)[0], np.shape(U)[1]

    # Find the single most independent sample
    first_sample = 0
    Err = 0

    for sample in range(N):

        u = U[:, sample].reshape(dL, 1)

        W = np.delete(U, sample, 1)

        # Compute the error between u and its projection onto cols(W)
        c = get_coefficients(u, W)
        err = get_error(u, W, c)

        if err > Err:
            first_sample = sample
            Err = err

    # Array with the most independent samples
    U_n = U[:, first_sample].reshape(dL, 1)

    # Array with the indices of the selected samples
    selected_samples = np.array([first_sample])

    # Create the selected sample set
    for _ in range(n-1):

        best_sample = 0
        Err = 0

        for sample in range(N):

            # If the sample is already in the set, pass to the next
            if sample in selected_samples:
                pass

            else:
                # Select the sample and find u_hat = his projection onto the Gamma subset
                u = U[:, sample].reshape(dL, 1)

                # Compute the error between u and its projection onto cols(U_n)
                err = error_computation(u, U_n)

                # If the current error is the largest, select the sample
                if err > Err:
                    best_sample = sample
                    # Update the largest value
                    Err = err

        # Add the selected sample to Gamma
        new_sample = U[:, best_sample].reshape(dL, 1)
        U_n = np.hstack((U_n, new_sample))

        # Add the new index to the selected indices array
        selected_samples = np.append(selected_samples, best_sample)

    print("--- Execution time with iterative method is %s sec ---" % (time.time() - start_time))

    # V = np.delete(U, selected_samples, 1)

    print('')
    print('The most important samples indices are:')
    print(selected_samples)

    return U_n, selected_samples


def random(U, n):
    """Return a 2D array with n random columns of U"""

    N = np.shape(U)[1]

    selected_samples = np.random.randint(0, N, n).reshape(n)

    print('')
    print('The indices of the random samples are:')
    print(selected_samples)

    U_n = U[:, selected_samples]

    return U_n, selected_samples
