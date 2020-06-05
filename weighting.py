import numpy as np
import numpy.matlib


def radial_weighting(u_pol, params):
    """ The cols of u are the snapshots, which are nodal values in a polar grid.
        To compute correctly the norms of the errors between snapshots, we weight
        these nodal values with the square root of the correspondent radius value. """

    n_r = params['n_r']

    # Column vector with radial grid values
    r_vector = params['RadialGrid'].reshape(n_r, 1)

    # The wighting is done with the square root of these values
    r_weight = np.sqrt(r_vector)

    # The vector is repeated for each value of the azimuthal grid (actually twice,
    # because the snapshots is made by two disk plots)
    dim_u = np.shape(u_pol)[0]
    rep = dim_u//n_r

    weight = np.matlib.repmat(r_weight, rep, 1)

    # The snapshots matrix is weighted with the square roots of radial values
    u_cart = np.multiply(weight, u_pol)

    return u_cart


def weighting(u_in, weight_vector, mode):
    """ The cols of u are weighted point-wise with the column vector weight_vector. If mode=repmat, the entire vector
    weight_vector is repeated until its dimension match the cols one of u_in, and then multiplied point-wise.
    If mode=repelem, every element of the vector weight_vector is repeated."""

    dim_u = np.shape(u_in)[0]
    dim_weight = np.shape(weight_vector)[0]

    rep = dim_u//dim_weight

    if mode == 'repmat':
        weight = np.matlib.repmat(weight_vector, rep, 1)
    elif mode == 'repelem':
        weight = np.repeat(weight_vector, repeats=rep, axis=0)
    else:
        weight = np.ones(dim_u, 1)
        print('Error: no weighting has been done')

    # The snapshots matrix is weighted the vector weight_vector
    u_out = np.multiply(weight, u_in)

    return u_out


def undo_radial_weighting(u_cart, params):
    """ The cols of U are the snapshots, which are nodal values in a polar grid.
        To compute correctly the norms of the errors between snapshots, we weight
        these nodal values with the square root of the correspondent radius value. """

    n_r = params['n_r']

    # Column vector with radial grid values
    r_vector = params['RadialGrid'].reshape(n_r, 1)

    # The wighting is done with the inverse of square root of these values
    r_weight = 1/np.sqrt(r_vector)

    # The vector is repeated for each value of the azimuthal grid (actually twice,
    # because the snapshots is made by two disk plots)

    dim_u = np.shape(u_pol)[0]
    rep = dim_u // n_r

    weight = np.matlib.repmat(r_weight, rep, 1)

    # The snapshots matrix is weighted with the square roots of radial values
    u_pol = np.multiply(weight, u_cart)

    return u_pol
