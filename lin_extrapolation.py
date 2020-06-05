import numpy as np


def lin_extrapolation(diskplot1, diskplot2, diskplot3, t1, t2):

    u1, u2, u3 = diskplot1.u, diskplot2.u, diskplot3.u

    t11, t21 = diskplot1.theta1, diskplot1.theta2
    t12, t22 = diskplot2.theta1, diskplot2.theta2
    t13, t23 = diskplot3.theta1, diskplot3.theta2

    m = np.array([[t11, t21, 1],
                  [t12, t22, 1],
                  [t13, t23, 1]])

    inv_m = np.linalg.inv(m)

    a = inv_m[0, 0] * u1 + inv_m[0, 1] * u2 + inv_m[0, 2] * u3
    b = inv_m[1, 0] * u1 + inv_m[1, 1] * u2 + inv_m[1, 2] * u3
    c = inv_m[2, 0] * u1 + inv_m[2, 1] * u2 + inv_m[2, 2] * u3

    return a*t1 + b*t2 + c
