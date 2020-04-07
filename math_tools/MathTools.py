import math
import numpy as np


def mat2rpy(R):
    a = ((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2
    a = max(a,-1)
    a = min(a,1)

    theta = np.arccos(a)
    if theta == 0:
        return np.zeros(3)
    else:
        multi = 1 / (2 * math.sin(theta))

        rpy = np.zeros(3)

        rpy[0] = multi * (R[2, 1] - R[1, 2]) * theta
        rpy[1] = multi * (R[0, 2] - R[2, 0]) * theta
        rpy[2] = multi * (R[1, 0] - R[0, 1]) * theta
        return rpy.copy()


def rpy2mat(rpy):
    m = np.zeros((3, 3))
    sgamma = np.sin(rpy[0])
    cgamma = np.cos(rpy[0])
    sbeta = np.sin(rpy[1])
    cbeta = np.cos(rpy[1])
    salpha = np.sin(rpy[2])
    calpha = np.cos(rpy[2])

    m[0, 0] = calpha * cbeta
    m[0, 1] = calpha * sbeta * sgamma - salpha * cgamma
    m[0, 2] = calpha * sbeta * cgamma + salpha * sgamma

    m[1, 0] = salpha * cbeta
    m[1, 1] = salpha * sbeta * sgamma + calpha * cgamma
    m[1, 2] = salpha * sbeta * cgamma - calpha * sgamma

    m[2, 0] = - sbeta
    m[2, 1] = cbeta * sgamma
    m[2, 2] = cbeta * cgamma
    return m


