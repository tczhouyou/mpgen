import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
os.sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from math_tools.Quaternion import Quaternion
from vmp import VMP

class QVMP(VMP):
    def __init__(self, kernel_num=30, sigma=0.01, elementary_type='linear', use_outrange_kernel=True):
        VMP.__init__(self, dim=4, kernel_num=kernel_num, sigma=sigma, elementary_type=elementary_type, use_outrange_kernel=use_outrange_kernel)
        self.q0 = np.array([1,0,0,0])
        self.qg = np.array([1,0,0,0])
        self.muW = np.concatenate([np.ones(shape=(kernel_num,1)), np.zeros(shape=(kernel_num, 3))], axis=1)

    def set_start_goal(self, q0, qg):
        self.q0 = q0
        self.qg = qg

    def train(self, trajectory):
        self.n_samples = np.shape(trajectory)[0]
        tvec = trajectory[:,0]
        X = self.linearDecay(tvec)
        self.q0 = trajectory[0,1:]
        self.qg = trajectory[-1,1:]
        Psi = self.__Psi__(X)
        H_q = self.H(X, self.q0, self.qg)

        F_q = Quaternion.qtraj_diff(H_q, trajectory[:,1:])
        pseudo_inv = np.linalg.inv(np.matmul(np.transpose(Psi), Psi) + self.lamb * np.eye(self.kernel_num))
        self.muW = np.matmul(np.matmul(pseudo_inv, np.transpose(Psi)), F_q)

        return F_q

    def roll(self, qs=None, X=None):
        if qs is None:
            ts = np.array([0,1])
            ts = np.expand_dims(ts, axis=1)
            q = np.stack([self.q0,self.qg], axis=0)
            qs = np.concatenate([ts, q], axis=1)

        if X is None:
            X = self.linearDecayCanonicalSystem(1, 0, self.n_samples)

        ts = qs[:,0]
        H_q = np.zeros(shape=(len(X), 4))
        for i in range(len(X)):
            x = X[i]
            t = 1 - x
            ind = np.searchsorted(ts, t)

            if ind != 0:
                q0 = qs[ind-1,:]
                q1 = qs[ind, :]
            else:
                q0 = qs[0,:]
                q1 = qs[1,:]

            tq = (t - q0[0]) / (q1[0] - q0[0])
            H_q[i,:] = self.h(tq, q0[1:], q1[1:])

        Psi = self.__Psi__(X)
        F_q = np.matmul(Psi,self.muW)
        Xi = Quaternion.get_multi_qtraj(H_q, F_q)
        t = 1 - np.expand_dims(X,1)
        traj = np.concatenate([t, Xi], axis=1)
        return Quaternion.normalize_traj(traj)

    def h(self, t, q0, q1):
        if self.ElementaryType is 'linear':
            return Quaternion.slerp(t, q0, q1)
        if self.ElementaryType is 'minjerk':
            ratio = 6 * np.power(t, 5) - 15 * np.power(t,4) + 10 * np.power(t,3)
            return Quaternion.slerp(ratio, q0, q1)

    def H(self, X, q0, q1):
        if self.ElementaryType is 'linear':
            rvec = 1 - X
            return Quaternion.get_slerp_traj_(q0, q1, rvec)
        if self.ElementaryType is 'minjerk':
            rvec = 6 * np.power(1-X, 5) - 15 * np.power(1-X, 4) + 10 * np.power(1-X,3)
            return Quaternion.get_slerp_traj_(q0, q1, rvec)



if __name__ == '__main__':
    qvmp = QVMP(kernel_num=10, elementary_type='minjerk')
    qtraj = np.loadtxt('quaterniontest.csv', delimiter=',')
    F_q = qvmp.train(qtraj)
    q0 = qtraj[0, 1:]
    qvia = qtraj[20,1:]
    qg = qtraj[-1, 1:]

    ts = np.array([0, 0.5, 1])
    ts = np.expand_dims(ts, axis=1)
    q = np.stack([q0, qvia, qg], axis=0)
    qs = np.concatenate([ts, q], axis=1)

    qnewtraj = qvmp.roll(qs)
    for i in range(4):
        plt.plot(qtraj[:,0], qtraj[:,i+1], 'k-.')
        plt.plot(qnewtraj[:,0], qnewtraj[:,i+1], 'r-')

    plt.show()