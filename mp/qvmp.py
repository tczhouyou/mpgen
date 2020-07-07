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
        ts = np.array([0, 1])
        ts = np.expand_dims(ts, axis=1)
        q = np.stack([self.q0, self.qg], axis=0)
        self.qs = np.concatenate([ts, q], axis=1)

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

        ts = np.array([0, 1])
        ts = np.expand_dims(ts, axis=1)
        q = np.stack([self.q0, self.qg], axis=0)
        self.qs = np.concatenate([ts, q], axis=1)
        return F_q

    def set_qs(self, ts, qlist):
        ts = np.expand_dims(ts, axis=1)
        q = np.stack(qlist, axis=0)
        self.qs = np.concatenate([ts, q], axis=1)

    def get_target(self, t):
        ts = self.qs[:,0]
        ind = np.searchsorted(ts, t)
        if ind != 0:
            q0 = self.qs[ind - 1, :]
            q1 = self.qs[ind, :]
        else:
            q0 = self.qs[0, :]
            q1 = self.qs[1, :]

        tq = (t - q0[0]) / (q1[0] - q0[0])
        h_q = self.h(tq, q0[1:], q1[1:])
        f_q = self.get_position(t)
        y_q = Quaternion.qmulti(h_q, f_q)
        y_q = Quaternion.normalize(y_q)
        return y_q

    def get_position(self, t):
        x = 1 - t
        return np.matmul(self.__psi__(x), self.muW)

    def roll(self, X=None):
        if X is None:
            X = self.linearDecayCanonicalSystem(1, 0, self.n_samples)

        Xi = np.zeros(shape=(len(X),4))
        for i in range(len(X)):
            x = X[i]
            t = 1 - x
            Xi[i,:] = self.get_target(t)

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
    qvmp.set_qs(ts, [q0, qvia,q0])
    qnewtraj = qvmp.roll()
    for i in range(4):
        plt.plot(qtraj[:,0], qtraj[:,i+1], 'k-.')
        plt.plot(qnewtraj[:,0], qnewtraj[:,i+1], 'r-')

    plt.show()