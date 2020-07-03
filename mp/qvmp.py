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
        H_q = self.H(X)

        F_q = Quaternion.qtraj_diff(H_q, trajectory[:,1:])
        pseudo_inv = np.linalg.inv(np.matmul(np.transpose(Psi), Psi) + self.lamb * np.eye(self.kernel_num))
        self.muW = np.matmul(np.matmul(pseudo_inv, np.transpose(Psi)), F_q)

        return F_q

    def roll(self, q0, qg, X=None):
        if X is None:
            X = self.linearDecayCanonicalSystem(1, 0, self.n_samples)

        self.q0 = q0
        self.qg = qg
        H_q = self.H(X)
        Psi = self.__Psi__(X)
        F_q = np.matmul(Psi,self.muW)
        Xi = Quaternion.get_multi_qtraj(H_q, F_q)
        t = 1 - np.expand_dims(X,1)
        traj = np.concatenate([t, Xi], axis=1)
        return Quaternion.normalize_traj(traj)

    def h(self, x):
        if self.ElementaryType is 'linear':
            ratio = 1 - x
            return Quaternion.slerp(ratio, self.q0, self.qg)
        if self.ElementaryType is 'minjerk':
            ratio = 6 * np.power(1-x, 5) - 15 * np.power(1-x,4) + 10 * np.power(1-x,3)
            return Quaternion.slerp(ratio, self.q0, self.qg)

    def H(self, X):
        if self.ElementaryType is 'linear':
            rvec = 1 - X
            return Quaternion.get_slerp_traj_(self.q0, self.qg, rvec)
        if self.ElementaryType is 'minjerk':
            rvec = 6 * np.power(1-X, 5) - 15 * np.power(1-X, 4) + 10 * np.power(1-X,3)
            return Quaternion.get_slerp_traj_(self.q0, self.qg, rvec)


if __name__ == '__main__':
    qvmp = QVMP(kernel_num=10, elementary_type='minjerk')
    qtraj = np.loadtxt('quaterniontest.csv', delimiter=',')
    # F_q = qvmp.train(qtraj)
    q0 = qtraj[0, 1:]
    qg = qtraj[-1, 1:]
    qnewtraj = qvmp.roll(q0, qg)
    for i in range(4):
        plt.plot(qtraj[:,0], qtraj[:,i+1], 'k-.')
        plt.plot(qnewtraj[:,0], qnewtraj[:,i+1], 'r-')
        # plt.plot(qnewtraj[:,0], F_q[:,i], 'b-')

    plt.show()