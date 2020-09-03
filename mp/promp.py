import numpy as np
import matplotlib.pyplot as plt


class ProMP:
    def __init__(self, dim, kernel_num=30, sigma=0.01, use_outrange_kernel=True):
        self.kernel_num = kernel_num
        if use_outrange_kernel:
            self.centers = np.linspace(1.2, -0.2, kernel_num)
        else:
            self.centers = np.linspace(1,0,kernel_num)

        self.D = sigma
        self.lamb = 0.01
        self.dim = dim
        self.n_samples = 100
        self.muW = np.zeros(shape=(kernel_num, self.dim))

    def __psi__(self, x):
        return np.exp(-0.5 * np.multiply(np.square(x - self.centers), 1/self.D))

    def __Psi__(self, X):
        X = np.array(X)
        Xmat = np.transpose(np.tile(X, (self.kernel_num,1)))
        Cmat = np.tile(self.centers, (np.shape(X)[0], 1))
        return np.exp(-0.5 * np.multiply(np.square(Xmat - Cmat), 1 / self.D))

    def train(self, trajectories):
        if len(np.shape(trajectories)) == 2:
            trajectories = np.expand_dims(trajectories, 0)

        n_data = np.shape(trajectories)[0]
        self.n_samples = np.shape(trajectories)[1]
        n_dim = np.shape(trajectories)[2] - 1

        self.dim = n_dim
        X = self.linearDecayCanonicalSystem(1,0, self.n_samples )
        Psi = self.__Psi__(X)
        F = trajectories[:,:,1:]

        pseudo_inv = np.linalg.inv(np.matmul(np.transpose(Psi), Psi) + self.lamb * np.eye(self.kernel_num))
        W = np.matmul(np.matmul(pseudo_inv, np.transpose(Psi)), F)

        self.muW = np.sum(W, 0) / n_data

        return

    def save_weights_to_file(self, filename):
        np.savetxt(filename, self.muW, delimiter=',')

    def load_weights_from_file(self, filename):
        self.muW = np.loadtxt(filename, delimiter=',')

    def get_weights(self):
        return self.muW

    def get_flatten_weights(self):
        return self.muW.flatten('F')

    def set_weights(self, ws):
        if np.shape(ws)[-1] == self.dim * self.kernel_num:
            self.muW = np.reshape(ws, (self.kernel_num, self.dim), 'F')
        elif np.shape(ws)[0] == self.kernel_num and np.shape(ws)[-1] == self.dim:
            self.muW = ws
        else:
            raise Exception("The weights have wrong shape. It should have {} rows (for kernel number) and {} columns (for dimensions)".format(self.kernel_num, self.dim))

    def get_position(self, t):
        x = 1 - t
        return np.matmul(self.__psi__(x), self.muW)

    def roll(self):
        X = self.linearDecayCanonicalSystem(1,0, self.n_samples)
        Psi = self.__Psi__(X)
        Xi = np.matmul(Psi,self.muW)

        t = 1 - np.expand_dims(X,1)
        traj = np.concatenate([t, Xi], axis=1)
        return traj

    def get_target(self, t):
        action = np.transpose(self.h(1-t)) + self.get_position(t)
        return action

    def get_action_with_explored_param(self, t, ws):
        self.set_weights(ws)
        return self.get_action(t)

    def test(self):
        traj = self.roll()
        for i in range(np.shape(traj)[-1]):
            plt.plot(traj[:,i])

        plt.show()

    def cansys(self, t):
        return 1 - t

    def linearDecayCanonicalSystem(self, t0, t1, numOfSamples):
        return np.linspace(t0, t1, numOfSamples)



if __name__ == '__main__':
    promp = ProMP(1)
    t = np.linspace(0,1,1000)
    traj0 = np.stack([t,np.sin(t * 2 * np.pi)])
    traj0 = np.transpose(traj0)
    traj1 = np.stack([t,np.cos(t * 2 * np.pi)])
    traj1 = np.transpose(traj1)

    trajs = np.stack([traj0, traj1])
    promp.train(trajs)
    Xi = promp.roll() #y0=(traj0[0,1:]+traj1[0,1:])/2, g=(traj0[-1,1:]+traj1[-1,1:])/2)

    plt.plot(t, traj0[:,1:], 'r')
    plt.plot(t, traj1[:,1:], 'g')
    plt.plot(t, Xi[:,1:], 'b')
    plt.show()

    xi = promp.get_position(0.5,1)
    print(xi)