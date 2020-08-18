import gzip
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from mp.vmp import VMP
from obs_avoid_envs import ObsExp


class CollectDocking:
    def __init__(self, figure, ax, num_data, bfname, traj_dir):
        self.figure = figure
        self.ax = ax
        self.press = False
        self.vmp = VMP(dim=2, kernel_num=10)
        self.bfname = bfname
        self.traj_dir = traj_dir

        if not os.path.exists(traj_dir):
            try:
                os.makedirs(traj_dir)
            except OSError:
                raise

        self.id = 0


        self.num_data = num_data
        self.trqueries = np.random.uniform(low=(0,0,0,10,10,0), high=(10,10,2*np.pi, 20,20,2*np.pi),size=(num_data,6))
        obsExp = ObsExp(exp_name="Docking")
        self.envs = obsExp.get_envs(self.trqueries)
        self.next()
        self.connect()

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidkey = self.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        self.trajpoints.append(np.array([x, y]))
        self.press = True

    def on_motion(self, event):
        if not self.press: return
        x, y = event.xdata, event.ydata
        self.ax.plot(x, y, 'r.')
        self.trajpoints.append(np.array([x, y]))
        self.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = False
        traj = np.stack(self.trajpoints)
        timestamps = np.linspace(0, 1, np.shape(traj)[0])
        timestamps = np.expand_dims(timestamps, axis=1)
        traj = np.concatenate([timestamps, traj], axis=1)

        self.vmp.train(traj)
        newTraj = self.vmp.roll(traj[0,1:], traj[-1,1:])
        self.ax.cla()
        self.ax.set_xlim(-5, 25)
        self.ax.set_ylim(-5, 25)
        self.ax.set_aspect('equal')
        self.ax.plot(traj[:, 1], traj[:, 2], 'k-.')
        self.ax.plot(newTraj[:, 1], newTraj[:, 2], 'g-')
        self.env.plot(self.ax)

        weight = self.vmp.get_flatten_weights()
        starts = traj[0,1:]
        goals = traj[-1,1:]
        with open(self.bfname + "_" + "queries.csv", "a") as f:
            cquery = np.expand_dims(self.current_query, axis=0)
            np.savetxt(f, cquery, delimiter=',', fmt='%.3f')
        with open(self.bfname + "_" + "weights.csv", "a") as f:
            weight = np.expand_dims(weight, axis=1)
            np.savetxt(f, weight.transpose(), delimiter=',', fmt='%.5f')
        with open(self.bfname + "_" + "starts.csv", "a") as f:
            starts = np.expand_dims(starts, axis=0)
            np.savetxt(f, starts, delimiter=',', fmt='%.3f')
        with open(self.bfname + "_" + "goals.csv", "a") as f:
            goals = np.expand_dims(goals, axis=0)
            np.savetxt(f, goals, delimiter=',', fmt='%.3f')

        traj_fname = self.traj_dir + '/traj_' + str(self.id)
        np.savetxt(traj_fname, traj, delimiter=',', fmt='%.3f')
        # if self.id < self.num_data:
        #     self.next()
        # else:
        #     self.disconnect()

    def on_key(self, event):
        if event.key == 'z':
            self.next()

    def next(self):
        self.ax.cla()
        self.ax.set_xlim(-5, 25)
        self.ax.set_ylim(-5, 25)
        self.ax.set_aspect('equal')
        self.env = self.envs[self.id]
        self.current_query = self.trqueries[self.id,:]
        self.env.plot(self.ax)
        self.trajpoints = []
        msg = 'collect ' + str(self.id) + '-th data'
        self.figure.suptitle(msg)
        plt.draw()
        self.id = self.id + 1

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-n", "--num_data", dest="n_data", type="int", default=5)
    parser.add_option("-b", "--bfname", dest="bfname", type="string", default="docking")
    parser.add_option("-t", "--traj_dir", dest="traj_dir", type="string", default="docking_rawdata")

    (options, args) = parser.parse_args(sys.argv)
    fig, axes = plt.subplots(1, 1)

    collector = CollectDocking(fig, axes, options.n_data, options.bfname, options.traj_dir)
    plt.show()


