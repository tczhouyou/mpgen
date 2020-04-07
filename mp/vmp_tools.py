import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')

from vmp import VMP
import numpy as np
from math_tools.process_traj import TrajNormalizer


class VMPDataTools:
    def __init__(self, vmp, trajectory_data_directory, basic_data_filename):
        self.vmp = vmp
        self.data_dir = trajectory_data_directory
        self.basic_fname = basic_data_filename

    def get_flatten_weights_file(self, max_num_data=10000):
        weights = []
        starts = []
        goals = []

        for i in range(max_num_data):
            fname = self.basic_fname + "_" + str(i + 1) + ".csv"
            fstr = os.path.join(self.data_dir, fname)
            if not os.path.exists(fstr):
                continue

            print(fstr)
            traj = np.loadtxt(fstr, delimiter=',')
            traj = traj[:, :3]
            tprocessor = TrajNormalizer(traj)
            traj = tprocessor.normalize_timestamp()

            self.vmp.train(traj)
            weight = self.vmp.get_flatten_weights()
            y0 = traj[0, 1:]
            g = traj[-1, 1:]

            weights.append(weight)
            starts.append(y0)
            goals.append(g)

        ws = np.stack(weights)
        gs = np.stack(goals)
        sts = np.stack(starts)
        return ws, sts, gs
