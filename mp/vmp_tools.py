import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')

from vmp import VMP
from promp import ProMP
import numpy as np
from math_tools.process_traj import TrajNormalizer
from math_tools.Quaternion import Quaternion
from optparse import OptionParser

class VMPDataTools:
    def __init__(self, mp, trajectory_data_directory, basic_data_filename):
        self.mp = mp
        self.data_dir = trajectory_data_directory
        self.basic_fname = basic_data_filename

    def get_flatten_weights_file(self, max_num_data=10000, ind=[0,1,2], process=None):
        weights = []
        starts = []
        goals = []

        for i in range(max_num_data):
            fname = self.basic_fname + "_" + str(i + 1) # + ".csv"
            fstr = os.path.join(self.data_dir, fname)
            if not os.path.exists(fstr):
                continue

            print(fstr)
            traj = np.loadtxt(fstr, delimiter=',')
            if ind is not None:
                traj = traj[:, ind]

            if process is not None:
                traj = process(traj)

            tprocessor = TrajNormalizer(traj)
            traj = tprocessor.normalize_timestamp()

            self.mp.train(traj)
            weight = self.mp.get_flatten_weights()
            y0 = traj[0, 1:]
            g = traj[-1, 1:]

            weights.append(weight)
            starts.append(y0)
            goals.append(g)

        ws = np.stack(weights)
        gs = np.stack(goals)
        sts = np.stack(starts)
        return ws, sts, gs



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m", "--mp_name", dest="mpname", type="string", default="VMP")
    parser.add_option("-t", "--trajectory_data_directory", dest="tfname", type="string", default="traj_dir")
    parser.add_option("-b", "--basic_data_filename", dest="bfname", type="string", default="basic_fname_")
    parser.add_option("-n", "--num_dim", dest="ndim", type="int", default=2)
    parser.add_option("-k", "--num_ker", dest="nker", type="int", default=10)
    (options, args) = parser.parse_args(sys.argv)

    if options.mpname == "VMP":
        mp = VMP(dim=options.ndim, kernel_num=options.nker)

    if options.mpname == "ProMP":
        mp = ProMP(dim=options.ndim, kernel_num=options.nker)

    tool = VMPDataTools(mp=mp, trajectory_data_directory=options.tfname, basic_data_filename=options.bfname)
    wouts, starts, goals = tool.get_flatten_weights_file(ind=None)
    np.savetxt('out_weights', wouts, delimiter=',')