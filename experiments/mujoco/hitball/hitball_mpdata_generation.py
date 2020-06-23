import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
from mp.vmp import VMP
from mp.vmp_tools import VMPDataTools
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-r", "--raw_dir", dest="raw_dir", type="string", default="hitball_mpdata_v0")
parser.add_option("-d", "--mp_dir", dest="mp_dir", type="string", default="hitball_dataset_v0")
(options, args) = parser.parse_args(sys.argv)

vmp = VMP(dim=2, kernel_num=10)
vmp_data_processor = VMPDataTools(vmp, options.raw_dir, "hitball")
weights, starts, goals = vmp_data_processor.get_flatten_weights_file()

np.savetxt(options.mp_dir + '/hitball_weights.csv', np.stack(weights), delimiter=',')
np.savetxt(options.mp_dir + '/hitball_starts.csv', np.stack(starts), delimiter=',')
np.savetxt(options.mp_dir + '/hitball_goals.csv', np.stack(goals), delimiter=',')
