import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
from mp.qvmp import QVMP
from mp.vmp_tools import VMPDataTools
from optparse import OptionParser
from math_tools.Quaternion import Quaternion

parser = OptionParser()
parser.add_option("-r", "--raw_dir", dest="raw_dir", type="string", default="balanceball_rawdata")
parser.add_option("-d", "--mp_dir", dest="mp_dir", type="string", default="balanceball_mpdata")
(options, args) = parser.parse_args(sys.argv)

vmp = QVMP(kernel_num=10, elementary_type='minjerk')
vmp_data_processor = VMPDataTools(vmp, options.raw_dir, "balanceball")
weights, starts, goals = vmp_data_processor.get_flatten_weights_file(ind=None, process=Quaternion.process_qtraj)

np.savetxt(options.mp_dir + '/balanceball_weights.csv', np.stack(weights), delimiter=',')
np.savetxt(options.mp_dir + '/balanceball_starts.csv', np.stack(starts), delimiter=',')
np.savetxt(options.mp_dir + '/balanceball_goals.csv', np.stack(starts), delimiter=',')
