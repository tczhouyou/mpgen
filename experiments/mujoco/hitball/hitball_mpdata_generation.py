import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
from mp.vmp import VMP
from mp.vmp_tools import VMPDataTools
from hitball_exp import Armar6HitBallExp, init_ball_pos, raw_data_dir, mp_data_dir

vmp = VMP(dim=2, kernel_num=10)
vmp_data_processor = VMPDataTools(vmp, raw_data_dir, "hitball")
weights, starts, goals = vmp_data_processor.get_flatten_weights_file()

np.savetxt(mp_data_dir + '/hitball_weights.csv', np.stack(weights), delimiter=',')
np.savetxt(mp_data_dir + '/hitball_starts.csv', np.stack(starts), delimiter=',')
np.savetxt(mp_data_dir + '/hitball_goals.csv', np.stack(goals), delimiter=',')
