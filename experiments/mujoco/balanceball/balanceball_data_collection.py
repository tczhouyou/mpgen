import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
from mp.vmp import VMP
from armar6_controllers.armar6_low_controller import TaskSpaceImpedanceController, TaskSpaceVelocityController
from armar6_controllers.armar6_high_controller import TaskSpaceVMPController, TaskSpacePositionVMPController
from balanceball_exp import Armar6BalanceBallExp, ENV_DIR, INIT_BALL_POS
import progressbar
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--num_data", dest="n_data", type="int", default=10)
parser.add_option("-m", "--env_path", dest="env_path", type="string", default="balanceball_exp.xml")
parser.add_option("-d", "--mp_dir", dest="mp_dir", type="string", default="balanceball_mpdata_v0")
parser.add_option("-c", "--vel", action="store_true", dest="is_vel", default=False)
parser.add_option("-p", "--draw", action="store_true", dest="is_draw", default=False)
parser.add_option("-r", "--raw_dir", dest="raw_dir", type="string", default="balanceball_mpdata_v0")
(options, args) = parser.parse_args(sys.argv)

env_dir = os.environ['MPGEN_DIR'] + ENV_DIR
env_path = env_dir + options.env_path
mp_dir = options.mp_dir


EXP_NUM = options.n_data
vmp = VMP(dim=2, elementary_type='minjerk')
env = Armar6BalanceBallExp(low_ctrl=TaskSpaceVelocityController, high_ctrl=TaskSpacePositionVMPController(vmp), env_path=env_path, isdraw=options.is_draw)
env.is_ball_pos_change = False

if not os.path.exists(options.raw_dir):
    try:
        os.makedirs(options.raw_dir)
    except OSError:
        raise

if not os.path.exists(options.mp_dir):
    try:
        os.makedirs(options.mp_dir)
    except OSError:
        raise

queries = []

for i in progressbar.progressbar(range(EXP_NUM)):
    is_failed = True
    while is_failed:
        start, _ = env.reset(init_ball_pos=INIT_BALL_POS)
        goal = start.copy()
        goal[:2] = INIT_BALL_POS.copy()[:2]

        dgx = np.random.uniform(low=-1.5, high=1.5)
        dgy = np.random.uniform(low=-2.3, high=-.9)
        env.high_ctrl.vmp.set_start_goal(start[:2], start[:2], dg=[0, 0])
        env.high_ctrl.target_quat = start[3:]
        env.high_ctrl.target_z = start[2]
        env.high_ctrl.desired_joints = np.array([0, -0.42, 0, 0, 2.0, 3.14, 0, 0])

        final_ball_pos, ts_traj, js_traj, is_error = env.run()
        if is_error or final_ball_pos[1] < 1.2:
            continue

        filename = options.raw_dir + '/hitball_' + str(i + 1) + '.csv'
        np.savetxt(filename, ts_traj, delimiter=',')

        query = np.array([final_ball_pos[0], final_ball_pos[1]])
        queries.append(query)

        qfile = options.mp_dir + '/hitball_queries.csv'
        np.savetxt(qfile, np.stack(queries), delimiter=',')
        is_failed=False
