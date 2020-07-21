import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
from mp.qvmp import QVMP
from armar6_controllers.armar6_low_controller import TaskSpaceImpedanceController, TaskSpaceVelocityController
from armar6_controllers.armar6_high_controller import TaskSpaceVMPController, TaskSpacePositionVMPController
from balanceball_exp import Armar6BalanceBallExp, ENV_DIR, INIT_BALL_POS, INIT_JOINT_POS
import progressbar
from optparse import OptionParser
from math_tools.Quaternion import Quaternion

parser = OptionParser()
parser.add_option("-n", "--num_data", dest="n_data", type="int", default=10)
parser.add_option("-d", "--mp_dir", dest="mp_dir", type="string", default="balanceball_mpdata")
parser.add_option("-p", "--draw", action="store_true", dest="is_draw", default=False)
parser.add_option("-r", "--raw_dir", dest="raw_dir", type="string", default="balanceball_rawdata")
parser.add_option("-e", "--env", dest="env_file", type="string", default="balanceball_exp_v1.xml")

(options, args) = parser.parse_args(sys.argv)

env_dir = os.environ['MPGEN_DIR'] + ENV_DIR
env_path = env_dir + options.env_file
mp_dir = options.mp_dir


EXP_NUM = options.n_data
qvmp = QVMP(kernel_num=10, elementary_type='minjerk')
env = Armar6BalanceBallExp(low_ctrl=TaskSpaceVelocityController, high_ctrl=TaskSpacePositionVMPController(qvmp=qvmp), env_path=env_path, isdraw=options.is_draw)
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
        q0 = start[3:]
        q1 = start[3:]

        ax1 = np.array([0,1,0])
        if np.random.uniform(0, 1) < 0.5:
            ang1 = np.random.uniform(low=-30 * np.pi/180, high=-20 * np.pi/180)
        else:
            ang1 = np.random.uniform(low=20 * np.pi/180, high=30 * np.pi/180)

        qrot1 = Quaternion.from_axis_angle(ax1, ang1)

        ax2 = np.array([1,0,0])
        ang2 = np.random.uniform(low=-30 * np.pi/180, high=-20 * np.pi/180)
        qrot2 = Quaternion.from_axis_angle(ax2, ang2)

        ax3 = np.random.uniform(low=-1, high=1, size=3)
        if np.random.uniform(0, 1) < 0.5:
            ang3 = np.random.uniform(low=-30 * np.pi / 180, high=-20 * np.pi / 180)
        else:
            ang3 = np.random.uniform(low=20 * np.pi / 180, high=30 * np.pi / 180)

        ax3[2] = 0
        ax3 = ax3 / np.linalg.norm(ax3)
        qrot3 = Quaternion.from_axis_angle(ax3, ang3)

        qv1 = Quaternion.qmulti(qrot1, q0)
        qv2 = Quaternion.qmulti(qrot2, q0)
        qv3 = Quaternion.qmulti(qrot3, q0)

        qs = [q0, qv1, qv2, qv3, q1]
        ts = np.array([0, 0.25, 0.5, 0.75, 1])

        env.high_ctrl.qvmp.set_qs(ts, qs)
        env.high_ctrl.target_quat = start[3:]
        env.high_ctrl.target_posi = start[:3]
        env.high_ctrl.desired_joints = INIT_JOINT_POS

        final_ball_pos, qtraj, jtraj, is_error = env.run()

        if is_error:
            continue

        filename = options.raw_dir + '/balanceball_' + str(i + 1) + '.csv'
        np.savetxt(filename, qtraj, delimiter=',')

        query = np.array([final_ball_pos[0], final_ball_pos[1]])
        queries.append(query)
        is_failed=False

qfile = options.mp_dir + '/balanceball_queries.csv'
np.savetxt(qfile, queries, delimiter=',')
