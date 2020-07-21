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
from hitball_exp import Armar6HitBallExpV0, Armar6HitBallExpV1, ENV_DIR, INIT_BALL_POS
import progressbar
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--num_data", dest="n_data", type="int", default=10)
parser.add_option("-d", "--mp_dir", dest="mp_dir", type="string", default="hitball_mpdata")
parser.add_option("-p", "--draw", action="store_true", dest="is_draw", default=False)
parser.add_option("-r", "--raw_dir", dest="raw_dir", type="string", default="hitball_rawdata")
parser.add_option("-v", "--exp_version", dest="exp_version", type="string", default="v0")
(options, args) = parser.parse_args(sys.argv)
env_dir = os.environ['MPGEN_DIR'] + ENV_DIR
mp_dir = options.mp_dir


EXP_NUM = options.n_data
vmp = VMP(dim=2, elementary_type='minjerk')

if options.exp_version == "v1":
    env_path = env_dir + "hitball_exp_v1.xml"
    env = Armar6HitBallExpV1(low_ctrl=TaskSpaceVelocityController, high_ctrl=TaskSpacePositionVMPController(vmp),
                             env_path=env_path, isdraw=options.is_draw)
elif options.exp_version == "v0":
    env_path = env_dir + "hitball_exp_v0.xml"
    env = Armar6HitBallExpV0(low_ctrl=TaskSpaceImpedanceController, high_ctrl=TaskSpacePositionVMPController(vmp),
                             env_path=env_path, isdraw=options.is_draw)
elif options.exp_version == "v2":
    env_path = env_dir + "hitball_exp_v2.xml"
    env = Armar6HitBallExpV1(low_ctrl=TaskSpaceVelocityController, high_ctrl=TaskSpacePositionVMPController(vmp),
                             env_path=env_path, isdraw=options.is_draw)
else:
    raise Exception("Unknown Experiment Version")

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

        dgx = 1.0
        dgy = -1.8
        if np.random.uniform(low=0, high=1) < 0.5:
            dgx = np.random.uniform(low=-1.0, high=-0.5)
        else:
            dgx = np.random.uniform(low=0.5, high=1.0)
        #
        dgy = np.random.uniform(low=-1.8, high=-1.0)
        env.high_ctrl.vmp.set_start_goal(start[:2], goal[:2], dg=[dgx, dgy])
        env.high_ctrl.target_quat = start[3:]
        env.high_ctrl.target_posi = start[:3]
        env.high_ctrl.desired_joints = np.array([0, -0.2, 0, 0, 1.8, 3.14, 0, 0])

        final_ball_pos, ts_traj, js_traj, is_error = env.run()

        if is_error or final_ball_pos[1] < 2.2:
            continue

        filename = options.raw_dir + '/hitball_' + str(i + 1) + '.csv'
        np.savetxt(filename, ts_traj, delimiter=',')

        query = np.array([final_ball_pos[0], final_ball_pos[1]])
        queries.append(query)

        qfile = options.mp_dir + '/hitball_queries.csv'
        np.savetxt(qfile, np.stack(queries), delimiter=',')
        is_failed=False
