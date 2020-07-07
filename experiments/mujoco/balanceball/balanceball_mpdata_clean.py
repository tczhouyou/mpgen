import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
from mp.qvmp import QVMP
from armar6_controllers.armar6_low_controller import TaskSpaceImpedanceController, TaskSpaceVelocityController, get_actuator_data
from armar6_controllers.armar6_high_controller import TaskSpacePositionVMPController
from balanceball_exp import Armar6BalanceBallExp, ENV_DIR, INIT_BALL_POS, INIT_JOINT_POS
from optparse import OptionParser
from math_tools.Quaternion import Quaternion
env_dir = os.environ['MPGEN_DIR'] + ENV_DIR

parser = OptionParser()
parser.add_option("-d", "--mp_dir", dest="mp_dir", type="string", default="balanceball_mpdata")
parser.add_option("-p", "--draw", action="store_true", dest="is_draw", default=False)

(options, args) = parser.parse_args(sys.argv)

env_path = env_dir + "balanceball_exp.xml"
mp_dir = options.mp_dir

qvmp = QVMP(kernel_num=10, elementary_type='minjerk')
ostarts = np.loadtxt(mp_dir + '/balanceball_starts.csv', delimiter=',')
ogoals = np.loadtxt(mp_dir + '/balanceball_goals.csv', delimiter=',')
oqueries = np.loadtxt(mp_dir + '/balanceball_queries.csv', delimiter=',')
oweights = np.loadtxt(mp_dir + '/balanceball_weights.csv', delimiter=',')

env = Armar6BalanceBallExp(low_ctrl=TaskSpaceVelocityController, high_ctrl=TaskSpacePositionVMPController(qvmp=qvmp), env_path=env_path, isdraw=options.is_draw)

success_rate = 0
num_exp = np.shape(oqueries)[0]

nstarts = []
ngoals = []
nqueries = []
nweights = []

for i in range(num_exp):
    print('current exp: {}'.format(i))
    query = oqueries[i,:]
    weight = oweights[i,:]
    start = ostarts[i,:]
    goal = ogoals[i,:]
    st, _ = env.reset(init_ball_pos=INIT_BALL_POS, target_pos=query)

    env.high_ctrl.target_quat = st[3:]
    env.high_ctrl.target_posi = st[:3]
    env.high_ctrl.desired_joints = INIT_JOINT_POS

    qvmp.set_weights(weight)
    qvmp.set_start_goal(start, start)
    final_ball_pos, qtraj, jtraj, is_error = env.run()

    if is_error:
        print('error .... ')
        print('final_ball_pos: {}'.format(final_ball_pos))
        is_error = True

    if not is_error and np.linalg.norm(query - final_ball_pos[:2]) < 0.02:
        success_rate = success_rate + 1

    if not is_error:
        nstarts.append(start)
        ngoals.append(start)
        nqueries.append(final_ball_pos[:2])
        nweights.append(weight)

    print('success_rate: {}'.format(success_rate / (i+1)))

np.savetxt(mp_dir + '/balanceball_queries.csv', np.stack(nqueries), delimiter=',')
np.savetxt(mp_dir + '/balanceball_weights.csv', np.stack(nweights), delimiter=',')
np.savetxt(mp_dir + '/balanceball_starts.csv', np.stack(nstarts), delimiter=',')
np.savetxt(mp_dir + '/balanceball_goals.csv', np.stack(ngoals), delimiter=',')
