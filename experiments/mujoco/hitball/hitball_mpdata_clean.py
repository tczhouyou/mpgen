import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import numpy as np
from mp.vmp import VMP
from armar6_controllers.armar6_low_controller import TaskSpaceImpedanceController, TaskSpaceVelocityController, get_actuator_data
from armar6_controllers.armar6_high_controller import TaskSpacePositionVMPController
from hitball_exp import Armar6HitBallExpV0, Armar6HitBallExpV1, ENV_DIR, INIT_BALL_POS
from optparse import OptionParser


env_dir = os.environ['MPGEN_DIR'] + ENV_DIR

parser = OptionParser()
parser.add_option("-d", "--mp_dir", dest="mp_dir", type="string", default="hitball_mpdata_v0")
parser.add_option("-p", "--draw", action="store_true", dest="is_draw", default=False)
parser.add_option("-v", "--version", dest="exp_version", type="string", default="v1")

(options, args) = parser.parse_args(sys.argv)
mp_dir = options.mp_dir

vmp = VMP(dim=2, kernel_num=10, use_outrange_kernel=False)

ostarts = np.loadtxt(mp_dir + '/hitball_starts.csv', delimiter=',')
ogoals = np.loadtxt(mp_dir + '/hitball_goals.csv', delimiter=',')
oqueries = np.loadtxt(mp_dir + '/hitball_queries.csv', delimiter=',')
oweights = np.loadtxt(mp_dir + '/hitball_weights.csv', delimiter=',')

if options.exp_version == "v1":
    env_path = env_dir + "hitball_exp_v1.xml"
    env = Armar6HitBallExpV1(low_ctrl=TaskSpaceVelocityController, high_ctrl=TaskSpacePositionVMPController(vmp), env_path=env_path, isdraw=options.is_draw)
elif options.exp_version == "v0":
    env_path = env_dir + "hitball_exp_v0.xml"
    env = Armar6HitBallExpV0(low_ctrl=TaskSpaceImpedanceController, high_ctrl=TaskSpacePositionVMPController(vmp), env_path=env_path, isdraw=options.is_draw)
elif options.exp_version == "v2":
    env_path = env_dir + "hitball_exp_v2.xml"
    env = Armar6HitBallExpV1(low_ctrl=TaskSpaceVelocityController, high_ctrl=TaskSpacePositionVMPController(vmp), env_path=env_path, isdraw=options.is_draw)



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
    env.high_ctrl.desired_joints = np.array([0, -0.2, 0, 0, 1.8, 3.14, 0, 0])

    vmp.set_weights(weight)
    vmp.set_start_goal(start, INIT_BALL_POS[:2])
    final_ball_pos, _, _, is_error = env.run()
    if is_error or final_ball_pos[1] < 2.0:
        print('error .... ')
        print('final_ball_pos: {}'.format(final_ball_pos[1]))
        is_error = True

    if not is_error and np.linalg.norm(query - final_ball_pos[:2]) < 0.08:
        success_rate = success_rate + 1

    if not is_error:
        nstarts.append(start)
        ngoals.append(INIT_BALL_POS[:2])
        nqueries.append(final_ball_pos[:2])
        nweights.append(weight)

    print('success_rate: {}'.format(success_rate / (i+1)))

np.savetxt(mp_dir + '/hitball_queries.csv', np.stack(nqueries), delimiter=',')
np.savetxt(mp_dir + '/hitball_weights.csv', np.stack(nweights), delimiter=',')
np.savetxt(mp_dir + '/hitball_starts.csv', np.stack(nstarts), delimiter=',')
np.savetxt(mp_dir + '/hitball_goals.csv', np.stack(ngoals), delimiter=',')
