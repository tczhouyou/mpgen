import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../experiments/mujoco')

import numpy as np
from mp.qvmp import QVMP
from armar6_controllers.armar6_low_controller import TaskSpaceImpedanceController, TaskSpaceVelocityController, get_actuator_data
from armar6_controllers.armar6_high_controller import TaskSpacePositionVMPController
from balanceball.balanceball_exp import Armar6BalanceBallExp, ENV_DIR, INIT_BALL_POS, INIT_JOINT_POS
from optparse import OptionParser
env_dir = os.environ['MPGEN_DIR'] + ENV_DIR

parser = OptionParser()
parser.add_option("-q", "--query_file", dest="query_file", type="string", default="balanceball_queries.csv")
parser.add_option("-w", "--weights_file", dest="weights_file",type="string",  default="balanceball_weights.csv")

(options, args) = parser.parse_args(sys.argv)

env_path = env_dir + "balanceball_exp.xml"
qvmp = QVMP(kernel_num=10, elementary_type='minjerk')
oqueries = np.loadtxt(options.query_file, delimiter=',')
oweights = np.loadtxt(options.weights_file, delimiter=',')

env = Armar6BalanceBallExp(low_ctrl=TaskSpaceVelocityController, high_ctrl=TaskSpacePositionVMPController(qvmp=qvmp), env_path=env_path, isdraw=True)

success_rate = 0
num_exp = np.shape(oqueries)[0]

for i in range(num_exp):
    print('current exp: {}'.format(i))
    query = oqueries[i,:]
    weight = oweights[i,:]
    start = np.array([5.064471645756087881e-01,-5.013931619887760371e-01,-4.863804606063221736e-01,-5.055197465427094805e-01])
    goal = np.array([5.064471645756087881e-01,-5.013931619887760371e-01,-4.863804606063221736e-01,-5.055197465427094805e-01])
    st, _ = env.reset(init_ball_pos=INIT_BALL_POS, target_pos=query)

    env.high_ctrl.target_quat = st[3:]
    env.high_ctrl.target_posi = st[:3]
    env.high_ctrl.desired_joints = INIT_JOINT_POS

    qvmp.set_weights(weight)
    qvmp.set_start_goal(start, start)
    env.run()

