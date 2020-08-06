import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../experiments/mujoco')

import numpy as np
from models.baselines import MultiDimSkRegressor, sample_baseline

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR

from mp.qvmp import QVMP

from experiments.mujoco.armar6_controllers.armar6_low_controller import TaskSpaceVelocityController, TaskSpaceImpedanceController
from experiments.mujoco.armar6_controllers.armar6_high_controller import TaskSpacePositionVMPController
from experiments.mujoco.balanceball.balanceball_exp import evaluate_balanceball, ENV_DIR

_svr = SVR(gamma='scale', C=1.0, epsilon=0.1)
svr = MultiDimSkRegressor(_svr)

_gpr = GaussianProcessRegressor()
gpr = MultiDimSkRegressor(_gpr)
possible_models = {'SVR': svr, 'GPR': gpr, 'Uniform': None}

def train_evaluate_baseline_for_balanceball(model_name, trqueries, trvmps, tdata, sample_num=1,
                                            env_file="balanceball_exp_v1.xml", isdraw=False, num_test=100):

    model = possible_models[model_name]
    if model is not None:
        model.fit(trqueries, trvmps)

    mp = QVMP(kernel_num=10)

    if num_test > np.shape(tdata)[0]:
        num_test = np.shape(tdata)[0]-1

    tqueries = tdata[:num_test, 0:2]
    tstarts = tdata[:num_test, 2:6]
    tgoals = tdata[:num_test, 6:10]
    wmeans = model.predict(tqueries)

    if model is not None:
        wouts = sample_baseline(wmeans, sample_num)
    else:
        wouts = np.random.uniform(low=trvmps.min(), high=trvmps.max(),
                                  size=(100, sample_num, np.shape(trvmps)[1]))

    srate = evaluate_balanceball(wouts, tqueries, tstarts, tgoals, low_ctrl=TaskSpaceVelocityController,
                                 high_ctrl=TaskSpacePositionVMPController(qvmp=mp),
                                 env_path=ENV_DIR + env_file, isdraw=isdraw)

    return srate


