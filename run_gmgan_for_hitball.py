import os, inspect, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, './experiments/mujoco')

import numpy as np
from mp.vmp import VMP

from experiments.mujoco.armar6_controllers.armar6_low_controller import TaskSpaceVelocityController, TaskSpaceImpedanceController
from experiments.mujoco.armar6_controllers.armar6_high_controller import TaskSpacePositionVMPController
from experiments.mujoco.hitball.hitball_exp import evaluate_hitball, ENV_DIR, EXP_DIR, Armar6HitBallExpV0, Armar6HitBallExpV1

def train_evaluate_gmgan_for_hitball(gmgan, trqueries, trvmps, tdata, use_entropy=False, max_epochs=20000, sup_max_epoch=0,
                                     isvel=True, env_file="hitball_exp_v1.xml", sample_num=1, isdraw=False, num_test=100,
                                     g_lrate=0.002, d_lrate=0.002, EXP=Armar6HitBallExpV1):
    if use_entropy:
        gmgan.entropy_ratio = 1
    else:
        gmgan.entropy_ratio = 0

    gmgan.lratio['entropy'] = 10
    gmgan.lratio['adv_cost'] = 100
    gmgan.gen_sup_lrate = g_lrate
    gmgan.gen_adv_lrate = g_lrate
    gmgan.dis_lrate = d_lrate
    gmgan.sup_max_epoch = sup_max_epoch

    train_input = np.random.uniform(low=np.min(trqueries, axis=0), high=np.max(trqueries, axis=0), size=(10000, np.shape(trqueries)[1]))

    gmgan.create_network()
    gmgan.init_train()
    gmgan.train(train_context=train_input, real_context=trqueries, real_response=trvmps, max_epochs=max_epochs, is_load=False, is_save=False)
    mp = VMP(dim=2, kernel_num=10)

    if num_test > np.shape(tdata)[0]:
        num_test = np.shape(tdata)[0]-1

    tqueries = tdata[:num_test, 0:2]
    starts = tdata[:num_test, 2:4]
    goals = tdata[:num_test, 4:6]
    wout = gmgan.generate(tqueries, 8000, sample_num)

    if isvel:
        srate = evaluate_hitball(wout, tqueries, starts, goals,
                                 low_ctrl=TaskSpaceVelocityController,
                                 high_ctrl=TaskSpacePositionVMPController(mp),
                                 env_path=ENV_DIR+env_file, isdraw=isdraw, EXP=EXP)
    else:
        srate = evaluate_hitball(wout, tqueries, starts, goals,
                                 low_ctrl=TaskSpaceImpedanceController,
                                 high_ctrl=TaskSpacePositionVMPController(mp),
                                 env_path=ENV_DIR+env_file, isdraw=isdraw, EXP=EXP)

    return srate



