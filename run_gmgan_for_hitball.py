import os, inspect, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, './experiments/mujoco')

import numpy as np
from mp.vmp import VMP

from experiments.mujoco.hitball.hitball_exp import evaluate_hitball

def train_evaluate_gmgan_for_hitball(gmgan, trqueries, trvmps, tdata, use_entropy=False, max_epochs=20000, sup_max_epoch=0):
    if use_entropy:
        gmgan.entropy_ratio = 1
    else:
        gmgan.entropy_ratio = 0

    gmgan.lratio['entropy'] = 1000
    gmgan.lratio['adv_cost'] = 0
    gmgan.gen_sup_lrate = 0.00003
    gmgan.gen_adv_lrate = 0.00003
    gmgan.dis_lrate = 0.0002
    gmgan.sup_max_epoch = sup_max_epoch

    train_input = np.random.uniform(low=np.min(trqueries, axis=0), high=np.max(trqueries, axis=0), size=(100, np.shape(trqueries)[1]))

    gmgan.create_network()
    gmgan.init_train()
    gmgan.train(train_context=train_input, real_context=trqueries, real_response=trvmps, max_epochs=max_epochs, is_load=False, is_save=False)
    mp = VMP(dim=2, kernel_num=10)

    tqueries = tdata[:100, 0:2]
    starts = tdata[:100, 2:4]
    goals = tdata[:100, 4:6]
    wout = gmgan.generate(tqueries, 8000)
    srate = evaluate_hitball(mp, wout, tqueries, starts, goals)
    return srate



