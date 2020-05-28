import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from models.mdnmp import MDNMP
from mp.vmp import VMP
import sys
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from experiments.obs_avoid_envs import ObsExp

parser = OptionParser()
parser.add_option("-s", "--save", action="store_true", dest="save_image", default=False)
(options, args) = parser.parse_args(sys.argv)

goals = np.loadtxt('data/modeCollapse_goals.csv', delimiter=',')
vmps = np.loadtxt('data/modeCollapse_weights.csv', delimiter=',')
weights = np.ones(shape=(np.shape(goals)[0], 1))

# generate exp
num_test = 20
obsExp = ObsExp(exp_name='goAroundObs')
nmodel = obsExp.nmodel
start = obsExp.start
colors = obsExp.colors
obs = obsExp.obs
testgoals = obsExp.gen_test(num_test=num_test)

knum = np.shape(vmps)[1]
dim = np.shape(goals)[1]
vmp = VMP(2, kernel_num=int(knum / dim), use_outrange_kernel=False)

train_goals = np.copy(goals)
train_ws = np.copy(vmps)
train_ispos = np.copy(weights)
testgoals = np.array([[0.1,1]])

MAX_EXP = 3
n_samples = 5
fig, axes = plt.subplots(nrows=MAX_EXP, ncols=2)
if MAX_EXP == 1:
    axes = np.expand_dims(axes, axis=0)

colors = [(0,0.43,0.73), (0, 0.73, 0.43)]
use_entropy_cost = [False, True]

nn_structure = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [40],
                'scale_layers': [40],
                'mixing_layers': [40]}
mdnmp = MDNMP(n_comps=nmodel, d_input=2, d_output=20, nn_structure=nn_structure)

for expID in range(MAX_EXP):
    print('=========== Exp: %1d ============' % (expID))
    for kid in range(len(use_entropy_cost)):
        if use_entropy_cost[kid]:
            mdnmp.lratio['entropy'] = 100.0
            print("===> train entropy MDN")
        else:
            mdnmp.lratio['entropy'] = 0.0
            print("===> train original MDN")

        mdnmp.build_mdn(learning_rate=0.00005)
        mdnmp.init_train()
        mdnmp.train(train_goals, train_ws, train_ispos, max_epochs=5000, is_load=False, is_save=False)
        wout, outdict = mdnmp.predict(testgoals, n_samples=n_samples)

        idx = outdict['compIDs'].astype(int)
        axes[expID, kid].clear()
        axes[expID, kid].set_ylim([-2, 2])
        axes[expID, kid].set_xlim([-2.5, 2.5])
        obs.plot(axes[expID, kid])

        for i in range(np.shape(vmps)[0]):
            vmp.set_weights(vmps[i, :])
            traj = vmp.roll(start, goals[i, :])
            axes[expID, kid].plot(goals[i,0], goals[i,1], 'ko')
            axes[expID, kid].plot(traj[:, 1], traj[:, 2], 'k-.', linewidth=2, alpha=0.7)

        print('the selected modes for 5 samples are {}'.format(idx))
        print('probability for different modes: {}'.format(outdict['mc']))
        for i in range(np.shape(wout)[0]):
            cgoals = testgoals[i, :]
            for j in range(n_samples):
                vmp.set_weights(wout[i,j,:])
                traj = vmp.roll(start, cgoals)
                axes[expID, kid].plot(traj[:, 1], traj[:, 2], '-', color=colors[idx[i,j]], linewidth=2)

            axes[expID, kid].plot(testgoals[i,0], testgoals[i,1], 'go')

        axes[expID, kid].set_axis_off()
        axes[expID,kid].set_aspect(1)
        plt.draw()


# add titles
cols = ['Original MDN', 'Entropy MDN']
for ax, col in zip(axes[0], cols):
    ax.set_title(col)

fig.tight_layout()
plt.show()

