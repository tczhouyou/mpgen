import os, inspect
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')
from models.mdnmp import MDNMP
from mp.vmp import VMP
import sys
from optparse import OptionParser
import numpy as np
from sklearn.model_selection import train_test_split
from experiments.obs_avoid_envs import ObsExp
from experiments.evaluate_exps import evaluate_docking, evaluate_docking_for_all_models
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int")

(options, args) = parser.parse_args(sys.argv)
queries = np.loadtxt('data/docking_queries.csv', delimiter=',')
vmps = np.loadtxt('data/docking_weights.csv', delimiter=',')
starts = np.loadtxt('data/docking_starts.csv', delimiter=',')
goals = np.loadtxt('data/docking_goals.csv', delimiter=',')

# clean the data
wtest = np.expand_dims(vmps, axis=1)
cc, successId = evaluate_docking(wtest, queries, starts,goals)
data = np.concatenate([queries, starts, goals], axis=1)
data = data[successId, :]
vmps = vmps[successId,:]

# prepare the experiment
obsExp = ObsExp(exp_name="Docking")
nmodel = options.nmodel
knum = np.shape(vmps)[1]
dim = obsExp.envDim

use_new_cost = [False, True]
mdn_names = ["Original MDN", "Entropy MDN"]
nsamples = [1, 10, 30, 50, 70]
vmp = VMP(2, kernel_num=10)
MAX_EXPNUM = 20
rstates = np.random.randint(0, 100, size=MAX_EXPNUM)
srates = np.zeros(shape=(2, len(nsamples)))

nn_structure = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [40],
                'scale_layers': [40],
                'mixing_layers': [40]}

for k in range(len(use_new_cost)):
    mdnmp = MDNMP(n_comps=nmodel, d_input=6, d_output=knum, nn_structure=nn_structure, scaling=1)

    if use_new_cost[k]:
        mdnmp.lratio['entropy'] = 100
    else:
        mdnmp.lratio['entropy'] = 0

    csrates = np.zeros(shape=(MAX_EXPNUM,len(nsamples)))

    for expId in range(MAX_EXPNUM):
        trdata, tdata, trvmps, tvmps = train_test_split(data, vmps, test_size=0.3, random_state=rstates[expId])
        trdata, _, trvmps, _ = train_test_split(trdata, trvmps, test_size=0.8, random_state=rstates[expId])
        print("use {} data for training and {} data for testing".format(np.shape(trdata)[0], np.shape(tdata)[0]))
        print("======== Exp: {} with {} ========".format(expId, mdn_names[k]))

        weights = np.ones(shape=(np.shape(trvmps)[0], 1))
        train_weights = np.copy(weights)

        trqueries = trdata[:,0:6]
        mdnmp.build_mdn(learning_rate=0.0001)
        mdnmp.init_train()
        mdnmp.train(trqueries, trvmps, train_weights, max_epochs=5000, is_load=False, is_save=False)
        tqueries = tdata[:, 0:6]

        for i in range(len(nsamples)):
            wout, _ = mdnmp.predict(tqueries, nsamples[i])
            starts = tdata[:, 6:8]
            goals = tdata[:, 8:10]
            srate, _ = evaluate_docking(wout, tqueries, starts, goals)
            csrates[expId, i] = srate

    srates[k,:] = np.mean(csrates, axis=0)


# plot
x = np.arange(len(nsamples))
fig, ax = plt.subplots()
width = 0.35
rects1 = ax.bar(x- width/2, srates[0,:], width, label=mdn_names[0])
rects2 = ax.bar(x + width/2, srates[1,:], width, label=mdn_names[1])
ax.set_ylabel('Success rate')
ax.set_title('Sample Number')
ax.set_xticks(x)
ax.set_xticklabels(nsamples)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % (height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()