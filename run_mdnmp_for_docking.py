import os, inspect
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')
from models.mdnmp import MDNMP
import sys
from optparse import OptionParser
import numpy as np
from sklearn.model_selection import train_test_split
from experiments.evaluate_exps import evaluate_docking, evaluate_docking_for_all_models
import matplotlib.pyplot as plt


def train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata, use_entropy=True, max_epochs=20000):
    if use_entropy:
        mdnmp.lratio['entropy'] = 200
    else:
        mdnmp.lratio['entropy'] = 0

    weights = np.ones(shape=(np.shape(trvmps)[0], 1))
    train_weights = np.copy(weights)
    mdnmp.build_mdn(learning_rate=0.00005)
    mdnmp.init_train()
    mdnmp.train(trqueries, trvmps, train_weights, max_epochs=max_epochs, is_load=False, is_save=False)

    tqueries = tdata[:, 0:6]
    wout, _ = mdnmp.predict(tqueries, 1)
    starts = tdata[:, 6:8]
    goals = tdata[:, 8:10]
    srate, _ = evaluate_docking(wout, tqueries, starts, goals)
    return srate

def run_mdnmp_for_docking(nmodel=3, MAX_EXPNUM=20, use_entropy_cost=[False, True], model_names=["Original MDN", "Entropy MDN"], nsamples=[1, 10, 30, 50, 70]):
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
    knum = np.shape(vmps)[1]

    rstates = np.random.randint(0, 100, size=MAX_EXPNUM)
    srates = {}

    nn_structure = {'d_feat': 20,
                    'feat_layers': [40],
                    'mean_layers': [60],
                    'scale_layers': [60],
                    'mixing_layers': [60]}

    for k in range(len(use_entropy_cost)):
        mdnmp = MDNMP(n_comps=nmodel, d_input=6, d_output=knum, nn_structure=nn_structure, scaling=1)

        if use_entropy_cost[k]:
            mdnmp.lratio['entropy'] = 200
        else:
            mdnmp.lratio['entropy'] = 0

        csrates = np.zeros(shape=(MAX_EXPNUM,len(nsamples)))

        for expId in range(MAX_EXPNUM):
            trdata, tdata, trvmps, tvmps = train_test_split(data, vmps, test_size=0.3, random_state=rstates[expId])
            trdata, _, trvmps, _ = train_test_split(trdata, trvmps, test_size=0.3, random_state=rstates[expId])
            print("use {} data for training and {} data for testing".format(np.shape(trdata)[0], np.shape(tdata)[0]))
            print("======== Exp: {} with {} ========".format(expId, model_names[k]))

            weights = np.ones(shape=(np.shape(trvmps)[0], 1))
            train_weights = np.copy(weights)

            trqueries = trdata[:,0:6]
            mdnmp.build_mdn(learning_rate=0.00005)
            mdnmp.init_train()
            mdnmp.train(trqueries, trvmps, train_weights, max_epochs=20000, is_load=False, is_save=False)
            tqueries = tdata[:, 0:6]

            for i in range(len(nsamples)):
                wout, _ = mdnmp.predict(tqueries, nsamples[i])
                starts = tdata[:, 6:8]
                goals = tdata[:, 8:10]
                srate, _ = evaluate_docking(wout, tqueries, starts, goals)
                csrates[expId, i] = srate

        srates[model_names[k]] = np.mean(csrates, axis=0)

    return srates


if __name__ == '__main__':
    use_entropy_cost = [False, True]
    model_names = ["Original MDN", "Entropy MDN"]
    nsamples = [1, 10, 30, 50, 70]
    MAX_EXPNUM = 5

    parser = OptionParser()
    parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=None)
    (options, args) = parser.parse_args(sys.argv)
    nmodel = 3
    if options.nmodel is not None:
        nmodel = options.nmodel

    srates = run_mdnmp_for_docking(nmodel, MAX_EXPNUM, use_entropy_cost, model_names, nsamples)

    x = np.arange(len(nsamples))
    fig, ax = plt.subplots()
    width = 0.35
    rects1 = ax.bar(x - width / 2, srates[model_names[0]], width, label=model_names[0])
    rects2 = ax.bar(x + width / 2, srates[model_names[1]], width, label=model_names[1])
    ax.set_ylabel('Success Rate - MDNMP for Docking')
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