import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')
from models.mdgan import cMDGAN
import sys
from optparse import OptionParser
import numpy as np
from sklearn.model_selection import train_test_split
from experiments.evaluate_exps import evaluate_docking, evaluate_docking_for_all_models
import matplotlib.pyplot as plt


def train_evaluate_mdgan_for_docking(mdgan, trqueries, trvmps, tdata, use_entropy=False, max_epochs=20000):
    if use_entropy:
        mdgan.entropy_ratio = 0.2
    else:
        mdgan.entropy_ratio = 0.0

    train_input = np.random.uniform(low=np.min(trqueries, axis=0), high=np.max(trqueries, axis=0),
                                    size=(10000, np.shape(trqueries)[1]))
    mdgan.create_network(num_real_data=np.shape(trqueries)[0])
    mdgan.init_train()
    mdgan.train(train_context=train_input, real_context=trqueries, real_response=trvmps, max_epochs=max_epochs,
                is_load=False, is_save=False)

    tqueries = tdata[:, 0:6]
    starts = tdata[:, 6:8]
    goals = tdata[:, 8:10]
    wout = mdgan.generate_multi(tqueries, 1)
    srate, _ = evaluate_docking(wout, tqueries, starts, goals)
    return srate

def run_mdgan_for_docking(nmodel=3, MAX_EXPNUM=1, nsamples=[1, 10, 30, 50], num_train_input=1000):
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

    nn_structure = {'generator': [40], 'discriminator': [20], 'lambda': [10], 'd_response': [40,5], 'd_context': [10,5]}
    mdgan = cMDGAN(n_comps=nmodel, context_dim=6, response_dim=knum, noise_dim=1, nn_structure=nn_structure)
    mdgan.gen_lrate = 0.0002
    mdgan.dis_lrate = 0.0002
    mdgan.entropy_ratio = 0.0

    csrates = np.zeros(shape=(MAX_EXPNUM, len(nsamples)))
    # generate training context
    train_input = np.random.uniform(low=np.min(queries, axis=0), high=np.max(queries, axis=0), size=(num_train_input, np.shape(queries)[1]))

    for expId in range(MAX_EXPNUM):
        trdata, tdata, trvmps, tvmps = train_test_split(data, vmps, test_size=0.3, random_state=rstates[expId])
        trdata, _, trvmps, _ = train_test_split(trdata, trvmps, test_size=0.3, random_state=rstates[expId])
        print("use {} data for training and {} data for testing".format(np.shape(trdata)[0], np.shape(tdata)[0]))
        print("======== Exp: {} ========".format(expId))
        trqueries = trdata[:,0:6]

        mdgan.create_network(num_real_data=np.shape(trdata)[0])
        mdgan.init_train()
        mdgan.train(train_context=train_input, real_context=trqueries, real_response=trvmps, max_epochs=150000, is_load=False, is_save=False)

        tqueries = tdata[:, 0:6]
        for i in range(len(nsamples)):
            wout = mdgan.generate_multi(tqueries, nsamples[i])
            starts = tdata[:, 6:8]
            goals = tdata[:, 8:10]
            srate, _ = evaluate_docking(wout, tqueries, starts, goals)
            csrates[expId, i] = srate

    srates = np.mean(csrates, axis=0)
    return srates


if __name__ == '__main__':
    nsamples = [1, 10, 30, 50, 70]
    MAX_EXPNUM = 5

    parser = OptionParser()
    parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=None)
    (options, args) = parser.parse_args(sys.argv)
    nmodel = 3
    if options.nmodel is not None:
        nmodel = options.nmodel

    srates = run_mdgan_for_docking(nmodel, MAX_EXPNUM, nsamples, num_train_input=10000)

    print(srates)
    x = np.arange(len(nsamples))
    fig, ax = plt.subplots()
    width = 0.35
    rects1 = ax.bar(x , srates, width, label='MDGAN')
    ax.set_ylabel('Success Rate - MDGAN for Docking')
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
    fig.tight_layout()
    plt.show()