import os, inspect, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, './experiments/mujoco')

import numpy as np
from models.mdnmp import MDNMP
from mp.qvmp import QVMP

from sklearn.model_selection import train_test_split
from optparse import OptionParser

from experiments.mujoco.balanceball.balanceball_exp import evaluate_balanceball, ENV_DIR, EXP_DIR
from experiments.mujoco.balanceball.balanceball_exp import Armar6BalanceBallExp
from experiments.mujoco.armar6_controllers.armar6_low_controller import TaskSpaceVelocityController
from experiments.mujoco.armar6_controllers.armar6_high_controller import TaskSpacePositionVMPController

import tensorflow as tf
if tf.__version__ < '2.0.0':
    import tflearn
    VAR_INIT = tflearn.initializations.uniform(minval=-.1, maxval=.1, seed=42)
else:
    from tensorflow.keras import initializers
    VAR_INIT = initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=42)


ENV_FILE = "balanceball_exp.xml"

def train_evaluate_mdnmp_for_balanceball(mdnmp, trqueries, trvmps, tdata, use_entropy=False, max_epochs=20000, sample_num=1,
                                         isdraw=False, num_test=100, learning_rate=0.002, env_file="balanceball_exp.xml", EXP=Armar6BalanceBallExp):
    if use_entropy:
        mdnmp.lratio['mce'] = 10
    else:
        mdnmp.lratio['mce'] = 0

    weights = np.ones(shape=(np.shape(trvmps)[0], 1))
    train_weights = np.copy(weights)
    mdnmp.build_mdn(learning_rate=learning_rate)
    mdnmp.init_train()
    mdnmp.train(trqueries, trvmps, train_weights, max_epochs=max_epochs, is_load=False, is_save=False)
    mp = QVMP(kernel_num=10)

    if num_test > np.shape(tdata)[0]:
        num_test = np.shape(tdata)[0]-1

    tqueries = tdata[:num_test, 0:2]
    starts = tdata[:num_test, 2:4]
    goals = tdata[:num_test, 4:6]
    wout, _ = mdnmp.predict(tqueries, sample_num)

    srate = evaluate_balanceball(wout, tqueries, starts, goals,
                             low_ctrl=TaskSpaceVelocityController,
                             high_ctrl=TaskSpacePositionVMPController(qvmp=mp),
                             env_path=ENV_DIR+env_file, isdraw=isdraw)


    return srate


def run_mdnmp_for_balanceball(nmodel=3, MAX_EXPNUM=20, use_entropy_cost=[False, True],
                          model_names=["Original MDN", "Entropy MDN"], nsamples=[10, 30, 50],
                          env_file="balanceball_exp.xml", data_dir="balanceball_mpdata", isdraw=False):
    # prepare data
    data_dir = os.environ['MPGEN_DIR'] + EXP_DIR + data_dir
    queries = np.loadtxt(data_dir + '/balanceball_queries.csv', delimiter=',')
    vmps = np.loadtxt(data_dir + '/balanceball_weights.csv', delimiter=',')
    starts = np.loadtxt(data_dir + '/balanceball_starts.csv', delimiter=',')
    goals = np.loadtxt(data_dir + '/balanceball_goals.csv', delimiter=',')

    if np.shape(queries)[-1] == np.shape(goals)[0]:
        queries = np.expand_dims(queries, axis=-1)

    inputs = np.concatenate([queries, starts, goals], axis=1)
    # prepare model
    nn_structure = {'d_feat': 20,
                    'feat_layers': [40],
                    'mean_layers': [60],
                    'scale_layers': [60],
                    'mixing_layers': [10]}

    d_input = np.shape(queries)[-1]
    d_output = np.shape(vmps)[1]

    mp = QVMP(kernel_num=10)
    mdnmp = MDNMP(n_comps=nmodel, d_input=d_input, d_output=d_output, nn_structure=nn_structure, var_init=VAR_INIT)

    rstates = np.random.randint(0, 100, size=MAX_EXPNUM)
    n_test = 100

    srates = {}
    allres = np.zeros(shape=(len(model_names), MAX_EXPNUM, len(nsamples)))
    for modelId in range(len(model_names)):
        if use_entropy_cost[modelId]:
            mdnmp.lratio['mce'] = 20
        else:
            mdnmp.lratio['mce'] = 0

        csrates = np.zeros(shape=(MAX_EXPNUM,len(nsamples)))
        for expId in range(MAX_EXPNUM):
            mdnmp.build_mdn(learning_rate=0.0001)
            mdnmp.init_train()

            trdata, tdata, trvmps, tvmps = train_test_split(inputs, vmps, test_size=0.9, random_state=rstates[expId])
            print("use {} data for training and {} data for testing".format(np.shape(trdata)[0], np.shape(tdata)[0]))
            print("======== Exp: {} with {} ========".format(expId, model_names[modelId]))

            is_pos = np.ones(shape=(np.shape(trvmps)[0], 1))
            trqueries = trdata[:,0:d_input]
            mdnmp.train(trqueries, trvmps, is_pos, max_epochs=10000, is_load=False, is_save=False)

            tqueries = tdata[:n_test, 0:d_input]
            starts = tdata[:n_test, d_input:d_input+4]
            goals = tdata[:n_test, d_input+4:]

            

            for sampleId in range(len(nsamples)):
                wout, _ = mdnmp.predict(tqueries, nsamples[sampleId])
                srate = evaluate_balanceball(wout, tqueries, starts, goals,
                                             low_ctrl=TaskSpaceVelocityController,
                                             high_ctrl=TaskSpacePositionVMPController(qvmp=mp),
                                             env_path=ENV_DIR + env_file, isdraw=isdraw)

                csrates[expId, sampleId] = srate
                allres[modelId, expId, sampleId] = srate

        srates[model_names[modelId]] = np.mean(csrates, axis=0)

    return srates, allres

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=3)
    parser.add_option("--env_file", dest="env_file", type="string", default="balanceball_exp.xml")
    parser.add_option("--data_dir", dest="data_dir", type="string", default="balanceball_mpdata")
    parser.add_option("--file", dest="fname", type="string", default="res_mdnmp_balanceball")
    (options, args) = parser.parse_args(sys.argv)
    nmodel = options.nmodel

    use_entropy_cost = [True]
    model_names = ["Entropy MDN"]
    MAX_EXPNUM = 10
    nsamples = [10, 30, 50]

    srates, allres = run_mdnmp_for_balanceball(nmodel, MAX_EXPNUM, use_entropy_cost, model_names, nsamples,
                                           env_file=options.env_file,
                                           data_dir=options.data_dir, isdraw=True)

    res_file = open(options.fname, 'a')
    for modelId in range(len(model_names)):
        res_file.write(model_names[modelId] + '\n')
        np.savetxt(res_file, np.array(allres[modelId,:,:]), delimiter=',')

    res_file.close()
    # import matplotlib.pyplot as plt
    # x = np.arange(len(nsamples))
    # fig, ax = plt.subplots()
    # width = 0.35
    # rects1 = ax.bar(x - width / 2, srates[model_names[0]], width, label=model_names[0])
    # rects2 = ax.bar(x + width / 2, srates[model_names[1]], width, label=model_names[1])
    # ax.set_ylabel('Success Rate - MDNMP for Hitball')
    # ax.set_title('Sample Number')
    # ax.set_xticks(x)
    # ax.set_xticklabels(nsamples)
    # ax.legend()
    #
    #
    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('%.2f' % (height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    #
    #
    # autolabel(rects1)
    # autolabel(rects2)
    # fig.tight_layout()
    # plt.show()
