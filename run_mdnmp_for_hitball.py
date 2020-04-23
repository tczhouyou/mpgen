import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, './experiments/mujoco')

import numpy as np
from models.mdnmp import MDNMP
from mp.vmp import VMP

from sklearn.model_selection import train_test_split
from experiments.mujoco.hitball.hitball_exp import evaluate_hitball
from optparse import OptionParser


def run_mdnmp_for_hitball(nmodel=3, MAX_EXPNUM=20, use_entropy_cost=[False, True], model_names=["Original MDN", "Entropy MDN"], nsamples=[1, 10, 30, 50, 70]):
    # prepare data
    data_dir = 'experiments/mujoco/hitball/hitball_mpdata'
    queries = np.loadtxt(data_dir + '/hitball_queries.csv', delimiter=',')
    vmps = np.loadtxt(data_dir + '/hitball_weights.csv', delimiter=',')
    starts = np.loadtxt(data_dir + '/hitball_starts.csv', delimiter=',')
    goals = np.loadtxt(data_dir + '/hitball_goals.csv', delimiter=',')

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

    mp = VMP(dim=2, kernel_num=10)
    mdnmp = MDNMP(n_comps=nmodel, d_input=d_input, d_output=d_output, nn_structure=nn_structure)

    rstates = np.random.randint(0, 100, size=MAX_EXPNUM)
    n_test = 100

    srates = {}
    allres = np.zeros(shape=(len(model_names), MAX_EXPNUM, len(nsamples)))
    for modelId in range(len(model_names)):
        if use_entropy_cost[modelId]:
            mdnmp.lratio['entropy'] = 1000
        else:
            mdnmp.lratio['entropy'] = 0

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
            tstarts = tdata[:n_test, d_input:d_input+2]
            tgoals = tdata[:n_test, d_input+2:]

            for sampleId in range(len(nsamples)):
                wout, _ = mdnmp.predict(tqueries, nsamples[sampleId])
                srate = evaluate_hitball(mp, wout, tqueries, tstarts, tgoals)
                csrates[expId, sampleId] = srate
                allres[modelId, expId, sampleId] = srate

        srates[model_names[modelId]] = np.mean(csrates, axis=0)

    return srates, allres

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=None)
    (options, args) = parser.parse_args(sys.argv)
    nmodel = 3
    if options.nmodel is not None:
        nmodel = options.nmodel


    use_entropy_cost = [False, True]
    model_names = ["Original MDN", "Entropy MDN"]
    MAX_EXPNUM = 5
    nsamples = [1, 10, 30, 50]

    srates, allres = run_mdnmp_for_hitball(nmodel, MAX_EXPNUM, use_entropy_cost, model_names, nsamples)

    res_file = open("result_mdnmp", 'a')
    for modelId in range(len(model_names)):
        res_file.write(model_names[modelId] + '\n')
        np.savetxt(res_file, np.array(allres[modelId,:,:]), delimiter=',')
        np.savetxt(res_file, np.array(srates[model_names[modelId]]), delimiter=',')

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
