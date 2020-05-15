import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, './experiments/mujoco')


import numpy as np
from models.baselines import MultiDimSkRegressor, sample_baseline

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from experiments.mujoco.hitball.hitball_exp import evaluate_hitball
from mp.vmp import VMP


def run_baselines_for_hitball(MAX_EXPNUM=20, nsamples=[1, 10, 30, 50], model_names=['Uniform', 'SVR', 'GPR']):
    data_dir = 'experiments/mujoco/hitball/hitball_mpdata'
    queries = np.loadtxt(data_dir + '/hitball_queries.csv', delimiter=',')
    vmps = np.loadtxt(data_dir + '/hitball_weights.csv', delimiter=',')
    starts = np.loadtxt(data_dir + '/hitball_starts.csv', delimiter=',')
    goals = np.loadtxt(data_dir + '/hitball_goals.csv', delimiter=',')

    if np.shape(queries)[-1] == np.shape(goals)[0]:
        queries = np.expand_dims(queries, axis=-1)

    inputs = np.concatenate([queries, starts, goals], axis=1)
    d_input = np.shape(queries)[-1]

    # prepare model
    _svr = SVR(gamma='scale', C=1.0, epsilon=0.1)
    svr = MultiDimSkRegressor(_svr)

    kernel = RBF(42, (1e-2, 1e2)) + WhiteKernel()
    _gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr = MultiDimSkRegressor(_gpr)

    mp = VMP(dim=2, kernel_num=10)

    models = {'SVR': svr, 'GPR': gpr, 'Uniform': None}
    rstates = np.random.randint(0, 100, size=MAX_EXPNUM)
    n_test = 100

    srates = {}
    allres = np.zeros(shape=(len(model_names), MAX_EXPNUM, len(nsamples)))
    for modelId in range(len(model_names)):
        model = models[model_names[modelId]]
        csrates = np.zeros(shape=(MAX_EXPNUM,len(nsamples)))

        for expId in range(MAX_EXPNUM):
            trdata, tdata, trvmps, tvmps = train_test_split(inputs, vmps, test_size=0.9, random_state=rstates[expId])
            print("use {} data for training and {} data for testing".format(np.shape(trdata)[0], np.shape(tdata)[0]))
            print("======== Exp: {} with {} ========".format(expId, model_names[modelId]))
            trqueries = trdata[:, 0:d_input]
            tqueries = tdata[:n_test, 0:d_input]
            tstarts = tdata[:n_test, d_input:d_input + 2]
            tgoals = tdata[:n_test, d_input + 2:]

            if model is not None:
                model.fit(trqueries, trvmps)
                wmeans = model.predict(tqueries)

            for sampleId in range(len(nsamples)):
                if model is not None:
                    wouts = sample_baseline(wmeans, nsamples[sampleId])
                else:
                    wouts = np.random.uniform(low=vmps.min(), high=vmps.max(), size=(n_test,nsamples[sampleId], np.shape(vmps)[1]))

                srate = evaluate_hitball(mp, wouts, tqueries, tstarts, tgoals)
                csrates[expId, sampleId] = srate
                allres[modelId, expId, sampleId] = srate

        srates[model_names[modelId]] = np.mean(csrates, axis=0)

    return srates, allres


if __name__ == '__main__':
    MAX_EXPNUM = 5
    nsamples = [10, 30, 50, 70]
    model_names = ['GPR']
    srates, allres = run_baselines_for_hitball(MAX_EXPNUM, nsamples, model_names)

    res_file = open("result_baseline", 'a')
    for modelId in range(len(model_names)):
        res_file.write(model_names[modelId] + '\n')
        np.savetxt(res_file, np.array(allres[modelId, :, :]), delimiter=',')
        np.savetxt(res_file, np.array(srates[model_names[modelId]]), delimiter=',')
