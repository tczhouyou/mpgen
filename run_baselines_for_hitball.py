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
from mp.vmp import VMP

from experiments.mujoco.hitball.hitball_exp import evaluate_hitball, ENV_DIR, EXP_DIR
from experiments.mujoco.armar6_controllers.armar6_low_controller import TaskSpaceVelocityController, TaskSpaceImpedanceController
from experiments.mujoco.armar6_controllers.armar6_high_controller import TaskSpacePositionVMPController
from optparse import OptionParser

_svr = SVR(gamma='scale', C=1.0, epsilon=0.1)
svr = MultiDimSkRegressor(_svr)

kernel = RBF(42, (1e-2, 1e2)) + WhiteKernel()
_gpr = GaussianProcessRegressor()
gpr = MultiDimSkRegressor(_gpr)
possible_models = {'SVR': svr, 'GPR': gpr, 'Uniform': None}

def train_evaluate_baseline_for_hitball(model_name, trqueries, trvmps, tdata, sample_num=1,
                                        isvel=True, env_file="hitball_exp_v1.xml", isdraw=False, num_test=100):

    model = possible_models[model_name]
    if model is not None:
        model.fit(trqueries, trvmps)

    mp = VMP(dim=2, kernel_num=10)

    if num_test > np.shape(tdata)[0]:
        num_test = np.shape(tdata)[0]-1

    tqueries = tdata[:num_test, 0:2]
    tstarts = tdata[:num_test, 2:4]
    tgoals = tdata[:num_test, 4:6]
    wmeans = model.predict(tqueries)

    if model is not None:
        wouts = sample_baseline(wmeans, sample_num)
    else:
        wouts = np.random.uniform(low=trvmps.min(), high=trvmps.max(),
                                  size=(100, sample_num, np.shape(trvmps)[1]))

    if isvel:
        srate = evaluate_hitball(wouts, tqueries, tstarts, tgoals,
                                 low_ctrl=TaskSpaceVelocityController,
                                 high_ctrl=TaskSpacePositionVMPController(mp),
                                 env_path=ENV_DIR + env_file, isdraw=isdraw)
    else:
        srate = evaluate_hitball(wouts, tqueries, tstarts, tgoals,
                                 low_ctrl=TaskSpaceImpedanceController,
                                 high_ctrl=TaskSpacePositionVMPController(mp),
                                 env_path=ENV_DIR + env_file, isdraw=isdraw)
    return srate


def run_baselines_for_hitball(MAX_EXPNUM=20, nsamples=[1, 10, 30, 50], model_names=['Uniform', 'SVR', 'GPR'],
                              isvel=False, env_file="hitball_exp_v0.xml", data_dir="hitball_mpdata_v0"):
    data_dir = os.environ['MPGEN_DIR'] + EXP_DIR + data_dir
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

                if isvel:
                    srate = evaluate_hitball(wouts, tqueries, tstarts, tgoals,
                                             low_ctrl=TaskSpaceVelocityController,
                                             high_ctrl=TaskSpacePositionVMPController(mp),
                                             env_path=ENV_DIR + env_file)
                else:
                    srate = evaluate_hitball(wouts, tqueries, tstarts, tgoals,
                                             low_ctrl=TaskSpaceImpedanceController,
                                             high_ctrl=TaskSpacePositionVMPController(mp),
                                             env_path=ENV_DIR + env_file)

                csrates[expId, sampleId] = srate
                allres[modelId, expId, sampleId] = srate

        srates[model_names[modelId]] = np.mean(csrates, axis=0)

    return srates, allres


if __name__ == '__main__':
    MAX_EXPNUM = 1
    nsamples = [1]
    model_names = ['GPR']

    parser = OptionParser()
    parser.add_option("--env_file", dest="env_file", type="string", default="hitball_exp_v0.xml")
    parser.add_option("--data_dir", dest="data_dir", type="string", default="hitball_mpdata_v0")
    parser.add_option("--vel", dest="is_vel", action="store_true", default=False)
    (options, args) = parser.parse_args(sys.argv)
    srates, allres = run_baselines_for_hitball(MAX_EXPNUM, nsamples, model_names,
                                               isvel=options.is_vel,
                                               env_file=options.env_file,
                                               data_dir=options.data_dir)

    res_file = open("result_baseline", 'a')
    for modelId in range(len(model_names)):
        res_file.write(model_names[modelId] + '\n')
        np.savetxt(res_file, np.array(allres[modelId, :, :]), delimiter=',')
        np.savetxt(res_file, np.array(srates[model_names[modelId]]), delimiter=',')
