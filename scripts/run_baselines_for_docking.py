import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')

import numpy as np
from models.baselines import MultiDimSkRegressor, sample_baseline

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from experiments.evaluate_exps import evaluate_docking

_svr = SVR(gamma='scale', C=1.0, epsilon=0.1)
svr = MultiDimSkRegressor(_svr)

_gpr = GaussianProcessRegressor()
gpr = MultiDimSkRegressor(_gpr)
possible_models = {'SVR': svr, 'GPR': gpr, 'Uniform': None}

def train_evaluate_baseline_for_docking(model_name, trqueries, trvmps, tdata, sample_num=1):

    model = possible_models[model_name]
    if model is not None:
        model.fit(trqueries, trvmps)

    tqueries = tdata[:, 0:6]
    starts = tdata[:, 6:8]
    goals = tdata[:, 8:10]
    wmeans = model.predict(tqueries)

    if model is not None:
        wouts = sample_baseline(wmeans, sample_num)
    else:
        wouts = np.random.uniform(low=trvmps.min(), high=trvmps.max(),
                                  size=(100, sample_num, np.shape(trvmps)[1]))

    srate, _ = evaluate_docking(wouts, tqueries, starts, goals)
    return srate


