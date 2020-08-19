import os, inspect, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../experiments/mujoco')

from models.baselines import MultiDimSkRegressor, sample_baseline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
import numpy as np
from mp.qvmp import QVMP

from sklearn.model_selection import train_test_split
from optparse import OptionParser
from experiments.exp_tools import get_training_data_from_2d_grid

parser = OptionParser()
parser.add_option("-q", "--qfile", dest="qfile", type="string", default="balanceball_queries.csv")
parser.add_option("-w", "--wfile", dest="wfile", type="string", default="balanceball_weights.csv")
parser.add_option("-d", "--result_dir", dest="rdir", type="string", default="results_real_balanceball")
parser.add_option("--grid_samples", dest="is_grid_samples", action="store_true", default=False)
parser.add_option("--num_train", dest="ntrain", type="int", default=10)
parser.add_option("--sample_num", dest="nsamples", type="int", default=10)
parser.add_option("--model_name", dest="model_name", type="string", default="mce")
(options, args) = parser.parse_args(sys.argv)


queries = np.loadtxt(options.qfile, delimiter=',')
vmps = np.loadtxt(options.wfile, delimiter=',')

queries[:,0] = queries[:,0]/20
queries[:,1] = queries[:,1]/30


# prepare model
if options.model_name == 'SVR':
    _svr = SVR(gamma='scale', C=1.0, epsilon=0.1)
    model = MultiDimSkRegressor(_svr)
else:
    _gpr = GaussianProcessRegressor()
    model = MultiDimSkRegressor(_gpr)


tratio = 0.5
if options.is_grid_samples:
    _, ids = get_training_data_from_2d_grid(options.ntrain, queries=queries)
    trqueries = queries[ids,:]
    trvmps = vmps[ids,:]
    _, tqueries, _, tvmps = train_test_split(queries, vmps, test_size=tratio, random_state=42)
else:
    trqueries, tqueries, trvmps, tvmps = train_test_split(queries, vmps, test_size=tratio, random_state=42)


model.fit(trqueries, trvmps)
wouts = model.predict(tqueries)

result_dir = options.rdir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


wfname = result_dir + '/' + options.model_name + '_balanceball_testing_weights.csv'
qfname = result_dir + '/' + options.model_name + '_balanceball_testing_queries.csv'
wfile = open(wfname, 'w+')
qfile = open(qfname, 'w+')

np.savetxt(wfile, wouts, delimiter=',')
tqueries[:,0] = tqueries[:,0] * 20
tqueries[:,1] = tqueries[:,1] * 30
np.savetxt(qfile, tqueries, delimiter=',')


