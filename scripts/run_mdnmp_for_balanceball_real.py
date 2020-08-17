import os, inspect, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../experiments/mujoco')


import numpy as np
from models.mdnmp import MDNMP
from mp.qvmp import QVMP

from sklearn.model_selection import train_test_split
from optparse import OptionParser
from experiments.exp_tools import get_training_data_from_2d_grid

import tensorflow as tf
if tf.__version__ < '2.0.0':
    import tflearn
    VAR_INIT = tflearn.initializations.uniform(minval=-.1, maxval=.1, seed=42)
else:
    from tensorflow.keras import initializers
    VAR_INIT = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)


parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=3)
parser.add_option("-q", "--qfile", dest="qfile", type="string", default="balanceball_queries.csv")
parser.add_option("-w", "--wfile", dest="wfile", type="string", default="balanceball_weights.csv")
parser.add_option("-d", "--result_dir", dest="rdir", type="string", default="results_real_balanceball")
parser.add_option("--grid_samples", dest="is_grid_samples", action="store_true", default=False)
parser.add_option("--num_train", dest="ntrain", type="int", default=10)
parser.add_option("--sample_num", dest="nsamples", type="int", default=10)
parser.add_option("--model_name", dest="model_name", type="string", default="mce")
parser.add_option("--max_epochs", dest="max_epochs", type="int", default=10000)
(options, args) = parser.parse_args(sys.argv)


queries = np.loadtxt(options.qfile, delimiter=',')
vmps = np.loadtxt(options.wfile, delimiter=',')

queries[:,0] = queries[:,0]/20
queries[:,1] = queries[:,1]/30


# prepare model
nn_structure = {'d_feat': 40,
                'feat_layers': [20],
                'mean_layers': [60],
                'scale_layers': [60],
                'mixing_layers': [20]}

d_input = np.shape(queries)[-1]
d_output = np.shape(vmps)[1]


tratio = 0.5
if options.is_grid_samples:
    _, ids = get_training_data_from_2d_grid(options.ntrain, queries=queries)
    trqueries = queries[ids,:]
    trvmps = vmps[ids,:]
    _, tqueries, _, tvmps = train_test_split(queries, vmps, test_size=tratio, random_state=42)
else:
    trqueries, tqueries, trvmps, tvmps = train_test_split(queries, vmps, test_size=tratio, random_state=42)



mdnmp = MDNMP(n_comps=options.nmodel, d_input=d_input, d_output=d_output, nn_structure=nn_structure,
              var_init=VAR_INIT, scaling=1.0)


lrate = 0.001

if options.model_name == "omce":
    mdnmp.lratio['entropy'] = 10
    mdnmp.is_orthogonal_cost = True
    mdnmp.is_mce_only = True
    mdnmp.is_normalized_grad = False
    mdnmp.cross_train = True
    mdnmp.nll_lrate = lrate
    mdnmp.ent_lrate = lrate
elif options.model_name == "elk":
    mdnmp.lratio['entropy'] = 10
    mdnmp.is_orthogonal_cost = True
    mdnmp.is_mce_only = False
    mdnmp.is_normalized_grad = False
    mdnmp.cross_train = True
    mdnmp.nll_lrate = lrate
    mdnmp.ent_lrate = lrate
elif options.model_name == "mce":
    mdnmp.lratio['entropy'] = 10
    mdnmp.is_orthogonal_cost = False
    mdnmp.is_mce_only = True
    mdnmp.is_normalized_grad = False
else:
    mdnmp.lratio['entropy'] = 0

mdnmp.build_mdn(learning_rate=lrate)
mdnmp.init_train()
is_pos = np.ones(shape=(np.shape(trvmps)[0], 1))
mdnmp.train(trqueries, trvmps, is_pos, max_epochs=options.max_epochs, is_load=False, is_save=False)

result_dir = options.rdir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

wout, _ = mdnmp.predict(tqueries, options.nsamples)

wfname = result_dir + '/' + options.model_name + '_balanceball_testing_weights.csv'
qfname = result_dir + '/' + options.model_name + '_balanceball_testing_queries.csv'
wfile = open(wfname, 'w+')
qfile = open(qfname, 'w+')
for i in range(np.shape(wout)[0]):
    weights = wout[i,:,:]
    np.savetxt(wfile, weights, delimiter=',')
    cquery = np.expand_dims(tqueries[i,:], axis=0)
    cqueries = np.tile(cquery, (np.shape(weights)[0], 1))
    cqueries[:,0] = cqueries[:,0] * 20
    cqueries[:,1] = cqueries[:,1] * 30
    np.savetxt(qfile, cqueries, delimiter=',')


