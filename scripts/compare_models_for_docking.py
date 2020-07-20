import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')

from models.mdnmp import MDNMP
from models.gmgan import GMGAN
import sys
from optparse import OptionParser
import numpy as np
from sklearn.model_selection import train_test_split
from experiments.evaluate_exps import evaluate_docking
from run_baselines_for_docking import train_evaluate_baseline_for_docking
from run_mdnmp_for_docking import train_evaluate_mdnmp_for_docking
from run_gmgan_for_docking import train_evaluate_gmgan_for_docking

import tensorflow as tf

if tf.__version__ < '2.0.0':
    import tflearn
    VAR_INIT = tflearn.initializations.normal(stddev=0.003, seed=42)
    VAR_INIT_DIS = tflearn.initializations.normal(stddev=0.1, seed=42)
else:
    from tensorflow.keras import initializers
    VAR_INIT = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
    VAR_INIT_DIS = initializers.RandomNormal(stddev=0.02, seed=42)


parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=3)
parser.add_option("-n", "--num_exp", dest="expnum", type="int", default=1)
parser.add_option("-d", "--result_dir", dest="result_dir", type="string", default="results_compare_docking")
(options, args) = parser.parse_args(sys.argv)


queries = np.loadtxt('../data/docking_queries.csv', delimiter=',')
vmps = np.loadtxt('../data/docking_weights.csv', delimiter=',')
starts = np.loadtxt('../data/docking_starts.csv', delimiter=',')
goals = np.loadtxt('../data/docking_goals.csv', delimiter=',')


wtest = np.expand_dims(vmps, axis=1)
cc, successId = evaluate_docking(wtest, queries, starts,goals)
data = np.concatenate([queries, starts, goals], axis=1)
data = data[successId, :]
vmps = vmps[successId,:]
knum = np.shape(vmps)[1]
rstates = np.random.randint(0, 100, size=options.expnum)

# create mdnmp
mdnmp_struct = {'d_feat': 20,
                'feat_layers': [40], #[60]
                'mean_layers': [60], #[60]
                'scale_layers': [60],
                'mixing_layers': [20]}
mdnmp = MDNMP(n_comps=options.nmodel, d_input=6, d_output=knum, nn_structure=mdnmp_struct, scaling=1, var_init=VAR_INIT)

# create gmgan
nn_structure = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [40], #[60]
                'scale_layers': [40], #[60]
                'mixing_layers': [40],
                'discriminator': [20],
                'lambda': [10],
                'd_response': [40,5],
                'd_context': [20,5]}
gmgan = GMGAN(n_comps=options.nmodel, context_dim=6, response_dim=knum, nn_structure=nn_structure, scaling=1,
              var_init=VAR_INIT, var_init_dis=VAR_INIT_DIS, batch_size=100)

# start experiment
num_train_data = np.array([100])
tsize = (np.shape(data)[0] -num_train_data)/np.shape(data)[0]
print(tsize)

result_dir = options.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

mdnmp.lratio = {'likelihood': 1, 'mce': 0, 'regularization': 0.00001, 'failure': 0, 'eub': 0}

for expId in range(options.expnum):
    baseline = np.zeros(shape=(1,len(tsize)))
    omdn = np.zeros(shape=(1, len(tsize)))
    mce = np.zeros(shape=(1, len(tsize)))
    omce = np.zeros(shape=(1, len(tsize)))
    oelk = np.zeros(shape=(1, len(tsize)))

    for i in range(len(tsize)):
        tratio = tsize[i]
        trdata, tdata, trvmps, tvmps = train_test_split(data, vmps, test_size=tratio, random_state=rstates[expId])
        print("======== exp: %1d for training dataset: %1d =======" % (expId, np.shape(trdata)[0]))

        trqueries = trdata[:, 0:6]
        print(">>>> train elk MDN")
        mdnmp.lratio['entropy'] = 1
        mdnmp.is_orthogonal_cost=True
        mdnmp.is_mce_only=False
        oelk[0, i] = train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata,
                                                            max_epochs=20000,
                                                            sample_num=10, learning_rate=0.00003)


        print(">>>> train ori MDN")
        mdnmp.lratio['entropy'] = 0
        omdn[0, i] = train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata,
                                                            max_epochs=20000,
                                                            sample_num=10, learning_rate=0.00003)


        print(">>>> train orthognal mce MDN")
        mdnmp.lratio['entropy'] = 3
        mdnmp.is_orthogonal_cost=True
        mdnmp.is_mce_only=True
        omce[0, i] = train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata,
                                                            max_epochs=20000,
                                                            sample_num=10, learning_rate=0.00003)
        print(">>>> train mce MDN")
        mdnmp.lratio['entropy'] = 3
        mdnmp.is_orthogonal_cost=False
        mdnmp.is_mce_only=True
        mce[0, i] = train_evaluate_mdnmp_for_docking(mdnmp, trqueries, trvmps, tdata,
                                                            max_epochs=20000,
                                                            sample_num=10, learning_rate=0.00003)


        print(">>>> train baselines")
        baseline[0, i] = train_evaluate_baseline_for_docking('GPR', trqueries, trvmps, tdata, sample_num=10)


    with open(result_dir + "/baseline", "a") as f:
        np.savetxt(f, np.array(baseline), delimiter=',', fmt='%.3f')
   # with open(result_dir + "/entropy_gmgan", "a") as f:
    #    np.savetxt(f, np.array(egmgan_res), delimiter=',', fmt='%.3f')
    with open(result_dir + "/omdn", "a") as f:
        np.savetxt(f, np.array(omdn), delimiter=',', fmt='%.3f')
    with open(result_dir + "/mce", "a") as f:
        np.savetxt(f, np.array(mce), delimiter=',', fmt='%.3f')
    with open(result_dir + "/omce", "a") as f:
        np.savetxt(f, np.array(omce), delimiter=',', fmt='%.3f')
    with open(result_dir + "/oelk", "a") as f:
        np.savetxt(f, np.array(oelk), delimiter=',', fmt='%.3f')
