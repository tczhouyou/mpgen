import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')
from models.mdgan import cMDGAN
from models.mdnmp import MDNMP
from models.gmgan import GMGAN
import sys
from optparse import OptionParser
import numpy as np
from sklearn.model_selection import train_test_split
from run_mdnmp_for_hitball import train_evaluate_mdnmp_for_hitball
from run_gmgan_for_hitball import train_evaluate_gmgan_for_hitball
from run_baselines_for_hitball import train_evaluate_baseline_for_hitball
import tensorflow as tf

if tf.__version__ < '2.0.0':
    import tflearn
    VAR_INIT = tflearn.initializations.uniform(minval=-.003, maxval=.003, seed=42)
    VAR_INIT_DIS = tflearn.initializations.normal(stddev=0.1, seed=42)
else:
    from tensorflow.keras import initializers
    VAR_INIT = initializers.RandomNormal(stddev=0.0003, seed=42)
    VAR_INIT_DIS = initializers.RandomNormal(stddev=0.002, seed=42)

parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=3)
parser.add_option("-n", "--num_exp", dest="expnum", type="int", default=1)
parser.add_option("--num_test", dest="ntest", type="int", default=100)
parser.add_option("-d", "--result_dir", dest="result_dir", type="string", default="results_compare_docking")
parser.add_option("--draw", dest="isdraw", action="store_true", default=False)
(options, args) = parser.parse_args(sys.argv)

data_dir = 'experiments/mujoco/hitball/hitball_mpdata_v1'
queries = np.loadtxt(data_dir + '/hitball_queries.csv', delimiter=',')
vmps = np.loadtxt(data_dir + '/hitball_weights.csv', delimiter=',')
starts = np.loadtxt(data_dir + '/hitball_starts.csv', delimiter=',')
goals = np.loadtxt(data_dir + '/hitball_goals.csv', delimiter=',')
data = np.concatenate([queries, starts, goals], axis=1)

rstates = np.random.randint(0, 100, size=options.expnum)
d_input = np.shape(queries)[-1]
d_output = np.shape(vmps)[1]

mdnmp_struct = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [40],
                'scale_layers': [40],
                'mixing_layers': [20]}
mdnmp = MDNMP(n_comps=options.nmodel, d_input=d_input, d_output=d_output, nn_structure=mdnmp_struct, scaling=1,
              var_init=VAR_INIT)

nn_structure = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [40],
                'scale_layers': [40],
                'mixing_layers': [20],
                'discriminator': [40],
                'lambda': [10],
                'd_response': [40,5],
                'd_context': [10,5]}
gmgan = GMGAN(n_comps=options.nmodel, context_dim=d_input, response_dim=d_output, nn_structure=nn_structure, scaling=1,
              var_init=VAR_INIT, var_init_dis=VAR_INIT_DIS)


# start experiment
num_train_data = np.array([50, 100, 200])
tsize = (np.shape(data)[0] -num_train_data)/np.shape(data)[0]

result_dir = options.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for expId in range(options.expnum):
    baseline_res = np.zeros(shape=(1,len(tsize)))
    omdnmp_res = np.zeros(shape=(1, len(tsize)))
    emdnmp_res = np.zeros(shape=(1, len(tsize)))
    egmgan_res = np.zeros(shape=(1, len(tsize)))

    for i in range(len(tsize)):
        tratio = tsize[i]
        trdata, tdata, trvmps, tvmps = train_test_split(data, vmps, test_size=tratio, random_state=rstates[expId])
        print("======== exp: %1d for training dataset: %1d =======" % (expId, np.shape(trdata)[0]))
        trqueries = trdata[:, 0:2]

        print(">>>> train baselines")
        baseline_res[0, i] = train_evaluate_baseline_for_hitball("GPR", trqueries, trvmps, tdata,
                                                                 sample_num=10, isvel=True,
                                                                 env_file="hitball_exp_v1.xml",
                                                                 isdraw=options.isdraw, num_test=options.ntest)

        print(">>>> train entropy MDN")
        emdnmp_res[0, i] = train_evaluate_mdnmp_for_hitball(mdnmp, trqueries, trvmps, tdata, True, max_epochs=20000,
                                                            sample_num=10, isvel=True, env_file="hitball_exp_v1.xml",
                                                            isdraw=options.isdraw, num_test=options.ntest,
                                                            learning_rate=0.0001)

        print(">>>> train GMGANs")
        egmgan_res[0, i] = train_evaluate_gmgan_for_hitball(gmgan, trqueries, trvmps, tdata, False, max_epochs=20000,
                                                            sup_max_epoch=30001,
                                                            sample_num=10, isvel=True, env_file="hitball_exp_v1.xml",
                                                            isdraw=options.isdraw, num_test=options.ntest, g_lrate=0.0001, d_lrate=0.002)
        #
        print(">>>> train original MDN")
        omdnmp_res[0, i] = train_evaluate_mdnmp_for_hitball(mdnmp, trqueries, trvmps, tdata, False, max_epochs=20000,
                                                            sample_num=10, isvel=True, env_file="hitball_exp_v1.xml",
                                                            isdraw=options.isdraw, num_test=options.ntest, learning_rate=0.0001)


    with open(result_dir + "/baselines", "a") as f:
        np.savetxt(f, np.array(baseline_res), delimiter=',', fmt='%.3f')
    with open(result_dir + "/entropy_mdn", "a") as f:
        np.savetxt(f, np.array(emdnmp_res), delimiter=',', fmt='%.3f')
    with open(result_dir + "/entropy_gmgan", "a") as f:
        np.savetxt(f, np.array(egmgan_res), delimiter=',', fmt='%.3f')
    with open(result_dir + "/original_mdn", "a") as f:
        np.savetxt(f, np.array(omdnmp_res), delimiter=',', fmt='%.3f')


