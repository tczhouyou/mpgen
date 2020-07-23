import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../experiments/mujoco')

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
from experiments.mujoco.hitball.hitball_exp import Armar6HitBallExpV1
from experiments.exp_tools import get_training_data_from_2d_grid


if tf.__version__ < '2.0.0':
    import tflearn
    VAR_INIT = tflearn.initializations.uniform(minval=-.003, maxval=.003, seed=42)
    VAR_INIT_DIS = tflearn.initializations.normal(stddev=0.1, seed=42)
else:
    from tensorflow.keras import initializers
    VAR_INIT = initializers.RandomUniform(minval=-0.0003, maxval=0.0003, seed=42)
    VAR_INIT_DIS = initializers.RandomNormal(stddev=0.002, seed=42)

parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=3)
parser.add_option("-n", "--num_exp", dest="expnum", type="int", default=1)
parser.add_option("--num_test", dest="ntest", type="int", default=100)
parser.add_option("-d", "--result_dir", dest="result_dir", type="string", default="results_compare_hitball")
parser.add_option("--draw", dest="isdraw", action="store_true", default=False)
parser.add_option("-v", dest="version", type="string", default="v2")
parser.add_option("--grid_samples", dest="is_grid_samples", action="store_true", default=False)
parser.add_option("--num_train", dest="ntrain", type="int", default=10)

(options, args) = parser.parse_args(sys.argv)



if options.version == "v2":
    data_dir = '../experiments/mujoco/hitball/hitball_mpdata_v2'
    env_file = 'hitball_exp_v2.xml'
elif options.version == "v1":
    data_dir = '../experiments/mujoco/hitball/hitball_mpdata_v1'
    env_file = 'hitball_exp_v1.xml'

queries = np.loadtxt(data_dir + '/hitball_queries.csv', delimiter=',')
vmps = np.loadtxt(data_dir + '/hitball_weights.csv', delimiter=',')
starts = np.loadtxt(data_dir + '/hitball_starts.csv', delimiter=',')
goals = np.loadtxt(data_dir + '/hitball_goals.csv', delimiter=',')
data = np.concatenate([queries, starts, goals], axis=1)

vmps = vmps * 100
if options.is_grid_samples:
    _, ids = get_training_data_from_2d_grid(options.ntrain, queries=queries)
    trdata = data[ids,:]
    trvmps = vmps[ids,:]

rstates = np.random.randint(0, 100, size=options.expnum)
d_input = np.shape(queries)[-1]
d_output = np.shape(vmps)[1]

mdnmp_struct = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [60],
                'scale_layers': [60],
                'mixing_layers': [20]}
mdnmp = MDNMP(n_comps=options.nmodel, d_input=d_input, d_output=d_output, nn_structure=mdnmp_struct, scaling=1.0,
              var_init=VAR_INIT)

nn_structure = {'d_feat': 20,
                'feat_layers': [40],
                'mean_layers': [60],
                'scale_layers': [60],
                'mixing_layers': [10],
                'discriminator': [20],
                'lambda': [10],
                'd_response': [40,5],
                'd_context': [20,5]}
gmgan = GMGAN(n_comps=options.nmodel, context_dim=d_input, response_dim=d_output, nn_structure=nn_structure, scaling=1,
              var_init=VAR_INIT, var_init_dis=VAR_INIT_DIS, batch_size=100)


# start experiment
num_train_data = np.array([50])
tsize = (np.shape(data)[0] - num_train_data)/np.shape(data)[0]

result_dir = options.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


mdnmp.lratio = {'likelihood': 1, 'mce': 0, 'regularization': 0, 'failure': 0, 'eub': 0}
max_epochs = 30000
sample_num = 10
lrate = 0.00003
mdnmp.is_normalized_grad = False
for expId in range(options.expnum):
    baseline = np.zeros(shape=(1,len(tsize)))
    omdn = np.zeros(shape=(1, len(tsize)))
    mce = np.zeros(shape=(1, len(tsize)))
    omce = np.zeros(shape=(1, len(tsize)))
    oelk = np.zeros(shape=(1, len(tsize)))

    for i in range(len(tsize)):
        tratio = tsize[i]

        if options.is_grid_samples:
            _, tdata, _, tvmps = train_test_split(data, vmps, test_size=tratio, random_state=rstates[expId])
            print("======== exp: %1d with grided training data =======" % (expId))
        else:
            trdata, tdata, trvmps, tvmps = train_test_split(data, vmps, test_size=tratio, random_state=rstates[expId])
            print("======== exp: %1d for training dataset: %1d =======" % (expId, np.shape(trdata)[0]))

        trqueries = trdata[:, 0:2]

        print(">>>> train original MDN")
        mdnmp.lratio['entropy'] = 0
        omdn[0, i] = train_evaluate_mdnmp_for_hitball(mdnmp, trqueries, trvmps, tdata, max_epochs=max_epochs,
                                                            sample_num=sample_num, isvel=True, env_file=env_file,
                                                            isdraw=options.isdraw, num_test=options.ntest, learning_rate=lrate,
                                                            EXP=Armar6HitBallExpV1)
        print(">>>> train mce")
        mdnmp.lratio['entropy'] = 10
        mdnmp.is_orthogonal_cost=False
        mdnmp.is_mce_only=True
        mdnmp.is_normalized_grad=False
        mce[0, i] = train_evaluate_mdnmp_for_hitball(mdnmp, trqueries, trvmps, tdata,max_epochs=max_epochs,
                                                            sample_num=sample_num, isvel=True, env_file=env_file,
                                                            isdraw=options.isdraw, num_test=options.ntest,
                                                            learning_rate=lrate, EXP=Armar6HitBallExpV1)


        print(">>>> train orthogonal mce")
        mdnmp.lratio['entropy'] = 10
        mdnmp.is_orthogonal_cost=True
        mdnmp.is_mce_only=True
        mdnmp.is_normalized_grad=False
        mdnmp.cross_train=True
        mdnmp.nll_lrate=lrate
        mdnmp.ent_lrate=lrate
        omce[0, i] = train_evaluate_mdnmp_for_hitball(mdnmp, trqueries, trvmps, tdata,max_epochs=max_epochs,
                                                            sample_num=sample_num, isvel=True, env_file=env_file,
                                                            isdraw=options.isdraw, num_test=options.ntest,
                                                            learning_rate=lrate, EXP=Armar6HitBallExpV1)



        print(">>>> train elk")
        mdnmp.lratio['entropy'] = 10
        mdnmp.is_orthogonal_cost=True
        mdnmp.is_mce_only=False
        mdnmp.is_normalized_grad=False
        mdnmp.cross_train=True
        mdnmp.nll_lrate=lrate
        mdnmp.ent_lrate=lrate
        oelk[0, i] = train_evaluate_mdnmp_for_hitball(mdnmp, trqueries, trvmps, tdata,max_epochs=max_epochs,
                                                            sample_num=sample_num, isvel=True, env_file=env_file,
                                                            isdraw=options.isdraw, num_test=options.ntest,
                                                            learning_rate=lrate, EXP=Armar6HitBallExpV1)


        print(">>>> train baselines")
        baseline[0, i] = train_evaluate_baseline_for_hitball("GPR", trqueries, trvmps, tdata,  sample_num=sample_num,
                                                                 isvel = True, env_file = env_file,
                                                                 isdraw = options.isdraw, num_test = options.ntest)





    with open(result_dir + "/baseline", "a") as f:
        np.savetxt(f, np.array(baseline), delimiter=',', fmt='%.3f')
    with open(result_dir + "/omdn", "a") as f:
        np.savetxt(f, np.array(omdn), delimiter=',', fmt='%.3f')
    with open(result_dir + "/omce", "a") as f:
        np.savetxt(f, np.array(omce), delimiter=',', fmt='%.3f')
    with open(result_dir + "/oelk", "a") as f:
        np.savetxt(f, np.array(oelk), delimiter=',', fmt='%.3f')
    with open(result_dir + "/mce", "a") as f:
        np.savetxt(f, np.array(mce), delimiter=',', fmt='%.3f')
