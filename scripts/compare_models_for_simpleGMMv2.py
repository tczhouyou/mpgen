import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../mp')

from models.mdnmp import MDNMP
import sys
from optparse import OptionParser
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import tensorflow as tf
from math_tools.MathTools import draw_contour, mdn_to_gmm, calc_kl_mc
from sklearn.model_selection import train_test_split

if tf.__version__ < '2.0.0':
    import tflearn
    VAR_INIT = tflearn.initializations.normal(stddev=0.003, seed=42)
    VAR_INIT_DIS = tflearn.initializations.normal(stddev=0.1, seed=42)
else:
    from tensorflow.keras import initializers
    VAR_INIT = initializers.RandomUniform(minval=-0.0003, maxval=0.0003, seed=42)
    VAR_INIT_DIS = initializers.RandomNormal(stddev=0.02, seed=42)


def compare_gmm(outdict, gmmlist):
    means = outdict['means']
    dist = 0
    for i in range(len(gmmlist)):
        gmm = gmmlist[i]
        mean = means[i,:,:]

        dist1 = np.linalg.norm(gmm.means_[0,:] - mean[0,:]) + np.linalg.norm(gmm.means_[1,:] - mean[1,:])
        dist2 = np.linalg.norm(gmm.means_[0,:] - mean[1,:]) + np.linalg.norm(gmm.means_[1,:] - mean[0,:])
        dist = dist + np.minimum(dist1, dist2)

    return dist / len(gmmlist)

def create_gmm_from_angle(angles, num_data=1):
    ## create a grid of data for GMM
    gmm_list = []
    outputs = []
    inputs = []
    for i in range(np.shape(angles)[0]):
        gmm = mixture.GaussianMixture(n_components=2, covariance_type='diag')
        gmm.fit(np.random.uniform(-1.0, 1.0, (10, 2)))
        angle = angles[i]
        vecx = 10 * np.array([np.cos(angle), np.sin(angle)])
        vecy = 5 * np.array([np.cos(angle + np.pi /2 ), np.sin(angle + np.pi/2)])
        gmm.means_[0, :] = vecx + vecy
        gmm.means_[1, :] = vecx - vecy
        gmm.covariances_[0, :] = np.array([1, 1])
        gmm.covariances_[1, :] = np.array([1, 1])
        gmm.weights_[0] = 0.5
        gmm.weights_[1] = 0.5
        gmm_list.append(gmm)

        out, _ = gmm.sample(n_samples=num_data)
        outputs.append(out[0])
        for j in range(num_data):
            inputs.append(angles[i])

    inputs = np.array(inputs)
    inputs = np.expand_dims(inputs, axis=1)
    return gmm_list, np.stack(outputs), inputs




parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=4)
parser.add_option("-n", "--num_exp", dest="expnum", type="int", default=1)
(options, args) = parser.parse_args(sys.argv)



n_samples = np.array([100])

mdnmp_struct = {'d_feat': 5,
                'feat_layers': [10],
                'mean_layers': [10],
                'scale_layers': [10],
                'mixing_layers': [5]}
mdnmp = MDNMP(n_comps=options.nmodel, d_input=1, d_output=2, nn_structure=mdnmp_struct, scaling=1, var_init=VAR_INIT)

mdnmp.lratio = {'likelihood': 1, 'mce': 0, 'regularization': 0, 'failure': 0, 'eub': 0}
max_epochs = 20000
lrate = 0.00003

_, axes = plt.subplots(nrows=options.expnum, ncols=4)
axid = 0


angles = np.linspace(0, 2 * np.pi, 100)
_, outputs, inputs = create_gmm_from_angle(angles)
trinputs, tinputs, troutputs, toutputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
tgmmlist, _, _ = create_gmm_from_angle(tinputs.squeeze())
weights = np.ones(shape=(np.shape(troutputs)[0], 1))

for expId in range(options.expnum):
    print(">>>> train omdn")
    mdnmp.lratio['entropy'] = 0
    mdnmp.is_normalized_grad = False
    mdnmp.build_mdn(learning_rate=lrate)
    mdnmp.init_train()
    isSuccess = mdnmp.train(trinputs, troutputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
    if isSuccess:
        _, outdict = mdnmp.predict(tinputs)
        if options.expnum== 1:
            ax = axes[0]
        else:
            ax = axes[expId,0]

        cost = compare_gmm(outdict, tgmmlist)
        print('omdn ==> kl: {}'.format(cost))


    print(">>>> train mce")
    mdnmp.lratio['entropy'] =10
    mdnmp.is_orthogonal_cost=False
    mdnmp.is_mce_only=True
    mdnmp.is_normalized_grad = False
    mdnmp.cross_train = False
    mdnmp.build_mdn(learning_rate=lrate)
    mdnmp.init_train()
    isSuccess = mdnmp.train(trinputs, troutputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
    if isSuccess:
        _, outdict = mdnmp.predict(tinputs)
        if options.expnum == 1:
            ax = axes[0]
        else:
            ax = axes[expId, 0]

        cost = compare_gmm(outdict, tgmmlist)
        print('mce ==> kl: {}'.format(cost))

    print(">>>> train oelk")
    mdnmp.lratio['entropy'] =3
    mdnmp.is_orthogonal_cost = True
    mdnmp.is_mce_only = False
    mdnmp.is_normalized_grad = False
    mdnmp.cross_train = True
    mdnmp.nll_lrate = lrate
    mdnmp.ent_lrate =10 * lrate
    mdnmp.build_mdn(learning_rate=lrate)
    mdnmp.init_train()
    isSuccess = mdnmp.train(trinputs, troutputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
    if isSuccess:
        _, outdict = mdnmp.predict(tinputs)
        if options.expnum == 1:
            ax = axes[0]
        else:
            ax = axes[expId, 0]

        cost = compare_gmm(outdict, tgmmlist)
        print('oelk ==> kl: {}'.format(cost))

    print(">>>> train omce")
    mdnmp.lratio['entropy'] =10
    mdnmp.is_orthogonal_cost=True
    mdnmp.is_mce_only=True
    mdnmp.is_normalized_grad = False
    mdnmp.cross_train = True
    mdnmp.nll_lrate = lrate
    mdnmp.ent_lrate = 10 * lrate
    mdnmp.build_mdn(learning_rate=lrate)
    mdnmp.init_train()
    isSuccess = mdnmp.train(trinputs, troutputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
    if isSuccess:
        _, outdict = mdnmp.predict(tinputs)
        if options.expnum == 1:
            ax = axes[0]
        else:
            ax = axes[expId, 0]

        cost = compare_gmm(outdict, tgmmlist)
        print('omce ==> kl: {}'.format(cost))

    print('=======================================')


mean_kl = mean_kl / options.expnum
print(mean_kl)
plt.show()








