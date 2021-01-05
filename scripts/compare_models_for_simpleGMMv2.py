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
from math_tools.MathTools import draw_contour, mdn_to_gmm, calc_kl_mc, draw_contour_gmm
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
        vecx = np.array([np.cos(angle), np.sin(angle)])
        vecy = np.array([np.cos(angle + np.pi /2), np.sin(angle + np.pi/2)])
        gmm.means_[0, :] = vecx + vecy
        gmm.means_[1, :] = vecx - vecy
        gmm.covariances_[0, :] = np.array([.01, .01])
        gmm.covariances_[1, :] = np.array([.01, .01])
        gmm.weights_[0] = 0.5
        gmm.weights_[1] = 0.5
        gmm_list.append(gmm)

        out, _ = gmm.sample(n_samples=num_data)
        for j in range(num_data):
            inputs.append(angles[i])
            outputs.append(out[j])

    inputs = np.array(inputs)
    inputs = np.expand_dims(inputs, axis=1)
    return gmm_list, np.stack(outputs), inputs




parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=2)
parser.add_option("-n", "--num_exp", dest="expnum", type="int", default=1)
(options, args) = parser.parse_args(sys.argv)

mdnmp_struct = {'d_feat': 10,
                'feat_layers': [20],
                'mean_layers': [20],
                'scale_layers': [20],
                'mixing_layers': [10]}
mdnmp = MDNMP(n_comps=options.nmodel, d_input=1, d_output=2, nn_structure=mdnmp_struct, scaling=1, var_init=VAR_INIT)

mdnmp.lratio = {'likelihood': 1, 'mce': 0, 'regularization': 0, 'failure': 0, 'eub': 0}
max_epochs = 30000
lrate = 0.00005

num_train = [60]

X=np.linspace(-2,2,200)
Y=np.linspace(-2,2,200)

mdnmp.is_scale_scheduled=False
tangles = np.linspace(0, 2 * np.pi, 6)
tangles = tangles[:-1]

tgmmlist, touputs, tinputs = create_gmm_from_angle(tangles)

colors = ['b', 'g', 'r', 'c', 'm', 'y']

costs = np.zeros(shape=(options.expnum, 4))

isdraw = False
for expId in range(options.expnum):
    _, axes = plt.subplots(nrows=len(num_train), ncols=4)
    axid = 0

    for trid in range(len(num_train)):
        angles = np.linspace(0, 2 * np.pi, num_train[trid])
        _, troutputs, trinputs = create_gmm_from_angle(angles)
        weights = np.ones(shape=(np.shape(troutputs)[0], 1))

        print(">>>> train omdn")
        mdnmp.lratio['entropy'] = 0
        mdnmp.is_normalized_grad = False
        mdnmp.build_mdn(learning_rate=lrate)
        mdnmp.init_train()
        isSuccess, _, _, scale_cost = mdnmp.train(trinputs, troutputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)

        if isSuccess and scale_cost < 20:
            _, outdict = mdnmp.predict(tinputs)
            if len(num_train)== 1:
                ax = axes[0]
            else:
                ax = axes[trid,0]

            cost = compare_gmm(outdict, tgmmlist)
            costs[expId, 0] = cost
            print('omdn ==> kl: {}'.format(cost))

            if isdraw:
                for i in range(np.shape(tinputs)[0]):
                    tgmm = tgmmlist[i]
                    cgmm = mdn_to_gmm(outdict, ind=i)
                    draw_contour_gmm(ax, tgmm, X=X, Y=Y, color=colors[i], linestyle='dashed', linewidths=1.0)
                    draw_contour_gmm(ax, cgmm, X=X, Y=Y, color=colors[i], linewidths=2.0)

        plt.pause(0.1)
        print(">>>> train mce")
        mdnmp.lratio['entropy'] =10
        mdnmp.is_orthogonal_cost=False
        mdnmp.is_mce_only=True
        mdnmp.is_normalized_grad = False
        mdnmp.cross_train = False
        mdnmp.build_mdn(learning_rate=lrate)
        mdnmp.init_train()
        isSuccess, _, _, scale_cost = mdnmp.train(trinputs, troutputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
        if isSuccess and scale_cost < 20:
            _, outdict = mdnmp.predict(tinputs)
            if len(num_train)== 1:
                ax = axes[1]
            else:
                ax = axes[trid, 1]

            cost = compare_gmm(outdict, tgmmlist)
            costs[expId, 1] = cost

            print('mce ==> kl: {}'.format(cost))

            if isdraw:
                for i in range(np.shape(tinputs)[0]):
                    tgmm = tgmmlist[i]
                    cgmm = mdn_to_gmm(outdict, ind=i)
                    draw_contour_gmm(ax, tgmm, X=X, Y=Y, color=colors[i], linestyle='dashed', linewidths=1.0)
                    draw_contour_gmm(ax, cgmm, X=X, Y=Y, color=colors[i], linewidths=2.0)

        plt.pause(0.1)


        print(">>>> train omce")
        mdnmp.lratio['entropy'] = 10
        mdnmp.is_orthogonal_cost=True
        mdnmp.is_mce_only=True
        mdnmp.is_normalized_grad = False
        mdnmp.cross_train = True
        mdnmp.nll_lrate = lrate
        mdnmp.ent_lrate = 10 * lrate
        mdnmp.build_mdn(learning_rate=lrate)
        mdnmp.init_train()
        isSuccess, _, _, scale_cost = mdnmp.train(trinputs, troutputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
        if isSuccess and scale_cost < 20:
            _, outdict = mdnmp.predict(tinputs)
            if len(num_train)== 1:
                ax = axes[2]
            else:
                ax = axes[trid, 2]

            cost = compare_gmm(outdict, tgmmlist)
            costs[expId, 2] = cost

            print('omce ==> kl: {}'.format(cost))
            if isdraw:
                for i in range(np.shape(tinputs)[0]):
                    tgmm = tgmmlist[i]
                    cgmm = mdn_to_gmm(outdict, ind=i)
                    draw_contour_gmm(ax, tgmm, X=X, Y=Y, color=colors[i], linestyle='dashed', linewidths=1.0)
                    draw_contour_gmm(ax, cgmm, X=X, Y=Y, color=colors[i], linewidths=2.0)

        plt.pause(0.1)
        print(">>>> train oelk")
        mdnmp.lratio['entropy'] = 2
        mdnmp.is_orthogonal_cost = True
        mdnmp.is_mce_only = False
        mdnmp.is_normalized_grad = False
        mdnmp.cross_train = True
        mdnmp.nll_lrate = lrate
        mdnmp.ent_lrate = 10 * lrate
        mdnmp.build_mdn(learning_rate=lrate)
        mdnmp.init_train()
        isSuccess, _, _, scale_cost = mdnmp.train(trinputs, troutputs, weights, max_epochs=max_epochs, is_load=False,
                                                  is_save=False)
        if isSuccess and scale_cost < 20:
            _, outdict = mdnmp.predict(tinputs)
            if len(num_train) == 1:
                ax = axes[3]
            else:
                ax = axes[trid, 3]

            cost = compare_gmm(outdict, tgmmlist)
            costs[expId, 3] = cost

            print('oelk ==> kl: {}'.format(cost))

            if isdraw:
                for i in range(np.shape(tinputs)[0]):
                    tgmm = tgmmlist[i]
                    cgmm = mdn_to_gmm(outdict, ind=i)
                    draw_contour_gmm(ax, tgmm, X=X, Y=Y, color=colors[i], linestyle='dashed', linewidths=1.0)
                    draw_contour_gmm(ax, cgmm, X=X, Y=Y, color=colors[i], linewidths=2.0)

        plt.pause(0.1)
        print('=======================================')

    plt.pause(0.1)

np.savetxt('results_simplegmmv2.csv', costs, delimiter=',', fmt='%.3e')
plt.show()









