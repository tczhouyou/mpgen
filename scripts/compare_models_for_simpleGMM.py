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

if tf.__version__ < '2.0.0':
    import tflearn
    VAR_INIT = tflearn.initializations.normal(stddev=0.003, seed=42)
    VAR_INIT_DIS = tflearn.initializations.normal(stddev=0.1, seed=42)
else:
    from tensorflow.keras import initializers
    VAR_INIT = initializers.RandomUniform(minval=-0.00003, maxval=0.00003, seed=42)
    VAR_INIT_DIS = initializers.RandomNormal(stddev=0.02, seed=42)


parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=4)
parser.add_option("-n", "--num_exp", dest="expnum", type="int", default=1)
parser.add_option("-r", "--nrow", dest="nrow", type="int", default=2)
parser.add_option("-c", "--ncol", dest="ncol", type="int", default=2)
(options, args) = parser.parse_args(sys.argv)

## create a grid of data for GMM
gridx = np.linspace(0, 20, options.nrow)
gridy = np.linspace(0, 20, options.ncol)

gmm = mixture.GaussianMixture(n_components=options.nrow * options.ncol, covariance_type='diag')
gmm.fit(np.random.uniform(-1.0, 1.0, (10, 2)))
for i in range(len(gridx)):
    for j in range(len(gridy)):
        gmm.means_[i*options.nrow+j, :] = np.array([gridx[i], gridy[j]])
        gmm.covariances_[i*options.nrow+j,:] = np.array([1,1])
        gmm.weights_[i*options.nrow+j] = 1/(options.nrow * options.ncol)

# gmm.weights_[0] = 0.8/2
# gmm.weights_[1] = 0.8/2
# gmm.weights_[2] = 0.2/2
# gmm.weights_[3] = 0.2/2
mixingCoeffs = [[1/4,1/4,1/4,1/4], [0.8, 0.2/3, 0.2/3, 0.2/3]]
n_samples = 30

mdnmp_struct = {'d_feat': 10,
                'feat_layers': [40],
                'mean_layers': [40],
                'scale_layers': [40],
                'mixing_layers': [20]}
mdnmp = MDNMP(n_comps=options.nmodel, d_input=1, d_output=2, nn_structure=mdnmp_struct, scaling=1, var_init=VAR_INIT)
mdnmp.is_scale_scheduled = True

mdnmp.lratio = {'likelihood': 1, 'mce': 0, 'regularization': 0.00001, 'failure': 0, 'eub': 0}
max_epochs = 30000
lrate = 0.00003


X=np.linspace(-5,25,60)
Y=np.linspace(-5,25,60),

print('test kl: {}'.format(calc_kl_mc(gmm, gmm)))
mean_kl = np.zeros(shape=(len(mixingCoeffs), 4))
for expId in range(options.expnum):
    _, axes = plt.subplots(nrows=len(mixingCoeffs), ncols=4)
    axid = 0
    for i in range(len(mixingCoeffs)):
        gmm.weights_ = mixingCoeffs[i]
        outputs, y = gmm.sample(n_samples=n_samples)
        inputs = np.ones(shape=[n_samples, 1])
        ri = inputs[0, :]
        ri = np.expand_dims(ri, axis=1)
        weights = np.ones(shape=(np.shape(outputs)[0], 1))

        print(">>>> train omdn")
        mdnmp.lratio['entropy'] = 0
        mdnmp.is_normalized_grad = False
        mdnmp.build_mdn(learning_rate=lrate)
        mdnmp.init_train()
        isSuccess = mdnmp.train(inputs, outputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
        if isSuccess:
            _, outdict = mdnmp.predict(ri)
            if len(mixingCoeffs)== 1:
                ax = axes[0]
            else:
                ax = axes[i,0]

            ax.scatter(outputs[:, 0], outputs[:, 1])
            draw_contour(ax, outdict, X=X, Y=Y, color='red')
            cgmm = mdn_to_gmm(outdict)
            ckl = calc_kl_mc(gmm, cgmm)
            mean_kl[i, 0] = mean_kl[i, 0] + np.abs(ckl)
            print('omdn ==> kl: {}'.format(ckl))


        print(">>>> train mce")
        mdnmp.lratio['entropy'] =10
        mdnmp.is_orthogonal_cost=False
        mdnmp.is_mce_only=True
        mdnmp.is_normalized_grad = False
        mdnmp.cross_train = False
        mdnmp.build_mdn(learning_rate=lrate)
        mdnmp.init_train()
        isSuccess = mdnmp.train(inputs, outputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
        if isSuccess:
            _, outdict = mdnmp.predict(ri)
            if len(mixingCoeffs) == 1:
                ax = axes[1]
            else:
                ax = axes[i,1]

            ax.scatter(outputs[:, 0], outputs[:, 1])
            draw_contour(ax, outdict, X=X, Y=Y, color='red')
            cgmm = mdn_to_gmm(outdict)
            ckl = calc_kl_mc(gmm, cgmm)
            mean_kl[i, 1] = mean_kl[i, 1] + np.abs(ckl)
            print('mce ==> kl: {}'.format(ckl))


        print(">>>> train oelk")
        mdnmp.lratio['entropy'] =10
        mdnmp.is_orthogonal_cost = True
        mdnmp.is_mce_only = False
        mdnmp.is_normalized_grad = False
        mdnmp.cross_train = True
        mdnmp.nll_lrate = lrate
        mdnmp.ent_lrate =20 * lrate
        mdnmp.build_mdn(learning_rate=lrate)
        mdnmp.init_train()
        isSuccess = mdnmp.train(inputs, outputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
        if isSuccess:
            _, outdict = mdnmp.predict(ri)
            if len(mixingCoeffs) == 1:
                ax = axes[2]
            else:
                ax = axes[i,2]

            ax.scatter(outputs[:, 0], outputs[:, 1])
            draw_contour(ax, outdict, X=X, Y=Y, color='red')
            cgmm = mdn_to_gmm(outdict)
            ckl = calc_kl_mc(gmm, cgmm)
            mean_kl[i, 2] = mean_kl[i, 2] + np.abs(ckl)
            print('oelk ==> kl: {}'.format(ckl))


        print(">>>> train omce")
        mdnmp.lratio['entropy'] =10
        mdnmp.is_orthogonal_cost=True
        mdnmp.is_mce_only=True
        mdnmp.is_normalized_grad = False
        mdnmp.cross_train = True
        mdnmp.nll_lrate = lrate
        mdnmp.ent_lrate = 20 * lrate
        mdnmp.build_mdn(learning_rate=lrate)
        mdnmp.init_train()
        isSuccess = mdnmp.train(inputs, outputs, weights, max_epochs=max_epochs, is_load=False, is_save=False)
        if isSuccess:
            _, outdict = mdnmp.predict(ri)
            if len(mixingCoeffs) == 1:
                ax = axes[3]
            else:
                ax = axes[i,3]

            ax.scatter(outputs[:, 0], outputs[:, 1])
            draw_contour(ax, outdict, X=X, Y=Y, color='red')
            cgmm = mdn_to_gmm(outdict)
            ckl = calc_kl_mc(gmm, cgmm)
            mean_kl[i, 3] = mean_kl[i, 3] + np.abs(ckl)
            print('omce ==> kl: {}'.format(ckl))

        print('=======================================')



mean_kl = mean_kl / options.expnum
print(mean_kl)
plt.show()








