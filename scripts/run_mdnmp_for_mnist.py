import os, inspect, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../experiments/mujoco')


import numpy as np
from models.mdnmp import MDNMP
from mp.promp import ProMP
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from optparse import OptionParser
from experiments.exp_tools import get_training_data_from_2d_grid
import gzip

import tensorflow as tf
if tf.__version__ < '2.0.0':
    import tflearn
    VAR_INIT = tflearn.initializations.uniform(minval=-.1, maxval=.1, seed=42)
else:
    from tensorflow.keras import initializers
    VAR_INIT = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)


data_file = '../experiments/mnist/t10k-images-idx3-ubyte.gz'
train_input_file = '../experiments/mnist/handwriting_queries'
train_output_file = '../experiments/mnist/handwriting_weights'

parser = OptionParser()
parser.add_option("-m", "--nmodel", dest="nmodel", type="int", default=3)
parser.add_option("--num_test", dest="ntest", type="int", default=100)
parser.add_option("--sample_num", dest="nsamples", type="int", default=10)
parser.add_option("--model_name", dest="model_name", type="string", default="mce")
parser.add_option("--max_epochs", dest="max_epochs", type="int", default=10000)
(options, args) = parser.parse_args(sys.argv)

num_images = options.ntest
f = gzip.open(data_file)
f.read(16)
image_size = 28
buf = f.read(image_size * image_size * 10000)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(10000, image_size, image_size, 1)
idx = np.random.choice(10000, num_images, replace=False)
tdata = data[idx, :, :, :]


queries = np.loadtxt(train_input_file, delimiter=',')
weights = np.loadtxt(train_output_file, delimiter=',')
queries = queries / 255

# prepare model
nn_structure = {'d_feat': 100,
                'feat_layers': [1000,400],
                'mean_layers': [200,100],
                'scale_layers': [200,100],
                'mixing_layers': [200,20]}

d_input = np.shape(queries)[-1]
d_output = np.shape(weights)[1]

trqueries = queries
trweights = weights

mdnmp = MDNMP(n_comps=options.nmodel, d_input=d_input, d_output=d_output, nn_structure=nn_structure,
              var_init=VAR_INIT, scaling=1.0)


lrate = 0.00002

if options.model_name == "omce":
    mdnmp.lratio['entropy'] = 10
    mdnmp.is_orthogonal_cost = True
    mdnmp.is_mce_only = True
    mdnmp.is_normalized_grad = False
    mdnmp.cross_train = True
    mdnmp.nll_lrate = lrate
    mdnmp.ent_lrate = 10 * lrate
elif options.model_name == "elk":
    mdnmp.lratio['entropy'] = 10
    mdnmp.is_orthogonal_cost = True
    mdnmp.is_mce_only = False
    mdnmp.is_normalized_grad = False
    mdnmp.cross_train = True
    mdnmp.nll_lrate = lrate
    mdnmp.ent_lrate = 10 * lrate
elif options.model_name == "mce":
    mdnmp.lratio['entropy'] = 10
    mdnmp.is_orthogonal_cost = False
    mdnmp.is_mce_only = True
    mdnmp.is_normalized_grad = False
else:
    mdnmp.lratio['entropy'] = 0

mdnmp.build_mdn(learning_rate=lrate)
mdnmp.init_train()
is_pos = np.ones(shape=(np.shape(trweights)[0], 1))
mdnmp.train(trqueries, trweights, is_pos, max_epochs=options.max_epochs, is_load=False, is_save=False)

############ test
images = np.asarray(tdata).squeeze()
tqueries = np.reshape(images, newshape=(num_images, -1))
tqueries = tqueries / 255
wout, _ = mdnmp.predict(tqueries, options.nsamples)

promp = ProMP(dim=2, kernel_num=10)
for i in range(np.shape(wout)[0]):
    image = images[i]
    plt.cla()
    plt.imshow(image)

    for j in range(np.shape(wout)[1]):
        promp.set_weights(wout[i,j,:])
        traj = promp.roll()
        for k in range(np.shape(traj)[0]):
            plt.plot(traj[:k,1], 28 - traj[:k,2], 'r-')
            plt.xlim(0, 28)
            plt.ylim(0, 28)
            plt.pause(0.001)






