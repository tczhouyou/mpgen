import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import basic_nn
from basic_nn import fully_connected_nn
from util import sample_gmm
from basic_model import basicModel
from costfcn import gmm_nll_cost, model_entropy_cost, failure_cost

tf.compat.v1.disable_eager_execution()




class cMDGAN(basicModel):
    def __init__(self, n_comps, input_dim, out_dim, rand_input_dim, nn_structure,
                 batch_size=None, using_batch_norm=False, seed=42, eps=1e-20, scaling=0.1):
        basicModel.__init__(self, batch_size, using_batch_norm, seed, eps)
        self.n_comps = n_comps
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.d_out_dim = n_comps-1
        self.rand_input_dim = rand_input_dim

        self.nn_structure = nn_structure
        self.scaling = scaling

    def create_network(self):
        tf.compat.v1.reset_default_graph()
        self.data_input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.input_dim], name='input')
        self.rand_input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.rand_input_dim], name='rand_input')
        self.d_real_input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.out_dim + self.data_input], name='real')

        self.g_input = tf.concat([self.data_input, self.rand_input], axis=1)
        self.fake = fully_connected_nn(self.g_input, self.nn_structure['generator'], self.out_dim,
                                       latent_activation=tf.nn.relu, scope='generator')
        self.d_fake_input = tf.concat([self.fake, self.data_input], axis=1)



        
