import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import basic_nn
from util import sample_gmm
from basic_model import basicModel
from costfcn import gmm_nll_cost, model_entropy_cost, failure_cost

tf.compat.v1.disable_eager_execution()


class MDNMP(basicModel):
    def __init__(self, n_comps, d_input, d_output, nn_structure,
                 batch_size=None, using_batch_norm=False, seed=42, eps=1e-20, scaling=0.1):
        basicModel.__init__(self, batch_size, using_batch_norm, seed, eps)
        self.n_comps = n_comps
        self.d_input = d_input
        self.d_output = d_output

        self.nn_structure = nn_structure
        self.lratio = {'likelihood': 1, 'entropy': 100, 'regularization': 0.00001, 'failure': 0}
        self.scaling = scaling
        self.use_new_cost = False

    def create_network(self, scope='mdnmp', nn_type='v1'):
        tf.compat.v1.reset_default_graph()
        self.input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d_input], name='input')
        self.target = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d_output], name='target')
        self.is_positive = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='is_positive')
        self.outputs = getattr(basic_nn, 'mdn_nn_'+nn_type)(self.input, self.d_output * self.n_comps, self.n_comps,
                                                            self.nn_structure, self.using_batch_norm, scope=scope)

    def build_mdn(self, learning_rate=0.001, nn_type='v1'):
        self.create_network(nn_type=nn_type)
        var_list = [v for v in tf.compat.v1.trainable_variables()]
        mean_var_list = [v for v in tf.compat.v1.trainable_variables() if 'scale' not in v.name]
        scale_var_list = [v for v in tf.compat.v1.trainable_variables() if 'mean' not in v.name]

        reg_loss = [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()]
        reg_loss = self.lratio['regularization'] * tf.reduce_sum(reg_loss) / len(var_list)

        mean = self.outputs['mean']
        scale = self.outputs['scale']
        mc = self.outputs['mc']

        nll = gmm_nll_cost(self.target, mean, scale, mc, self.is_positive)
        entropy_loss = model_entropy_cost(self.n_comps, mc, self.is_positive, eps=1e-20)
        floss = failure_cost(self.target, mean, mc, 1-self.is_positive, neg_scale=0.1)

        cost = self.lratio['likelihood'] * nll + \
               self.lratio['entropy'] * entropy_loss + \
               self.lratio['regularization'] * reg_loss + \
               self.lratio['failure'] * floss

        self.opt_all = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=var_list)
        self.opt_mean = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=mean_var_list)
        self.opt_scale = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=scale_var_list)
        self.saver = tf.compat.v1.train.Saver()

        self.loss_dict = {'nll': nll, 'eloss': entropy_loss, 'floss': floss, 'cost': cost}

    def init_train(self, logfile='mdnmp_log'):
        tf.compat.v1.random.set_random_seed(self.seed)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.writer = tf.compat.v1.summary.FileWriter(logfile, self.sess.graph)

    def train(self, input, target, is_positive, max_epochs=1000, is_load=True, is_save=True, checkpoint_dir='mdnmp_checkpoint',
              model_dir='mdnmp_model', model_name='mdnmp'):
        self.global_step = 1
        if is_load:
            could_load, checkpoint_counter = self.load(self.sess, self.saver, checkpoint_dir, model_dir)
            if could_load:
                self.global_step = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for i in range(max_epochs):
            if self.batch_size is not None:
                idx = self.next_batch(np.shape(input)[0])
                batch_input = input[idx,:]
                batch_target = target[idx,:]
                batch_ispos = is_positive[idx,:]
            else:
                batch_input = input
                batch_target = target
                batch_ispos = is_positive

            feed_dict = {self.input: batch_input, self.target: batch_target, self.is_positive: batch_ispos}
            _, nll, eloss, cost, floss = self.sess.run([self.opt_all, self.loss_dict['nll'], self.loss_dict['eloss'],
                                                        self.loss_dict['cost'], self.loss_dict['floss']],
                                                        feed_dict=feed_dict)

            if i != 0 and i % 1000 == 0 and is_save:
                self.save(self.sess, self.saver, checkpoint_dir, model_dir, model_name)
                self.global_step = self.global_step + 1

            print("epoch: %1d, cost: %.3f, nll: %.3f, entropy_loss: %.3f, floss: %.3f" % (i, cost, nll, eloss, floss), end='\r', flush=True)

        print("Training Result: %1d, cost: %.3f, nll: %.3f, entropy_loss: %.3f, floss: %.3f" % (i, cost, nll, eloss, floss), end='\n')

    def predict(self, cinput, n_samples=1):
        mean, scale, mc = self.sess.run([self.outputs['mean'], self.outputs['scale'], self.outputs['mc']],
                                        feed_dict={self.input: cinput})

        n_data = np.shape(cinput)[0]

        scales = np.expand_dims(scale, axis=0)
        scales = np.reshape(scales,newshape=(n_data, self.d_output, self.n_comps), order='F')
        means = np.expand_dims(mean, axis=0)
        means = np.reshape(means,newshape=(n_data, self.d_output, self.n_comps), order='F')

        scales = np.transpose(scales, (0,2,1))
        means = np.transpose(means, (0,2,1))

        out = np.zeros(shape=(n_data, n_samples, self.d_output))
        idx = np.zeros(shape=(n_data, n_samples))
        for i in range(np.shape(means)[0]):
            out[i, :, :], idx[i, :] = sample_gmm(n_samples=n_samples, means=means[i, :, :],
                                                 scales=scales[i, :, :] * self.scaling, mixing_coeffs=mc[i, :])

        outdict = {'samples': out, 'compIDs': idx, 'mean': mean, 'scale': scale, 'mc': mc}
        return out, outdict
