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
from costfcn import gmm_likelihood_simplex

tf.compat.v1.disable_eager_execution()




class cMDGAN(basicModel):
    def __init__(self, n_comps, input_dim, out_dim, rand_input_dim, nn_structure,
                 batch_size=None, using_batch_norm=False, seed=42, eps=1e-20, scaling=0.1, gen_learning_rate=0.001, dis_learning_rate=0.001):
        basicModel.__init__(self, batch_size, using_batch_norm, seed, eps)
        self.n_comps = n_comps
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.d_out_dim = n_comps-1
        self.rand_input_dim = rand_input_dim

        self.nn_structure = nn_structure
        self.scaling = scaling
        self.gen_lrate = gen_learning_rate
        self.dis_lrate = dis_learning_rate

    def create_network(self, num_real_data):
        tf.compat.v1.reset_default_graph()
        self.data_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.input_dim), name='input')
        self.rand_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.rand_input_dim), name='rand_input')
        self.d_real_input = tf.compat.v1.placeholder(tf.float32, shape=(num_real_data, self.out_dim + self.input_dim), name='real')

        self.g_input = tf.concat([self.data_input, self.rand_input], axis=1)
        self.fake = fully_connected_nn(self.g_input, self.nn_structure['generator'], self.out_dim,
                                       latent_activation=tf.nn.leaky_relu, scope='generator')
        self.d_fake_input = tf.concat([self.fake, self.data_input], axis=1)

        self.d_input = tf.concat([self.d_real_input, self.d_fake_input], axis=0)
        self.d_output = fully_connected_nn(self.d_input, self.nn_structure['discriminator'], self.d_out_dim,
                                           latent_activation=tf.nn.leaky_relu, scope='discriminator')

        self.d_real_output = self.d_output[:num_real_data,:]
        self.d_fake_output = self.d_output[num_real_data:,:]


        self.lambda_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='lambda_input')
        self.lambda_output = fully_connected_nn(self.lambda_input, self.nn_structure['lambda'], self.d_out_dim,
                                                latent_activation=tf.nn.leaky_relu, scope='lambda')
        self.lambval = gmm_likelihood_simplex(self.lambda_output, self.d_out_dim)
        self.lamb_cost = tf.negative(tf.log(1e-8 + tf.reduce_sum(self.lambval)))
        self.lamb_vars = [v for v in tf.compat.v1.trainable_variables() if 'lambda' in v.name]
        self.lamb_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.lamb_cost, var_list=self.lamb_vars)


        self.lamb = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='lambda')
        self.fake_likelihood = gmm_likelihood_simplex(self.d_fake_output, self.d_out_dim)
        self.gen_cost = tf.reduce_mean(tf.log(1e-8 + self.lamb - self.fake_likelihood))

        self.real_likelihood = gmm_likelihood_simplex(self.d_real_output, self.d_out_dim)
        self.dis_cost = tf.negative(tf.reduce_mean(tf.log(1e-8 + self.real_likelihood))
                                    + tf.reduce_mean(tf.log(1e-8 + self.lamb - self.fake_likelihood)))

        self.gen_vars = [v for v in tf.compat.v1.trainable_variables() if 'generator' in v.name]
        self.gen_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.gen_lrate).minimize(self.gen_cost, var_list=self.gen_vars)
        self.dis_vars = [v for v in tf.compat.v1.trainable_variables() if 'discriminator' in v.name]
        self.dis_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.dis_lrate).minimize(self.dis_cost, var_list=self.dis_vars)

    def init_train(self, logfile='mdgan_log'):
        tf.compat.v1.random.set_random_seed(self.seed)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.writer = tf.compat.v1.summary.FileWriter(logfile, self.sess.graph)

    def train(self, input, real_data, max_epochs=1000, lambda_max_epochs=1000, is_load=True, is_save=True, checkpoint_dir='mdgan_checkpoint',
              model_dir='mdgan_model', model_name='mdgan'):
        self.global_step = 1
        if is_load:
            could_load, checkpoint_counter = self.load(self.sess, self.saver, checkpoint_dir, model_dir)
            if could_load:
                self.global_step = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for i in range(lambda_max_epochs):
            feed_dict = {self.lambda_input: np.ones(shape=(1,1))}
            _, lamb_cost, lamb = self.sess.run([self.lamb_opt, self.lamb_cost, self.lambval], feed_dict=feed_dict)
            print("epoch: %1d, cost: %.3f, lamb: %.3f" % (i, lamb_cost, lamb), end='\r', flush=True)

        print("epoch: %1d, cost: %.3f, lamb: %.3f" % (i, lamb_cost, lamb), end='\n')

        lamb = np.expand_dims(lamb, axis=0)

        for i in range(max_epochs):
            if self.batch_size is not None:
                idx = self.next_batch(np.shape(input)[0])
                batch_input = input[idx, :]
            else:
                batch_input = input

            n_data = np.shape(batch_input)[0]
            batch_rand_input = np.random.normal(size=(n_data, self.rand_input_dim), scale=0.001)

            feed_dict = {self.data_input: batch_input, self.rand_input: batch_rand_input,
                         self.lamb: lamb, self.d_real_input: real_data}

            _, gen_cost = self.sess.run([self.gen_opt, self.gen_cost], feed_dict=feed_dict)
            _, dis_cost = self.sess.run([self.dis_opt, self.dis_cost], feed_dict=feed_dict)

            if i != 0 and i % 1000 == 0 and is_save:
                self.save(self.sess, self.saver, checkpoint_dir, model_dir, model_name)
                self.global_step = self.global_step + 1

            print("epoch: %1d, gen_cost: %.3f, dis_cost: %.3f" % (i, gen_cost, dis_cost), end='\r', flush=True)


    def generate(self, input, rand_input):
        feed_dict = {self.data_input: input, self.rand_input: rand_input}
        out = self.sess.run(self.fake, feed_dict=feed_dict)
        return out
