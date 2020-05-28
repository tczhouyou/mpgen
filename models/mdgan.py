import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
from basic_nn import fully_connected_nn
from basic_model import basicModel
from costfcn import gmm_likelihood_simplex

tf.compat.v1.disable_eager_execution()

class cMDGAN(basicModel):
    def __init__(self, n_comps, context_dim, response_dim, noise_dim, nn_structure,
                 batch_size=None, using_batch_norm=False, seed=42, eps=1e-20,
                 gen_learning_rate=0.001, dis_learning_rate=0.001):
        basicModel.__init__(self, batch_size, using_batch_norm, seed, eps)
        self.n_comps = n_comps
        self.context_dim = context_dim
        self.response_dim = response_dim
        self.latent_dim = n_comps-1
        self.noise_dim = noise_dim

        self.nn_structure = nn_structure
        self.gen_lrate = gen_learning_rate
        self.dis_lrate = dis_learning_rate

    def create_generator(self):
        self.g_input = tf.concat([self.context, self.noise], axis=1)
        self.response = fully_connected_nn(self.g_input, self.nn_structure['generator'], self.response_dim,
                                                latent_activation=tf.nn.leaky_relu, scope='generator')
        self.gen_vars = [v for v in tf.compat.v1.trainable_variables() if 'generator' in v.name]

    def create_simple_discriminator(self):
        self.d_fake_input = tf.concat([self.response, self.context], axis=1)
        self.d_real_input = tf.concat([self.real_response, self.real_context], axis=1)
        self.d_input = tf.concat([self.d_real_input, self.d_fake_input], axis=0)
        self.d_output = fully_connected_nn(self.d_input, self.nn_structure['discriminator'], self.latent_dim,
                                           latent_activation=tf.nn.leaky_relu, scope='discriminator')

        self.dis_vars = [v for v in tf.compat.v1.trainable_variables() if 'discriminator' in v.name]

    def create_discriminator(self):
        self.d_response = tf.concat([self.real_response, self.response], axis=0)
        self.d_hidden_response = fully_connected_nn(self.d_response, self.nn_structure['d_response'][:-1],
                                                    self.nn_structure['d_response'][-1], latent_activation=tf.nn.leaky_relu,
                                                    scope='discriminator_response')

        self.d_context = tf.concat([self.real_context, self.context], axis=0)
        self.d_hidden_context = fully_connected_nn(self.d_context, self.nn_structure['d_context'][:-1],
                                                   self.nn_structure['d_context'][-1], latent_activation=tf.nn.leaky_relu,
                                                   scope='discriminator_context')

        self.d_hidden_input = tf.concat([self.d_hidden_response, self.d_hidden_context], axis=1)
        self.d_output = fully_connected_nn(self.d_hidden_input, self.nn_structure['discriminator'], self.latent_dim,
                                           latent_activation=tf.nn.leaky_relu, scope='discriminator')

        self.dis_vars = [v for v in tf.compat.v1.trainable_variables() if 'discriminator' in v.name]

    def create_lambda_network(self):
        self.lambda_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='lambda_input')
        self.lambda_output = fully_connected_nn(self.lambda_input, self.nn_structure['lambda'], self.latent_dim,
                                                latent_activation=tf.nn.leaky_relu, scope='lambda')
        self.lambval, _ = gmm_likelihood_simplex(self.lambda_output, self.latent_dim)
        self.lamb_cost = tf.negative(tf.math.log(1e-8 + tf.reduce_sum(self.lambval)))
        self.lamb_vars = [v for v in tf.compat.v1.trainable_variables() if 'lambda' in v.name]
        self.lamb_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.lamb_cost,
                                                                                       var_list=self.lamb_vars)


    def create_network(self, num_real_data):
        tf.compat.v1.reset_default_graph()
        self.context = tf.compat.v1.placeholder(tf.float32, shape=(None, self.context_dim), name='context')
        self.noise = tf.compat.v1.placeholder(tf.float32, shape=(None, self.noise_dim), name='noise')
        self.real_context = tf.compat.v1.placeholder(tf.float32, shape=(num_real_data, self.context_dim), name='real_context')
        self.real_response = tf.compat.v1.placeholder(tf.float32, shape=(num_real_data, self.response_dim), name='real_response')

        # create generator
        self.create_generator()

        # create discriminator
        if 'd_response' in self.nn_structure and 'd_context' in self.nn_structure:
            self.create_discriminator()
        else:
            self.create_simple_discriminator()

        self.d_real_output = self.d_output[:num_real_data,:]
        self.d_fake_output = self.d_output[num_real_data:,:]

        self.create_lambda_network()

        self.lamb = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='lambda')
        self.fake_likelihood, _ = gmm_likelihood_simplex(self.d_fake_output, self.latent_dim)
        self.real_likelihood, simplex_gmm = gmm_likelihood_simplex(self.d_real_output, self.latent_dim)

        gen_cost_logit = self.lamb - self.fake_likelihood + 1e-8
        dis_cost_logit = self.real_likelihood + 1e-8
        self.gen_cost = tf.reduce_mean(tf.math.log(gen_cost_logit))
        self.dis_cost = tf.negative(tf.reduce_mean(tf.math.log(dis_cost_logit))
                                    + tf.reduce_mean(tf.math.log(gen_cost_logit)))

        self.real_likelihood_mean = tf.reduce_mean(self.real_likelihood)
        self.fake_likelihood_mean = tf.reduce_mean(self.fake_likelihood)

        # beta1=0.5, see https://github.com/eghbalz/mdgan/blob/master/mdgan.ipynb
        self.gen_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.gen_lrate, beta1=0.5).minimize(self.gen_cost, var_list=self.gen_vars)
        self.dis_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.dis_lrate, beta1=0.5).minimize(self.dis_cost, var_list=self.dis_vars)

    def init_train(self,logfile='mdgan.log'):
        tf.compat.v1.random.set_random_seed(self.seed)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.writer = tf.compat.v1.summary.FileWriter(logfile, self.sess.graph)

    def train(self, train_context, real_context, real_response, max_epochs=1000, lambda_max_epochs=1000, is_load=True, is_save=True, checkpoint_dir='mdgan_checkpoint',
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
                idx = self.next_batch(np.shape(train_context)[0])
                batch_input = train_context[idx, :]
            else:
                batch_input = train_context

            n_data = np.shape(batch_input)[0]
            batch_rand_input = np.random.uniform(low=0, high=1, size=(n_data, self.noise_dim)).astype(np.float32)

            feed_dict = {self.context: batch_input, self.noise: batch_rand_input,
                         self.lamb: lamb, self.real_context: real_context, self.real_response: real_response}

            _, gen_cost = self.sess.run([self.gen_opt, self.gen_cost], feed_dict=feed_dict)
            _, dis_cost, rlm, flm = self.sess.run([self.dis_opt, self.dis_cost, self.real_likelihood_mean, self.fake_likelihood_mean], feed_dict=feed_dict)

            if i != 0 and i % 1000 == 0 and is_save:
                self.save(self.sess, self.saver, checkpoint_dir, model_dir, model_name)
                self.global_step = self.global_step + 1

            print("epoch: %1d, gen_cost: %.3f, dis_cost: %.3f, real_liklihood: %.3f, fake_likelihood: %.3f" % (i, gen_cost, dis_cost, rlm, flm), end='\r', flush=True)

        print("epoch: %1d, gen_cost: %.3f, dis_cost: %.3f, real_liklihood: %.3f, fake_likelihood: %.3f" % (max_epochs, gen_cost, dis_cost, rlm, flm), end='\n')


    def generate(self, context):
        n_data = np.shape(context)[0]
        noise =  np.random.uniform(low=0, high=1, size=(n_data, self.noise_dim)).astype(np.float32)
        feed_dict = {self.context: context, self.noise: noise}
        out = self.sess.run(self.response, feed_dict=feed_dict)
        return out

    def generate_multi(self, context, n_sample):
        n_data = np.shape(context)[0]
        out = np.zeros(shape=(n_data, n_sample, self.response_dim))

        for i in range(n_sample):
            noise = np.random.uniform(low=0, high=1, size=(n_data, self.noise_dim)).astype(np.float32)
            feed_dict = {self.context: context, self.noise: noise}
            out[:,i,:] = self.sess.run(self.response, feed_dict=feed_dict)

        return out

