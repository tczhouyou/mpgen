import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import basic_nn
from basic_nn import fully_connected_nn, sigmoid_act, tanh_act, leaky_relu_act
from basic_model import basicModel
from costfcn import gmm_likelihood_simplex, entropy_discriminator_cost, gmm_nll_cost, gmm_mce_cost
from tensorflow_probability import distributions as tfd
from util import sample_gmm


tf.compat.v1.disable_eager_execution()
if tf.__version__ < '2.0.0':
    import tflearn
    w_init = tflearn.initializations.normal(stddev=0.003, seed=42)
    w_init_dis = tflearn.initializations.normal(stddev=0.1, seed=42)
else:
    from tensorflow.keras import initializers
    w_init = initializers.RandomNormal(stddev=0.003, seed=42)
    w_init_dis = initializers.RandomNormal(stddev=0.1, seed=42)


class GMGAN:
    def __init__(self, n_comps, context_dim, response_dim, nn_structure,
                 batch_size=None, using_batch_norm=False, seed=42, eps=1e-20,
                 gen_sup_lrate=0.001, gen_adv_lrate=0.001, dis_learning_rate=0.001, entropy_ratio=0, scaling=1, var_init=None, var_init_dis=None):
        basicModel.__init__(self, batch_size, using_batch_norm, seed, eps)
        self.n_comps = n_comps
        self.context_dim = context_dim
        self.response_dim = response_dim
        self.latent_dim = n_comps-1

        self.nn_structure = nn_structure
        self.gen_sup_lrate = gen_sup_lrate
        self.gen_adv_lrate = gen_adv_lrate

        self.dis_lrate = dis_learning_rate
        self.entropy_ratio = entropy_ratio
        self.using_batch_norm = using_batch_norm
        self.scaling = scaling
        self.lratio = {'likelihood': 1, 'entropy': 0, 'adv_cost':200}
        self.sup_max_epoch = 0
        self.outputs = {}

        if var_init is None:
            self.var_init = w_init
        else:
            self.var_init = var_init

        if var_init_dis is None:
            self.var_init_dis = w_init_dis
        else:
            self.var_init_dis = var_init_dis

    def get_gmm(self, vec_mus, vec_scales, mixing_coeffs):
        n_comp = mixing_coeffs.get_shape().as_list()[1]
        mus = tf.split(vec_mus, num_or_size_splits=n_comp, axis=1)
        scales = tf.split(vec_scales, num_or_size_splits=n_comp, axis=1)
        gmm_comps = [tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale) for mu, scale in zip(mus, scales)]
        gmm = tfd.Mixture(cat=tfd.Categorical(probs=mixing_coeffs), components=gmm_comps)
        return gmm

    def generator(self, context, nn_type='v1'):
        g_outs = getattr(basic_nn, 'mdn_nn_' + nn_type)(context, self.response_dim * self.n_comps, self.n_comps,
                                                              self.nn_structure, self.using_batch_norm,
                                                              scope='generator', var_init=self.var_init)

        mean = g_outs['mean']
        scale = g_outs['scale']
        mc = g_outs['mc']
        gmm = self.get_gmm(mean, scale, mc)

        return g_outs, gmm.sample()

    def discriminator(self, context, response):
        d_hidden_response = fully_connected_nn(response, self.nn_structure['d_response'][:-1],
                                                    self.nn_structure['d_response'][-1], w_init=self.var_init_dis,
                                                    latent_activation=leaky_relu_act,
                                                    out_activation=None, scope='discriminator_response')

        d_hidden_context = fully_connected_nn(context, self.nn_structure['d_context'][:-1],
                                                   self.nn_structure['d_context'][-1], w_init=self.var_init_dis,
                                                   latent_activation=leaky_relu_act,
                                                   out_activation=None, scope='discriminator_context')

        d_hidden_input = tf.concat([d_hidden_response, d_hidden_context], axis=1)
        d_output = fully_connected_nn(d_hidden_input, self.nn_structure['discriminator'], self.latent_dim,
                                           w_init=self.var_init_dis,
                                           latent_activation=leaky_relu_act, out_activation=sigmoid_act,
                                           scope='discriminator')

        d_output = d_output * 5 - 2.5
        return d_output


    def create_lambda_network(self):
        self.lambda_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='lambda_input')
        self.lambda_output = fully_connected_nn(self.lambda_input, self.nn_structure['lambda'], self.latent_dim, w_init=self.var_init,
                                                latent_activation=leaky_relu_act, out_activation=sigmoid_act, scope='lambda')
        self.lambda_output = self.lambda_output * 5 - 2.5
        self.lambval, _ = gmm_likelihood_simplex(self.lambda_output, self.latent_dim)
        self.lamb_cost = tf.negative(tf.math.log(1e-8 + tf.reduce_sum(self.lambval)))
        self.lamb_vars = [v for v in tf.compat.v1.trainable_variables() if 'lambda' in v.name]
        self.lamb_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.lamb_cost,
                                                                                       var_list=self.lamb_vars)

    def create_network(self):
        self.context = tf.compat.v1.placeholder(tf.float32, shape=(None, self.context_dim), name='context')
        self.real_context = tf.compat.v1.placeholder(tf.float32, shape=(None, self.context_dim),name='real_context')
        self.real_response = tf.compat.v1.placeholder(tf.float32, shape=(None, self.response_dim),name='real_response')

        num_real_data = tf.shape(self.real_context)[0]
        all_context = tf.concat([self.real_context, self.context], axis=0)
        self.g_outs, self.all_response = self.generator(all_context)

        self.g_response = self.all_response[num_real_data:,:]
        self.response = self.all_response[num_real_data:, :]

        d_context = tf.concat([self.real_context, self.context], axis=0)
        d_response = tf.concat([self.real_response, self.response], axis=0)
        self.d_output = self.discriminator(d_context, d_response)

        self.d_real_output = self.d_output[:num_real_data,:]
        self.d_fake_output = self.d_output[num_real_data:,:]
        self.create_lambda_network()

        self.gen_vars = [v for v in tf.compat.v1.trainable_variables() if 'generator' in v.name]
        self.dis_vars = [v for v in tf.compat.v1.trainable_variables() if 'discriminator' in v.name]

        self.lamb = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='lambda')
        self.fake_likelihood, _ = gmm_likelihood_simplex(self.d_fake_output, self.latent_dim)
        self.real_likelihood, simplex_gmm = gmm_likelihood_simplex(self.d_real_output, self.latent_dim)

        # get generator costs: nll cost + adversarial cost
        mean = self.g_outs['mean'][:num_real_data,:]
        scale = self.g_outs['scale'][:num_real_data,:]
        mc = self.g_outs['mc'][:num_real_data,:]

        self.outputs['mean'] = self.g_outs['mean'][num_real_data:,:]
        self.outputs['scale'] = self.g_outs['scale'][num_real_data:,:]
        self.outputs['mc'] = self.g_outs['mc'][num_real_data:,:]

        is_positive = tf.ones(shape=(num_real_data, 1), dtype=tf.float32)
        self.nll = gmm_nll_cost(self.real_response, mean, scale, mc, is_positive)
        self.entropy_loss = gmm_mce_cost(mc, is_positive, eps=1e-20)
        self.gen_sup_cost = self.lratio['likelihood'] * self.nll + self.lratio['entropy'] * self.entropy_loss

        self.gen_adv_cost = tf.reduce_mean(tf.math.log(self.lamb - self.fake_likelihood + 1e-8))
        self.gen_cost = self.gen_sup_cost + self.lratio['adv_cost'] * self.gen_adv_cost

        # get discriminator costs
        self.dis_real_cost = tf.negative(tf.reduce_mean(tf.math.log(self.real_likelihood + 1e-8)))
        self.dis_fake_cost = tf.negative(tf.reduce_mean(tf.math.log(self.lamb - self.fake_likelihood + 1e-8)))
        self.dis_cost = self.dis_real_cost + self.dis_fake_cost
        self.entropy_cost = entropy_discriminator_cost(simplex_gmm, self.d_real_output)
        self.dis_cost = self.dis_cost + self.entropy_ratio * self.entropy_cost
        self.real_likelihood_mean = tf.reduce_mean(self.real_likelihood)
        self.fake_likelihood_mean = tf.reduce_mean(self.fake_likelihood)

        self.gen_adv_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.gen_adv_lrate).minimize(self.gen_cost, var_list=self.gen_vars)
        self.gen_sup_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.gen_sup_lrate).minimize(self.gen_sup_cost, var_list=self.gen_vars)
        self.dis_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.dis_lrate).minimize(self.dis_cost, var_list=self.dis_vars)

        self.reg_loss = [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()]


    def init_train(self,logfile='gmgan.log'):
        tf.compat.v1.random.set_random_seed(self.seed)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.writer = tf.compat.v1.summary.FileWriter(logfile, self.sess.graph)

    def train(self, train_context, real_context, real_response, max_epochs=1000, lambda_max_epochs=1000, is_load=True, is_save=True, checkpoint_dir='gmgan_checkpoint',
              model_dir='gmgan_model', model_name='gmgan'):
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

            feed_dict = {self.context: batch_input,
                         self.lamb: lamb, self.real_context: real_context, self.real_response: real_response}
            if i < self.sup_max_epoch:
                self.sess.run(self.gen_sup_opt, feed_dict=feed_dict)
            else:
                self.sess.run(self.gen_adv_opt, feed_dict=feed_dict)

            _, dis_cost, rlm, flm, ecost, gsup, gadv = \
                self.sess.run([self.dis_opt, self.dis_cost,
                               self.real_likelihood_mean, self.fake_likelihood_mean,
                               self.entropy_cost, self.gen_sup_cost, self.gen_adv_cost], feed_dict=feed_dict)

            if i != 0 and i % 1000 == 0 and is_save:
                self.save(self.sess, self.saver, checkpoint_dir, model_dir, model_name)
                self.global_step = self.global_step + 1

            print("epoch: %1d, gen_sup_cost: %.3f, gen_adv_cost: %.3f, dis_cost: %.3f, real_liklihood: %.3f, "
                  "fake_likelihood: %.3f, entropy_cost: %.3f" %
                  (i, gsup, gadv, dis_cost, rlm, flm, ecost), end='\r', flush=True)

        print("epoch: %1d, gen_sup_cost: %.3f, gen_adv_cost: %.3f, dis_cost: %.3f, real_liklihood: %.3f, "
              "fake_likelihood: %.3f, entropy_cost: %.3f" %
              (i, gsup, gadv, dis_cost, rlm, flm, ecost), end='\n')

    def predict(self, cinput, n_samples=1):
        rinput = np.random.uniform(low=np.min(cinput, axis=0), high=np.max(cinput, axis=0),size=(1, np.shape(cinput)[1]))
        mean, scale, mc = self.sess.run([self.outputs['mean'], self.outputs['scale'], self.outputs['mc']],
                                        feed_dict={self.context: cinput, self.real_context:rinput})

        n_data = np.shape(cinput)[0]

        scales = np.expand_dims(scale, axis=0)
        scales = np.reshape(scales, newshape=(n_data, self.response_dim, self.n_comps), order='F')
        means = np.expand_dims(mean, axis=0)
        means = np.reshape(means, newshape=(n_data, self.response_dim, self.n_comps), order='F')

        scales = np.transpose(scales, (0, 2, 1))
        means = np.transpose(means, (0, 2, 1))

        out = np.zeros(shape=(n_data, n_samples, self.response_dim))
        idx = np.zeros(shape=(n_data, n_samples))
        for i in range(np.shape(means)[0]):
            out[i, :, :], idx[i, :] = sample_gmm(n_samples=n_samples, means=means[i, :, :],
                                                 scales=scales[i, :, :] * self.scaling, mixing_coeffs=mc[i, :])

        outdict = {'samples': out, 'compIDs': idx, 'mean': mean, 'scale': scale, 'mc': mc}
        return out, outdict


    def generate(self, cinput, n_samples=100, n_output=1):
        rinput = np.random.uniform(low=np.min(cinput, axis=0), high=np.max(cinput, axis=0),
                                   size=(1, np.shape(cinput)[1]))
        mean, scale, mc = self.sess.run([self.outputs['mean'], self.outputs['scale'], self.outputs['mc']],
                                        feed_dict={self.context: cinput, self.real_context: rinput})

        n_data = np.shape(cinput)[0]

        scales = np.expand_dims(scale, axis=0)
        scales = np.reshape(scales, newshape=(n_data, self.response_dim, self.n_comps), order='F')
        means = np.expand_dims(mean, axis=0)
        means = np.reshape(means, newshape=(n_data, self.response_dim, self.n_comps), order='F')
        scales = np.transpose(scales, (0, 2, 1))
        means = np.transpose(means, (0, 2, 1))

        n_data = np.shape(cinput)[0]
        res = np.zeros(shape=(n_data, n_output, self.response_dim), dtype=np.float32)
        for i in range(n_data):
            souts, _ = sample_gmm(n_samples=n_samples, means=means[i, :, :], scales=scales[i, :, :] * self.scaling, mixing_coeffs=mc[i, :])
            sinputs = np.tile(cinput[i,:], [n_samples,1])
            k0 = 0
            real_likelihood = np.zeros(n_samples)
            while k0+100 < n_samples:
                rlikelihood = self.sess.run(self.real_likelihood, feed_dict={self.context:sinputs[k0:k0+100,:],
                                                                                 self.real_context:sinputs[k0:k0+100,:],
                                                                                 self.real_response:souts[k0:k0+100,:]})
                real_likelihood[k0:k0+100] = rlikelihood
                k0 = k0 + 100

            ind = np.argsort(-np.array(real_likelihood), axis=-1)
            res[i,:,:] = souts[ind[:n_output],:]

        return res

    def next_batch(self, size):
        idx = np.arange(0, size)
        np.random.shuffle(idx)
        idx = idx[:self.batch_size]

        return idx
