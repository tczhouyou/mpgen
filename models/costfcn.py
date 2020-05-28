import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
# import tensorflow_addons as tfa


def gmm_nll_cost(samples, vec_mus, vec_scales, mixing_coeffs, sample_valid):
    n_comp = mixing_coeffs.get_shape().as_list()[1]
    mus = tf.split(vec_mus, num_or_size_splits=n_comp, axis=1)
    scales = tf.split(vec_scales, num_or_size_splits=n_comp, axis=1)
    gmm_comps = [tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale) for mu, scale in zip(mus, scales)]
    gmm = tfd.Mixture(cat=tfd.Categorical(probs=mixing_coeffs), components=gmm_comps)
    loss = gmm.log_prob(samples)
    loss = tf.negative(tf.reduce_sum(tf.multiply(loss, sample_valid)))
    loss = tf.divide(loss, tf.reduce_sum(sample_valid))
    return loss


def model_entropy_cost(n_comps, mixing_coeffs, sample_valid, eps=1e-5):
    ratio = 1.0 / np.float(n_comps)
    max_entropy = - np.log(ratio)

    pos_mc = tf.multiply(mixing_coeffs, tf.tile(sample_valid, [1, n_comps]))
    prob = tf.divide(tf.reduce_sum(pos_mc, axis=0), tf.reduce_sum(sample_valid, axis=0))
    model_entropy = - tf.reduce_sum(tf.multiply(tf.math.log(prob + eps), prob))
    loss = max_entropy - model_entropy
    return loss


def failure_cost(samples, vec_mus, mixing_coeffs, sample_invalid, neg_scale=0.1):
    n_comp = mixing_coeffs.get_shape().as_list()[1]
    n_dim = samples.get_shape().as_list()[1]
    mus = tf.split(vec_mus, num_or_size_splits=n_comp, axis=1)
    smat = tf.ones(shape=(1,n_dim)) * neg_scale
    gmm_comps = [tfd.MultivariateNormalDiag(loc=mu, scale_diag=smat) for mu in mus]
    gmm = tfd.Mixture(cat=tfd.Categorical(probs=mixing_coeffs), components=gmm_comps)
    loss = gmm.log_prob(samples)
    loss = tf.reduce_sum(tf.multiply(loss, sample_invalid))
    return loss

# simplex cost refers to "Mixture Density Generative Adversarial Networks" Hamid et al. 2020
def simplex_coordinates( m ):
    # This function is adopted from the Simplex Coordinates library
    # https://people.sc.fsu.edu/~jburkardt/py_src/simplex_coordinates/simplex_coordinates.html
    x = np.zeros ( [ m, m + 1 ], dtype=np.float32)

    for j in range ( 0, m ):
        x[j,j] = 1.0

    a = ( 1.0 - np.sqrt ( float ( 1 + m ) ) ) / float ( m )

    for i in range ( 0, m ):
        x[i,m] = a
    c = np.zeros ( m )
    for i in range ( 0, m ):
        s = 0.0
        for j in range ( 0, m + 1 ):
            s = s + x[i,j]
        c[i] = s / float ( m + 1 )

    for j in range ( 0, m + 1 ):
        for i in range ( 0, m ):
            x[i,j] = x[i,j] - c[i]
    s = 0.0
    for i in range ( 0, m ):
        s = s + x[i,0] ** 2
        s = np.sqrt ( s )

    for j in range ( 0, m + 1 ):
        for i in range ( 0, m ):
            x[i,j] = x[i,j] / s

    ves = []
    for j in range(m+1):
        ves.append(x[:,j])

    return ves

# simplex cost refers to "Mixture Density Generative Adversarial Networks" Hamid et al. 2020
def gmm_likelihood_simplex(samples, odim):
    mu_s = simplex_coordinates(odim)
    scale = np.ones(odim, dtype=np.float32) * .25
    ngmm = odim + 1
    mixing_coeffs = np.ones(ngmm, dtype=np.float32) / ngmm
    gmm_comps = [tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale) for mu in mu_s]
    gmm = tfd.Mixture(cat=tfd.Categorical(probs=mixing_coeffs), components=gmm_comps)
    loss = gmm.prob(samples)
    return loss, gmm


# this cost refers to "DIVERSITY-SENSITIVE CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS" Dingdong et al. 2019
# def ds_generator_cost(noise, fake_samples):
#     loss_mat = tfa.losses.metric_learning.pairwise_distance(fake_samples)
#     noise_mat = tfa.losses.metric_learning.pairwise_distance(noise)
#     # loss = tf.

def entropy_discriminator_cost(simplex_gmm, d_real_output):
    gmm_comps = simplex_gmm.components
    data_probs = []
    for i in range(len(gmm_comps)):
        gmm_comp = gmm_comps[i]
        data_probs.append(gmm_comp.prob(d_real_output))

    prob_mat = tf.stack(data_probs, axis=1)
    prob_sum = tf.reduce_sum(prob_mat, axis=1)
    prob_sum = tf.expand_dims(prob_sum, axis=1)
    prob_sum = tf.tile(prob_sum, multiples=tf.constant([1, len(gmm_comps)], tf.int32))
    prob_mat = tf.math.divide(prob_mat, prob_sum)
    mixing_coeffs = tf.reduce_mean(prob_mat, axis=0)
    neg_entropy = tf.reduce_sum(tf.multiply(tf.math.log(mixing_coeffs + 1e-8), mixing_coeffs))
    return neg_entropy