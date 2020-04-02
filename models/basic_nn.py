import tensorflow as tf
import tflearn


def fully_connected_nn(inputs, layer_dims, out_dim, latent_activation=tflearn.activations.relu, out_activation=None,
         w_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003, seed=42), scope=None, is_batch_norm=False):
    with tf.compat.v1.variable_scope(scope or 'fcnn', reuse=tf.compat.v1.AUTO_REUSE):
        layer = inputs

        for i in range(len(layer_dims)):
            layer = tflearn.fully_connected(incoming=layer, n_units=layer_dims[i], scope='layer_' + str(i))
            if is_batch_norm:
                layer = tflearn.layers.normalization.batch_normalization(incoming=layer, scope='layer_' + str(i)+'_normalization')

            layer = latent_activation(layer)

        out = tflearn.fully_connected(incoming=layer, n_units=out_dim, weights_init=w_init)

        if out_activation is not None:
            out = out_activation(out)

        return out


def mdn_nn_v1(inputs, d_outputs, n_comps, nn_structure, using_batch_norm=False, scope='mdn_nn'):
    d_feat = nn_structure['d_feat']
    feat_layers = nn_structure['feat_layers']
    mean_layers = nn_structure['mean_layers']
    scale_layers = nn_structure['scale_layers']
    mixing_layers = nn_structure['mixing_layers']

    var_init = tflearn.initializations.uniform(minval=-.003, maxval=.003)

    feats = fully_connected_nn(inputs,  feat_layers,  d_feat, scope=scope + "_feat", w_init=var_init,
                               latent_activation=tflearn.activations.leaky_relu,
                               out_activation=tflearn.activations.leaky_relu, is_batch_norm=using_batch_norm)

    mean = fully_connected_nn(feats, mean_layers, d_outputs, scope=scope + '_mean',
                              w_init=var_init,
                              latent_activation=tflearn.activations.leaky_relu, out_activation=None,
                              is_batch_norm=using_batch_norm)

    scale = fully_connected_nn(feats, scale_layers, d_outputs, scope=scope + '_scale',
                               w_init=var_init,
                               latent_activation=tflearn.activations.leaky_relu, out_activation=None,
                               is_batch_norm=using_batch_norm)
    scale = tf.exp(scale)

    mc = fully_connected_nn(feats, mixing_layers, n_comps, scope=scope + '_mixing', w_init=var_init,
                            latent_activation=tflearn.activations.leaky_relu, out_activation=None,
                            is_batch_norm=using_batch_norm)
    mc = tf.nn.softmax(mc, axis=1)

    outputs = {'mean': mean, 'scale': scale, 'mc': mc}
    return outputs

