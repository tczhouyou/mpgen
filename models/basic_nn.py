import tensorflow as tf


if tf.__version__ < '2.0.0':
    import tflearn
    sigmoid_act = tflearn.activations.sigmoid
    tanh_act = tflearn.activations.tanh
    leaky_relu_act = tflearn.activations.leaky_relu
else:
    sigmoid_act = tf.keras.activations.sigmoid
    tanh_act = tf.keras.activations.tanh
    leaky_relu_act = tf.nn.leaky_relu


if tf.__version__ < '2.0.0':
    import tflearn

    def fully_connected_nn(inputs, layer_dims, out_dim, latent_activation=tflearn.activations.leaky_relu,
                           out_activation=None,
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
else:
    def fully_connected_nn(inputs, layer_dims, out_dim, latent_activation=tf.nn.leaky_relu,
                           out_activation=None, w_init='glorot_uniform', scope=None):

        with tf.compat.v1.variable_scope(scope or 'fcnn', reuse=tf.compat.v1.AUTO_REUSE):
            layer = inputs
            for i in range(len(layer_dims)):
                lfcn = tf.keras.layers.Dense(layer_dims[i], activation=latent_activation, kernel_initializer=w_init)
                layer = lfcn(layer)

            ofcn = tf.keras.layers.Dense(out_dim, activation=out_activation, kernel_initializer=w_init)
            out = ofcn(layer)
            return out


def mdn_nn_v1(inputs, d_outputs, n_comps, nn_structure, using_batch_norm=False, scope='mdn_nn', var_init=None):
    d_feat = nn_structure['d_feat']
    feat_layers = nn_structure['feat_layers']
    mean_layers = nn_structure['mean_layers']
    scale_layers = nn_structure['scale_layers']
    mixing_layers = nn_structure['mixing_layers']

    if tf.__version__ < "2.0.0": # if tf version is smaller than 2.0, using tflearn otherwise keras

        if var_init is None:
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
    else:
        from tensorflow.keras import initializers
        if var_init is None:
            var_init = initializers.RandomNormal(stddev=0.003, seed=42)

        feats = fully_connected_nn(inputs, feat_layers, d_feat, scope=scope + "_feat",
                                   latent_activation=leaky_relu_act,
                                   out_activation=leaky_relu_act, w_init=var_init)

        mean = fully_connected_nn(feats, mean_layers, d_outputs, scope=scope + '_mean',
                                  latent_activation=leaky_relu_act, out_activation=None, w_init=var_init)

        scale = fully_connected_nn(feats, scale_layers, d_outputs, scope=scope + '_scale',
                                   latent_activation=leaky_relu_act, out_activation=leaky_relu_act, w_init=var_init)
        scale = tf.exp(scale)
        scale = tf.clip_by_value(scale, 0, 10000)

        mc = fully_connected_nn(feats, mixing_layers, n_comps, scope=scope + '_mixing',
                                latent_activation=leaky_relu_act, out_activation=None, w_init=var_init)

        pmc = tf.clip_by_value(mc, -10, 10)
        mc = tf.nn.softmax(pmc, axis=1)


    outputs = {'mean': mean, 'scale': scale, 'mc': mc}#, 'pmc': pmc}
    return outputs


