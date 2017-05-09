import tensorflow as tf


def create_model(lr=.001, model_name='deep_neural_network'):
    with tf.variable_scope(model_name):
        input_ = tf.placeholder(tf.float32, shape=(None, 42), name='inputs')
        label_ = tf.placeholder(tf.float32, shape=(None, 1), name='values')

        if model_name == 'deep_neural_network':
            logits = create_deep_neural_network(input_)
        else:
            logits = create_conv_neural_network(input_)

        # regression layer
        out = tf.layers.dense(logits, 1, activation=tf.nn.tanh, name='out')

        # compute the loss
        loss = tf.reduce_mean(tf.squared_difference(out, label_))

        # optimize the network
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

        return input_, label_, out, loss, optimizer


def create_deep_neural_network(input_):
    with tf.variable_scope('deep_neural_network'):

        # fully connected layers
        cnn = tf.layers.dense(input_, 512, activation=tf.nn.relu, name='fc_1')
        logits = tf.layers.dense(cnn, 512, activation=tf.nn.relu, name='fc_2')

        return logits


def create_conv_neural_network(input_):
    with tf.variable_scope('conv_neural_network'):
        # fully connected layers
        conv = tf.reshape(input_, shape=[-1, 7, 6, 1], name='reshape')
        conv = tf.layers.conv2d(conv, 32, (3, 3), padding='same', name='conv_1')
        conv = tf.layers.conv2d(conv, 64, (3, 3), padding='same', name='conv_2')

        logits = tf.contrib.layers.flatten(conv)

        return logits
