import tensorflow as tf


def create_model(lr=.001, model_name='model'):
    with tf.variable_scope(model_name):
        input_ = tf.placeholder(tf.float32, shape=(None, 42), name='inputs')
        label_ = tf.placeholder(tf.float32, shape=(None, 1), name='values')

        # fully connected layers
        cnn = tf.layers.dense(input_, 512, activation=tf.nn.relu, name='fc_1')
        cnn = tf.layers.dense(cnn, 512, activation=tf.nn.relu, name='fc_2')
        #cnn = tf.layers.dropout(cnn, .5)
        #cnn = tf.layers.dense(cnn, 512, activation=tf.nn.relu, name='fc_3')
        #cnn = tf.layers.dropout(cnn, .5)

        # regression layer
        out = tf.layers.dense(cnn, 1, activation=tf.nn.tanh, name='out')

        # compute the loss
        loss = tf.reduce_mean(tf.squared_difference(out, label_))

        # optimize the network
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

        return input_, label_, out, loss, optimizer
