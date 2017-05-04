import tensorflow as tf


def create_model(lr=.001, model_name='model'):
    with tf.variable_scope(model_name):
        input_ = tf.placeholder(tf.float32, shape=(None, 6, 7, 2), name='inputs')
        label_ = tf.placeholder(tf.float32, shape=(None, 1), name='values')

        # conv 1
        cnn = tf.layers.conv2d(input_, 32, (4,4), activation=tf.nn.relu, padding='same', name='conv_1')
        cnn = tf.layers.dropout(cnn, .5, name='drop_1')
        cnn = tf.contrib.layers.batch_norm(cnn, 0.99)
        cnn = tf.identity(cnn, name='batch_1')

        # conv 2
        cnn = tf.layers.conv2d(cnn, 64, (4,4), activation=tf.nn.relu, padding='same', name='conv_2')
        cnn = tf.layers.dropout(cnn, .5, name='drop_2')
        cnn = tf.contrib.layers.batch_norm(cnn, 0.99)
        cnn = tf.identity(cnn, name='batch_2')

        # conv 3
        cnn = tf.layers.conv2d(cnn, 128, (4,4), activation=tf.nn.relu, padding='same', name='conv_3')
        cnn = tf.layers.dropout(cnn, .5, name='drop_3')
        cnn = tf.contrib.layers.batch_norm(cnn, 0.99)
        cnn = tf.identity(cnn, name='batch_3')

        # flatten the layers
        cnn = tf.contrib.layers.flatten(cnn)

        # fully connected layers
        cnn = tf.layers.dense(cnn, 512, activation=tf.nn.relu, name='fc_1')
        cnn = tf.layers.dense(cnn, 512, activation=tf.nn.relu, name='fc_2')

        # regression layer
        out = tf.layers.dense(cnn, 1, activation=tf.nn.tanh, name='out')

        # compute the loss
        loss = tf.reduce_mean(tf.squared_difference(out, label_))

        # optimize the network
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

        return input_, label_, out, loss, optimizer
