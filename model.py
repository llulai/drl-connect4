import tensorflow as tf


def create_model(lr=.001):
    input_ = tf.placeholder(tf.float32, shape=(None, 6, 7, 1))
    label_ = tf.placeholder(tf.float32, shape=(None, 1))

    cnn = tf.layers.conv2d(input_, 32, (4,4), activation=tf.nn.relu, padding='same')
    cnn = tf.layers.dropout(cnn, .5)
    cnn = tf.layers.conv2d(cnn, 64, (4,4), activation=tf.nn.relu, padding='same')
    cnn = tf.layers.dropout(cnn, .5)
    cnn = tf.layers.conv2d(cnn, 128, (4,4), activation=tf.nn.relu, padding='same')
    cnn = tf.layers.dropout(cnn, .5)

    cnn = tf.contrib.layers.flatten(cnn)

    cnn = tf.layers.dense(cnn, 512, activation=tf.nn.relu)
    cnn = tf.layers.dense(cnn, 512, activation=tf.nn.relu)

    logits = tf.layers.dense(cnn, 1, activation=None)

    out = tf.nn.sigmoid(logits)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_))

    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    return input_, label_, logits, out, cost, optimizer
