import tensorflow as tf


def create_model(lr=.001):
    input_ = tf.placeholder(tf.float32, shape=(None, 42))
    label_ = tf.placeholder(tf.float32, shape=(None, 1))

    dnn = tf.layers.dense(input_, 32, activation=tf.nn.relu)
    dnn = tf.layers.dense(dnn, 64, activation=tf.nn.relu)
    dnn = tf.layers.dense(dnn, 128, activation=tf.nn.relu)
    dnn = tf.layers.dense(dnn, 512, activation=tf.nn.relu)
    dnn = tf.layers.dense(dnn, 512, activation=tf.nn.relu)

    logits = tf.layers.dense(dnn, 1, activation=None)

    out = tf.nn.sigmoid(logits)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_))

    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    return input_, label_, logits, out, cost, optimizer
