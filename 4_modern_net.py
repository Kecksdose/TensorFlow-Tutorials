import tensorflow as tf
import numpy as np
import input_data


epochs = 20
learning_rate_1 = 0.001
learning_rate_2 = 0.09
np.random.seed(1337)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# this network is the same as the previous one except with an extra
# hidden layer + dropout
def model(capital_x, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    capital_x = tf.nn.dropout(capital_x, p_keep_input)
    h = tf.nn.relu(tf.matmul(capital_x, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX = mnist.train.images
trY = mnist.train.labels
teX = mnist.test.images
teY = mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(learning_rate_1,
                                     learning_rate_2).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(epochs):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                      p_keep_input: 0.8, p_keep_hidden: 0.5})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                     p_keep_input: 1.0,
                                                     p_keep_hidden: 1.0}))
