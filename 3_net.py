import tensorflow as tf
import numpy as np
import input_data


epochs = 20
learning_rate = 0.05
np.random.seed(1337)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    # this is a basic mlp, think 2 stacked logistic regressions
    h = tf.nn.sigmoid(tf.matmul(X, w_h))
    # note that we dont take the softmax at the end
    # because our cost fn does that for us
    return tf.matmul(h, w_o)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX = mnist.train.images
trY = mnist.train.labels
teX = mnist.test.images
teY = mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

# create symbolic variables
w_h = init_weights([784, 625])
w_o = init_weights([625, 10])

py_x = model(X, w_h, w_o)

# compute costs
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
# construct an optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(epochs):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))
