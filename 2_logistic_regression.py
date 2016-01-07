import tensorflow as tf
import numpy as np
import input_data


epochs = 20
learning_rate = 0.05
np.random.seed(1337)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# notice we use the same model as linear regression,
# this is because there is a baked in cost function which performs
# softmax and cross entropy
def model(X, w):
    return tf.matmul(X, w)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX = mnist.train.images
trY = mnist.train.labels
teX = mnist.test.images
teY = mnist.test.labels

# create symbolic variables
# X = 28 * 28 = 784
# Y = sum[0, 1, 2, ..., 9] = 10
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

# like in linear regression, we need a shared variable weight matrix
# for logistic regression
w = init_weights([784, 10])

py_x = model(X, w)

# compute mean cross entropy (softmax is applied internally)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
# construct optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# at predict time, evaluate the argmax of the logistic regression
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(epochs):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))
