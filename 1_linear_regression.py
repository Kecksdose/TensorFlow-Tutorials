import tensorflow as tf
import numpy as np


epochs = 20
learning_rate = 0.01
np.random.seed(1337)


# model is just X*w so this model line is pretty simple
def model(X, w):
    return tf.mul(X, w)

trX = np.linspace(-1, 1, 101)

# create a y value which is approximately linear but with some random
# noise
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

# create symbolic variables
X = tf.placeholder("float")
Y = tf.placeholder("float")


# create a shared variable (like theano.shared) for the weight matrix
w = tf.Variable(1.0, name="weights")
y_model = model(X, w)

# use sqr error for cost function
cost = (tf.pow(Y - y_model, 2))

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
# you need to initialize variables (in this case just variable W)
init = tf.initialize_all_variables()
sess.run(init)

for i in range(epochs):
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})

print(sess.run(w))  # something around 2
