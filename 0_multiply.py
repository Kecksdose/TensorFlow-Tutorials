import tensorflow as tf

# Create a symbolic variable 'a'
a = tf.placeholder("float")
# Create a symbolic variable 'b'
b = tf.placeholder("float")

# multiply the symbolic variables
y = tf.mul(a, b)

# create a session to evaluate the symbolic expressions
sess = tf.Session()

# eval expressions with parameters for a and b
print "%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})
print "%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3})
