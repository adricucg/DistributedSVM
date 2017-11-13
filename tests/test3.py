# more Tensorflow methods trying out

import tensorflow as tf

sess = tf.InteractiveSession()

b_add = tf.Variable(0.0)
for i in range(0,3):
    b_add = tf.add(b_add, tf.constant(1.0))


x = tf.constant([[4], [1], [5]])
y = tf.constant([[2], [2], [2]])
z = tf.constant([[3], [3], [3]])

a = tf.multiply(tf.multiply(y, z), x)
b = tf.reduce_sum(a)
c= tf.reshape(z, shape=[1,-1])

sess.run(tf.global_variables_initializer())

print(sess.run(b_add))
print(sess.run(b))
print(sess.run(c))

