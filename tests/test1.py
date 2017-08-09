# just to try and run different commands
# no actual proper code here
import numpy as np
import tensorflow as tf

x_i_data = tf.constant([1,3,6,9], shape=[1,4])
rA = tf.reshape(tf.reduce_sum(tf.square(x_i_data), 1), [-1, 1])

arr = np.array([[31, 23,  4, 24, 27, 34],
                [18,  3, 25,  0,  6, 35],
                [28, 14, 33, 22, 20,  8],
                [13, 30, 21, 19,  7,  9],
                [16,  1, 26, 32,  2, 29],
                [17, 12,  5, 11, 10, 15]])

tensor_arr = tf.constant(arr)

elems = np.array([31, 23,  4, 24, 27, 34])
elems1 = np.array([18,  3, 25,  0,  6, 35])
j = tf.constant([2,1,5,6,1,1],dtype=tf.int64)
squares = tf.map_fn(lambda x: x * x, elems, dtype=tf.int64)
trans = tf.transpose(squares)
t = tf.multiply(squares, j)
v = [elems, elems1]
i_min = tf.arg_min(x_i_data, 1)
val_min = tf.gather(x_i_data, i_min)

t_orig = tf.constant([0, 1, 3, 3])
t_new = tf.map_fn(
        lambda x: tf.case(pred_fn_pairs=[
                (tf.equal(x, 0), lambda: tf.constant(0)),
                (tf.equal(x, 1), lambda: tf.constant(7)),
                (tf.equal(x, 2), lambda: tf.constant(10)),
                (tf.equal(x, 3), lambda: tf.constant(13))],default=lambda: tf.constant(-1)),
        t_orig)

elem = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
alternate = tf.map_fn(lambda x: x[0] * x[1], elem, dtype=tf.int64)
index3 = tf.where(tf.equal(t_orig, 3))
index0 = tf.where(tf.equal(t_orig, 0))

l1 = tf.range(tf.size(tf.constant([1,2,3,4,5,6])))
l2 = tf.constant([1,4,5,6])

l3 = tf.Variable([1,2,3])
index = tf.constant([1, 2])
index = tf.reshape(index, shape=[2,1])

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

print(sess.run(tf.concat([index0, index3], 0)))
print(sess.run(tf.gather(j, tf.concat([index0, index3], 0))))
print(sess.run(squares))
print(sess.run(t))
print(sess.run(x_i_data))
print(sess.run(tf.square(x_i_data)))
print(sess.run(tf.reduce_sum(tf.square(x_i_data), 1)))
print(sess.run(tf.zeros(3)))

print(sess.run(tf.arg_min(elems, 0)))

#print(sess.run(tf.rank(arr)))
print(sess.run(tf.reduce_min(arr,1)))

b = tf.reduce_min(tf.stack(v, 1),1)
print(sess.run(tf.stack(v, 1)))
print(sess.run(b))
print(sess.run(i_min))
c = sess.run(val_min)
print(c)
print(c.shape)

print(sess.run(tf.size(tf.setdiff1d(l1, l2)[0])))

print(sess.run(tf.gather(l2, [1,2])))

print(sess.run(index))
print(sess.run(tf.scatter_nd_update(l3, index, [8, 9])))


res = sess.run(tf.gather(l3, [1, 2]))
print(res[0])
print(res[1])

sess.close()