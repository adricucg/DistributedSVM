from util import *
from sklearn.metrics.pairwise import rbf_kernel

_x = np.array([[5, 1, 3], [1, 2, 0], [4, 4, 8],
               [0.5, 0.5, 0.5], [4, 2, 3], [1, 1, 1],
               [1, 2, 0], [6, 7, 9], [2, 1, 0]])

_y = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1])

q = 9
C = 5
gamma = 0.05
eps = 1e-20
x_i_up = tf.constant([1,2,0], dtype=tf.float32)
x_i_low = tf.constant([4,4,8], dtype=tf.float32)

x_up = np.array([1,2,0])
x_low = np.array([4,4,8])
y_up = 1
y_low = -1
beta_up = -1
beta_low = 1

v1 = rbf_kernel(np.array([4,4,8]).reshape(1, -1), np.array([1,2,0]).reshape(1, -1), gamma=gamma)
v2 = rbf_kernel(np.array([1,2,0]).reshape(1, -1), np.array([1,2,0]).reshape(1, -1), gamma=gamma)
v3 = rbf_kernel(np.array([5,1,3]).reshape(1, -1), np.array([1,2,0]).reshape(1, -1), gamma=gamma)
v4 = rbf_kernel(np.array([0.5,0.5,0.5]).reshape(1, -1), np.array([1,2,0]).reshape(1, -1), gamma=gamma)

eta = 2 * \
                  rbf_kernel(x_low.reshape(1, -1), x_up.reshape(1, -1), gamma=gamma) \
                  - rbf_kernel(x_up.reshape(1, -1), x_up.reshape(1, -1), gamma=gamma) \
                  - rbf_kernel(x_low.reshape(1, -1), x_low.reshape(1, -1), gamma=gamma)

eta = eta[0][0].astype(np.float32)

alpha_up_old = 0.0
alpha_low_old = 0.0

# compute the new alpha values
alpha_up_new, alpha_low_new = \
    compute_alpha(alpha_up_old, alpha_low_old, y_up, y_low, beta_up, beta_low, eta, C, eps)

vup = (y_up * (alpha_up_new - alpha_up_old)).astype(np.float32)
vlow = (y_low * (alpha_low_new - alpha_low_old)).astype(np.float32)

# save data subsample as constant in the graph and pinned to a particular worker
Y = tf.constant(_y, dtype=tf.float32)
X = tf.constant(_x, dtype=tf.float32)

print(q)
print(Y.shape)
print(X.shape)

# this variable is pinned to each individual server
gradient_old = tf.get_variable('gradient_old_' + str(0), initializer=tf.negative(Y))
alpha = tf.get_variable('alpha_' + str(0), initializer=tf.zeros(q))

fn = lambda pair: tf.case(
    pred_fn_pairs=[
        (tf.logical_and(tf.greater(pair[1], 0.0), tf.less(pair[1], C)), lambda: tf.constant(0)),
        (tf.logical_and(tf.equal(pair[0], 1.0), tf.equal(pair[1], 0.0)), lambda: tf.constant(1)),
        (tf.logical_and(tf.equal(pair[0], -1.0), tf.equal(pair[1], C)), lambda: tf.constant(2)),
        (tf.logical_and(tf.equal(pair[0], 1.0), tf.equal(pair[1], C)), lambda: tf.constant(3)),
        (tf.logical_and(tf.equal(pair[0], -1.0), tf.equal(pair[1], 0.0)), lambda: tf.constant(4))
    ], default=lambda: tf.constant(-1))

set_indexes = tf.map_fn(fn, (Y, alpha), dtype=tf.int32)

# compute the gradient
kernel_up = tf.map_fn(lambda x: kernel_rbf(x_i_up, x), X)
kernel_low = tf.map_fn(lambda x: kernel_rbf(x_i_low, x), X)
term_up = tf.multiply(vup, kernel_up)
term_low = tf.multiply(vlow, kernel_low)
term = tf.add(term_up, term_low)
gradient_new = tf.add(gradient_old, term)

# shape of gradient for MNIST is (sample_size, )
gradient_cache = gradient_old.assign(gradient_new)

# we need the gradients separated based on:
#  I0 + I1 + I2/# I0 + I3 + I4
i_0 = tf.where(tf.equal(set_indexes, 0))
i_1 = tf.where(tf.equal(set_indexes, 1))
i_2 = tf.where(tf.equal(set_indexes, 2))
i_3 = tf.where(tf.equal(set_indexes, 3))
i_4 = tf.where(tf.equal(set_indexes, 4))

i0_i1_i2 = tf.reshape(tf.concat([i_0, i_1, i_2], 0), [-1])
i0_i3_i4 = tf.reshape(tf.concat([i_0, i_3, i_4], 0), [-1])

gradient_up = tf.gather(gradient_cache, i0_i1_i2)
gradient_low = tf.gather(gradient_cache, i0_i3_i4)

min_g = tf.reduce_min(gradient_up)
max_g = tf.reduce_max(gradient_low)

# get the actual indexes from the gradient cache
select_min_index = tf.reshape(tf.where(tf.equal(gradient_cache, min_g)), [-1])
select_max_index = tf.reshape(tf.where(tf.equal(gradient_cache, max_g)), [-1])

index_up = tf.reshape(tf.where(tf.equal(i0_i1_i2, select_min_index)), [-1])
index_low = tf.reshape(tf.where(tf.equal(i0_i3_i4, select_max_index)), [-1])

i_local_up = tf.gather(i0_i1_i2, index_up)
i_local_low = tf.gather(i0_i3_i4, index_low)

# corresponding samples for min and max gradients
x_i_up_local = tf.gather(X, i_local_up)
x_i_low_local = tf.gather(X, i_local_low)

target_i_up_local = tf.gather(Y, i_local_up)
target_i_low_local = tf.gather(Y, i_local_low)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # i0 = sess.run(i_0)
    # i1 = sess.run(i_1) # 0, 1, 3, 4, 6
    # i2 = sess.run(i_2)
    # i3 = sess.run(i_3)
    # i4 = sess.run(i_4) # 2, 7, 8

    i0i1i2 = sess.run(i0_i1_i2)
    i0i3i4 = sess.run(i0_i3_i4)

    s_min_index = sess.run(select_min_index)
    s_max_index = sess.run(select_max_index)

    indexup= sess.run(index_up) # [[0 0]]
    indexlow = sess.run(index_low) # [[3 0]]

    result = \
        sess.run(
            [
                i_local_up, i_local_low,
                min_g, max_g,
                x_i_up_local, x_i_low_local,
                target_i_up_local, target_i_low_local,
            ])

    all_i_up = result[0]
    all_i_low = result[1]
    beta_up = result[2]
    beta_low = result[3]
    all_x_up = result[4]
    all_x_low = result[5]
    all_target_up = result[6]
    all_target_low = result[7]

    print('i_up ', all_i_up) # 0
    print('i_low ', all_i_low) # 2
    print('beta up ', beta_up) # -0.87534
    print('beta low ', beta_low) # 5.96046e-08
    print('x up ', all_x_up) # [5 1 3]
    print('x low ', all_x_low) # [1 2 0]
    print('target up ', all_target_up) # 1
    print('target low ', all_target_low) # -1


