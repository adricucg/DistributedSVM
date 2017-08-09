from sklearn.metrics.pairwise import rbf_kernel
from util import *

q = 4
C = 10
p = 3
gamma = 0.02
q = 3
eps = 1e-20
min_gradient_list = []
max_gradient_list = []
x_i_low_list = []
x_i_up_list = []
target_i_up_list = []
target_i_low_list = []
i_low_list = []
i_up_list = []

_x = [[5, 1, 3], [1, 2, 0], [4, 4, 8],
      [0.5, 0.5, 0.5], [4, 2, 3], [1, 1, 1],
      [1, 2, 0], [6, 7, 9], [2, 1, 0]]

_y = [1, 1, -1, 1, 1, -1, 1, -1, -1]
#_y = [1, 1, -1, 1, -1, -1]

x_up = np.array([1, 2, 0])
x_low = np.array([4, 4, 8])
y_up = 1
y_low = -1
beta_up = -1
beta_low = 1

x_i_up = tf.constant(x_up, dtype=tf.float32)
x_i_low = tf.constant(x_low, dtype=tf.float32)

b_sum = tf.Variable(0.0)
b_count = tf.Variable(0.0)

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

v1 = rbf_kernel(np.array([4,4,8]).reshape(1, -1), np.array([4,4,8]).reshape(1, -1), gamma=gamma)
v2 = rbf_kernel(np.array([1,2,0]).reshape(1, -1), np.array([4,4,8]).reshape(1, -1), gamma=gamma)
v3 = rbf_kernel(np.array([5,1,3]).reshape(1, -1), np.array([4,4,8]).reshape(1, -1), gamma=gamma)
v4 = rbf_kernel(np.array([0.5,0.5,0.5]).reshape(1, -1), np.array([4,4,8]).reshape(1, -1), gamma=gamma)
v5 = rbf_kernel(np.array([4, 2, 3]).reshape(1, -1), np.array([4,4,8]).reshape(1, -1), gamma=gamma)
v6 = rbf_kernel(np.array([6, 7, 9]).reshape(1, -1), np.array([4,4,8]).reshape(1, -1), gamma=gamma)
v7 = rbf_kernel(np.array([2, 2, 2]).reshape(1, -1), np.array([4,4,8]).reshape(1, -1), gamma=gamma)
v8 = rbf_kernel(np.array([1,1,1]).reshape(1, -1), np.array([4,4,8]).reshape(1, -1), gamma=gamma)

i = 0
for k in range(0, p):

    y = _y[i:i+3]
    x = _x[i:i+3]
    Y = tf.constant(y, dtype=tf.float32)
    X = tf.constant(x, dtype=tf.float32)

    print(Y.shape)
    print(X.shape)

    with tf.variable_scope("svm_worker") as scope:
        gradient_old = tf.get_variable('gradient_old_' + str(k), initializer=tf.negative(Y))
        alpha = tf.get_variable('alpha_' + str(k), initializer=tf.zeros(q))

    fn = lambda pair: tf.case(
            pred_fn_pairs=[
                (tf.logical_and(tf.greater(pair[1], 0.0), tf.less_equal(pair[1], C)), lambda: tf.constant(0)),
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

    gradient_up = tf.gather(gradient_cache, tf.concat([i_0, i_1, i_2], 0))
    gradient_low = tf.gather(gradient_cache, tf.concat([i_0, i_3, i_4], 0))

    min_g = tf.reduce_min(gradient_up)
    max_g = tf.reduce_max(gradient_low)

    # get the actual indexes from the gradient cache
    i_local_low = tf.where(tf.equal(gradient_cache, max_g))
    i_local_up = tf.where(tf.equal(gradient_cache, min_g))

    # corresponding samples for min and max gradients
    x_i_up_local = tf.gather(X, i_local_up)
    x_i_low_local = tf.gather(X, i_local_low)

    target_i_up_local = tf.gather(Y, i_local_up)
    target_i_low_local = tf.gather(Y, i_local_low)

    gradient_i_0 = tf.gather(gradient_old, i_0)
    gradient_sum = tf.reduce_sum(gradient_i_0)
    count = tf.size(i_0)

    min_gradient_list.append(min_g)
    max_gradient_list.append(max_g)
    x_i_up_list.append(x_i_up_local)
    x_i_low_list.append(x_i_low_local)
    target_i_up_list.append(target_i_up_local)
    target_i_low_list.append(target_i_low_local)
    i_up_list.append(i_local_up)
    i_low_list.append(i_local_low)

    i = i + 3

for k in range(0, p):
    with tf.variable_scope("svm_worker") as scope:
        scope.reuse_variables()
        gradient_sum, count = compute_b_local(k, C)
    b_sum = tf.add(b_sum, gradient_sum)
    b_count = tf.add(b_count, tf.to_float(count))

global_min_g = tf.reduce_min(min_gradient_list)
global_max_g = tf.reduce_max(max_gradient_list)

i_min = tf.arg_min(min_gradient_list, 0)  # this will be a value between 1 and p
i_max = tf.arg_max(max_gradient_list, 0)  # this will be a value between 1 and p

b = tf.divide(b_sum, b_count)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # run this twice
    for i in range(0, 2):

        result = \
            sess.run(
                [
                    i_up_list, i_low_list,
                    global_min_g, global_max_g,
                    i_min, i_max,
                    x_i_up_list, x_i_low_list,
                    target_i_up_list, target_i_low_list,
                    gradient_new
                ])

        all_i_up = result[0]
        all_i_low = result[1]
        beta_up = result[2]
        beta_low = result[3]
        p_up = result[4]
        p_low = result[5]
        all_x_up = result[6]
        all_x_low = result[7]
        all_target_up = result[8]
        all_target_low = result[9]

        # update values for the next iteration
        x_up = all_x_up[p_up][0][0]
        x_low = all_x_low[p_low][0][0]
        y_up = all_target_up[p_up][0][0]
        y_low = all_target_low[p_low][0][0]
        i_up = all_i_up[p_up][0][0]
        i_low = all_i_low[p_low][0][0]

        print('beta up:', beta_up)
        print('beta low: ', beta_low)
        print('worker corresponding to beta up:', p_up)
        print('worker corresponding to beta low:', p_low)
        print('target for x_i_up:', y_up)
        print('target for x_i_low:', y_low)
        print('index i_up:', i_up)
        print('index i_low:', i_low)

        with tf.variable_scope("svm_worker") as scope:
            scope.reuse_variables()

            alpha_up_t = tf.get_variable('alpha_' + str(p_up))
            alpha_low_t = tf.get_variable('alpha_' + str(p_low))

        # get the existing values for alpha
        alpha_up = sess.run(alpha_up_t)
        alpha_low = sess.run(alpha_low_t)

        print('alpha up: ', alpha_up)
        print('alpha low: ', alpha_low)

        alpha_up_old = alpha_up[i_up]
        alpha_low_old = alpha_low[i_low]

        # compute alpha
        eta = 2 * \
              rbf_kernel(x_low.reshape(1, -1), x_up.reshape(1, -1), gamma=gamma) \
              - rbf_kernel(x_up.reshape(1, -1), x_up.reshape(1, -1), gamma=gamma) \
              - rbf_kernel(x_low.reshape(1, -1), x_low.reshape(1, -1), gamma=gamma)

        eta = eta[0][0]

        # compute the new alpha values
        alpha_up_new, alpha_low_new = \
            compute_alpha(alpha_up_old, alpha_low_old, y_up, y_low, beta_up, beta_low, eta, C, eps)

        print('alpha up new: ', alpha_up_new)
        print('alpha low new: ', alpha_low_new)

        # update the array with the new values
        alpha_up[i_up] = alpha_up_new
        alpha_low[i_low] = alpha_low_new

        assign_up = alpha_up_t.assign(alpha_up)
        assign_low = alpha_low_t.assign(alpha_low)

        # assign the new values
        sess.run(assign_up)
        sess.run(assign_low)

        print('........................................')

    for k in range(0, q):
        with tf.variable_scope("svm_worker") as scope:
            scope.reuse_variables()
            alpha = tf.get_variable('alpha_' + str(k))
            gradient_old = tf.get_variable('gradient_old_' + str(k))

        print('alpha: ', sess.run(alpha))
        print('gradient: ', sess.run(gradient_old))


    print('........................................')

    print('b: ', sess.run(b))