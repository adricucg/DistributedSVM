from util import *


def shrinked_svm_worker(sess, worker_index, active_indeces, x_i_up, x_i_low, v_up, v_low, C):

    # example 1:11000
    Y = tf.get_variable('Y_' + str(worker_index))
    X = tf.get_variable('X_' + str(worker_index))
    gradient = tf.get_variable('gradient_old_' + str(worker_index))
    alpha = tf.get_variable('alpha_' + str(worker_index))

    print('all samples shape: ', X.shape)

    Y_active =tf.gather(Y, active_indeces)
    X_active =tf.gather(X, active_indeces)
    gradient_active = tf.gather(gradient, active_indeces)
    alpha_active = tf.gather(alpha, active_indeces)

    print('active samples shape: ', X_active.shape)

    fn = lambda pair: tf.case(
        pred_fn_pairs=[
            (tf.logical_and(tf.greater(pair[1], 0.0), tf.less(pair[1], C)), lambda: tf.constant(0)),
            (tf.logical_and(tf.equal(pair[0], 1.0), tf.equal(pair[1], 0.0)), lambda: tf.constant(1)),
            (tf.logical_and(tf.equal(pair[0], -1.0), tf.equal(pair[1], C)), lambda: tf.constant(2)),
            (tf.logical_and(tf.equal(pair[0], 1.0), tf.equal(pair[1], C)), lambda: tf.constant(3)),
            (tf.logical_and(tf.equal(pair[0], -1.0), tf.equal(pair[1], 0.0)), lambda: tf.constant(4))
        ], default=lambda: tf.constant(-1))

    set_indexes = tf.map_fn(fn, (Y_active, alpha_active), dtype=tf.int32)

    # compute the gradient for active indexes
    kernel_up = tf.map_fn(lambda x: kernel_rbf(x_i_up, x), X_active)
    kernel_low = tf.map_fn(lambda x: kernel_rbf(x_i_low, x), X_active)
    term_up = tf.multiply(v_up, kernel_up)
    term_low = tf.multiply(v_low, kernel_low)
    term = tf.add(term_up, term_low)
    gradient_new = tf.add(gradient_active, term)

    # update the main gradient only for the active indexes
    update_index = tf.reshape(active_indeces, shape=[len(active_indeces), 1])
    gradient_update = tf.scatter_nd_update(gradient, update_index, gradient_new)
    #gradient_cache = gradient_active.assign(gradient_new)

    i_0 = tf.where(tf.equal(set_indexes, 0))
    i_1 = tf.where(tf.equal(set_indexes, 1))
    i_2 = tf.where(tf.equal(set_indexes, 2))
    i_3 = tf.where(tf.equal(set_indexes, 3))
    i_4 = tf.where(tf.equal(set_indexes, 4))

    i0_i1_i2 = tf.concat([i_0, i_1, i_2], 0)
    i0_i3_i4 = tf.concat([i_0, i_3, i_4], 0)

    gradient_up = tf.gather(gradient_new, i0_i1_i2)
    gradient_low = tf.gather(gradient_new, i0_i3_i4)

    min_g = tf.reduce_min(gradient_up)
    max_g = tf.reduce_max(gradient_low)

    # get the actual indexes from the gradient cache
    select_min_index = tf.reshape(tf.where(tf.equal(gradient_new, min_g)), [-1])
    select_max_index = tf.reshape(tf.where(tf.equal(gradient_new, max_g)), [-1])

    index_up = tf.reshape(tf.where(tf.equal(i0_i1_i2, select_min_index)), [-1])
    index_low = tf.reshape(tf.where(tf.equal(i0_i3_i4, select_max_index)), [-1])

    # example values between 1 and 5000
    i_active_up = tf.gather(i0_i1_i2, index_up)
    i_active_low = tf.gather(i0_i3_i4, index_low)

    i_local_up = tf.gather(active_indeces, i_active_up)
    i_local_low = tf.gather(active_indeces, i_active_low)

    # corresponding samples for min and max gradients
    x_i_up_local = tf.gather(X, i_local_up)
    x_i_low_local = tf.gather(X, i_local_low)

    # corresponding targets for min and max gradients
    target_i_up_local = tf.gather(Y, i_local_up)
    target_i_low_local = tf.gather(Y, i_local_low)

    return min_g, max_g, \
           x_i_up_local, x_i_low_local, \
           target_i_up_local, target_i_low_local, \
           i_local_up, i_local_low, \
           gradient_update


# separate the active and shrinked samples
def shrink_sets(sess, worker_index, C, beta_up, beta_low):

    Y = tf.get_variable('Y_' + str(worker_index))
    gradient = tf.get_variable('gradient_old_' + str(worker_index))
    alpha = tf.get_variable('alpha_' + str(worker_index))

    all_indexes =tf.range(tf.size(Y))

    fn = lambda pair: tf.case(
        pred_fn_pairs=[
            (tf.logical_and(tf.equal(pair[0], 1.0), tf.equal(pair[1], 0.0)), lambda: tf.constant(1)),
            (tf.logical_and(tf.equal(pair[0], -1.0), tf.equal(pair[1], C)), lambda: tf.constant(2)),
            (tf.logical_and(tf.equal(pair[0], 1.0), tf.equal(pair[1], C)), lambda: tf.constant(3)),
            (tf.logical_and(tf.equal(pair[0], -1.0), tf.equal(pair[1], 0.0)), lambda: tf.constant(4))
        ], default=lambda: tf.constant(-1))

    set_indexes = tf.map_fn(fn, (Y, alpha), dtype=tf.int32)

    i_1 = tf.where(tf.equal(set_indexes, 1))
    i_2 = tf.where(tf.equal(set_indexes, 2))
    i_3 = tf.where(tf.equal(set_indexes, 3))
    i_4 = tf.where(tf.equal(set_indexes, 4))

    i1_i2 = tf.concat([i_1, i_2], 0)
    i3_i4 = tf.concat([i_3, i_4], 0)

    # conditions for a sample to be shrinked
    less_beta_up = tf.reshape(tf.where(tf.less(gradient, beta_up)), [-1])
    greater_beta_low = tf.reshape(tf.where(tf.greater(gradient, beta_low)),[-1])

    # g = sess.run(gradient)
    # a = sess.run(alpha)
    # l1 = sess.run(i_1)
    # l2 = sess.run(i_2)
    # l3 = sess.run(i_3)
    # l4 = sess.run(i_4)
    # r = sess.run(greater_beta_low)
    # q = sess.run(less_beta_up)
    # m1 = sess.run(i1_i2)
    # m2 = sess.run(i3_i4)

    diff1 = tf.setdiff1d(greater_beta_low, tf.reshape(i1_i2, [-1]))[0]
    diff2 = tf.setdiff1d(less_beta_up, tf.reshape(i3_i4, [-1]))[0]

    shrink_indeces1 = tf.setdiff1d(greater_beta_low, diff1)[0]
    shrink_indeces2 = tf.setdiff1d(less_beta_up, diff2)[0]

    # the union of the indeces
    shrinked_indeces = tf.to_int32(tf.concat([shrink_indeces1, shrink_indeces2], 0))
    active_indeces = tf.setdiff1d(all_indexes, tf.reshape(shrinked_indeces, [-1]))[0]

    # r = sess.run(greater_beta_low)
    # p1 = sess.run(shrink_indeces1)
    # p2 = sess.run(shrink_indeces2)
    # v = sess.run(shrinked_indeces)
    # m = sess.run(active_indeces)

    return shrinked_indeces, active_indeces


def prepare_shrinked_svm_workers(sess,p, active_indeces_list, x_i_up, x_i_low, v_up, v_low, C):

    min_grad_list = []
    max_grad_list= []
    x_i_low_list = []
    x_i_up_list = []
    target_i_up_list = []
    target_i_low_list = []
    i_low_list = []
    i_up_list = []
    g_update_list = []

    for k in range(0, p):
        with tf.device("/job:job1/task:" + str(k)):

            # this is the actual gradient calculation for each active/non-shrinked data sample
            min_gradient, max_gradient, x_iup, x_ilow, target_i_up, target_i_low, i_up, i_low, g_update \
                = shrinked_svm_worker(sess, k, active_indeces_list[k], x_i_up, x_i_low, v_up, v_low, C)

        # data structures to hold shrinked min/max gradient, samples and indexes for up and low
        # will run this once we reached the shrinking threshold and we shrank the dataset
        min_grad_list.append(min_gradient)
        max_grad_list.append(max_gradient)
        x_i_up_list.append(x_iup)
        x_i_low_list.append(x_ilow)
        target_i_up_list.append(target_i_up)
        target_i_low_list.append(target_i_low)
        i_up_list.append(i_up)
        i_low_list.append(i_low)
        g_update_list.append(g_update)

    # get shrinked global min and max values
    global_min_g = tf.reduce_min(min_grad_list)
    global_max_g = tf.reduce_max(max_grad_list)

    # corresponding worker index for shrinked global min and max gradients
    i_min = tf.arg_min(min_grad_list, 0)
    i_max = tf.arg_max(max_grad_list, 0)

    output = [i_up_list, i_low_list,
              global_min_g, global_max_g,
              i_min, i_max,
              x_i_up_list, x_i_low_list,
              target_i_up_list, target_i_low_list,
              g_update_list]

    return output


def prepare_active_sets(sess, C, beta_low, beta_up, p, new_threshold):

    shrinked_indeces_list = []
    active_indeces_list = []

    for k in range(0, p):
        with tf.device("/job:job1/task:" + str(k)):
            # setup the shrinked datasets
            shrinked_i, active_i = shrink_sets(sess, k, C, beta_up, beta_low)

        shrinked_indeces_list.append(shrinked_i)
        active_indeces_list.append(active_i)
        #new_threshold = tf.add(new_threshold, tf.size(active_i))
        new_threshold = tf.add(new_threshold, tf.size(shrinked_i))

    return shrinked_indeces_list, active_indeces_list, new_threshold


def reconstruct_gradient(sess, worker_index, p, shrinked_indexes, temp_gradient):

    #print('temp gradient is: ', sess.run(temp_gradient))

    X = tf.get_variable('X_' + str(worker_index))
    Y = tf.get_variable('Y_' + str(worker_index))
    gradient = tf.get_variable('gradient_old_' + str(worker_index))
    X_shrinked = tf.gather(X, shrinked_indexes)
    Y_shrinked = tf.gather(Y, shrinked_indexes)

    for j in range(0, p):
        with tf.device("/job:job1/task:" + str(j)):
            alpha = tf.get_variable('alpha_' + str(j))
            X_j = tf.get_variable('X_' + str(j))
            Y_j = tf.get_variable('Y_' + str(j))

            alpha_mult_Y = tf.multiply(alpha, Y_j)
            kernel = matrix_kernel_rbf(X_shrinked, X_j)
            term = tf.matmul(kernel, tf.expand_dims(alpha_mult_Y, 1))

            term = tf.squeeze(term)

        temp_gradient = tf.add(temp_gradient, term)

    gradient_diff = tf.subtract(temp_gradient, Y_shrinked)

    update_index = tf.reshape(shrinked_indexes, shape=[len(shrinked_indexes), 1])

    gradient_new = tf.scatter_nd_update(gradient, update_index, gradient_diff)

            # m = sess.run(gradient)
            # n = sess.run(gradient_new)
            # z = sess.run(temp_gradient)
            # y = sess.run(Y_shrinked)
            # p = sess.run(gradient_diff)
            #
            # print(m)
            # print(z)
            # print(y)
            # print(p)
            # print(n)
            # print('.....')

    return gradient_new
