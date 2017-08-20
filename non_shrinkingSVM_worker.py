from util import *


def non_shrinked_svm_worker(start_index, stop_index, worker_index, x_i_up, x_i_low, v_up, v_low, C, cls, type):

    _x, _y = load_data(start_index, stop_index, cls, type)
    q = stop_index - start_index

    # these variables are pinned to each individual server
    # save data subsample as constant in the graph and pinned to a particular worker
    Y = tf.get_variable('Y_' + str(worker_index), initializer=tf.constant(_y, dtype=tf.float32))
    X = tf.get_variable('X_' + str(worker_index), initializer=tf.constant(_x, dtype=tf.float32))
    gradient_old = tf.get_variable('gradient_old_' + str(worker_index), initializer=tf.negative(tf.constant(_y, dtype=tf.float32)))
    alpha = tf.get_variable('alpha_' + str(worker_index), initializer=tf.zeros(q))

    print(X.shape)
    print(Y.shape)

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
    term_up = tf.multiply(v_up, kernel_up)
    term_low = tf.multiply(v_low, kernel_low)
    term = tf.add(term_up, term_low)
    gradient_new = tf.add(gradient_old, term)

    # shape of gradient for MNIST is (sample_size, )
    # update the actual gradient
    gradient_cache = gradient_old.assign(gradient_new)

    # we need the gradients separated based on:
    #  I0 + I1 + I2/# I0 + I3 + I4
    i_0 = tf.where(tf.equal(set_indexes, 0))
    i_1 = tf.where(tf.equal(set_indexes, 1))
    i_2 = tf.where(tf.equal(set_indexes, 2))
    i_3 = tf.where(tf.equal(set_indexes, 3))
    i_4 = tf.where(tf.equal(set_indexes, 4))

    i0_i1_i2 = tf.concat([i_0, i_1, i_2], 0)
    i0_i3_i4 = tf.concat([i_0, i_3, i_4], 0)

    gradient_up = tf.gather(gradient_cache, i0_i1_i2)
    gradient_low = tf.gather(gradient_cache, i0_i3_i4)

    min_g = tf.reduce_min(gradient_up)
    max_g = tf.reduce_max(gradient_low)

    # get the actual indexes from the gradient cache
    select_min_index = tf.reshape(tf.where(tf.equal(gradient_cache, min_g)), [-1])
    select_max_index = tf.reshape(tf.where(tf.equal(gradient_cache, max_g)), [-1])

    #select_min_index = tf.where(tf.equal(gradient_cache, min_g))
    #select_max_index = tf.where(tf.equal(gradient_cache, max_g))

    index_up = tf.reshape(tf.where(tf.equal(i0_i1_i2, select_min_index)), [-1])
    index_low = tf.reshape(tf.where(tf.equal(i0_i3_i4, select_max_index)), [-1])

    #index_up = tf.where(tf.equal(i0_i1_i2, select_min_index))
    #index_low = tf.where(tf.equal(i0_i3_i4, select_max_index))

    i_local_up = tf.gather(i0_i1_i2, index_up)
    i_local_low = tf.gather(i0_i3_i4, index_low)

    # corresponding samples for min and max gradients
    x_i_up_local = tf.gather(X, i_local_up)
    x_i_low_local = tf.gather(X, i_local_low)

    # corresponding targets for min and max gradients
    target_i_up_local = tf.gather(Y, i_local_up)
    target_i_low_local = tf.gather(Y, i_local_low)

    return min_g, max_g, \
           x_i_up_local, x_i_low_local, \
           target_i_up_local, target_i_low_local, \
           i_local_up, i_local_low


def prepare_non_shrinked_workers(p, q, start_index, stop_index, x_i_up, x_i_low, v_up, v_low, C, cls, type):

    min_gradient_list = []
    max_gradient_list = []
    x_i_low_list = []
    x_i_up_list = []
    target_i_up_list = []
    target_i_low_list = []
    i_low_list = []
    i_up_list = []

    for k in range(0, p):
        with tf.device("/job:job1/task:" + str(k)):
            # pin the compute expensive operations to different servers in the cluster
            # this is the actual gradient calculation for all data samples
            min_gradient, max_gradient, x_iup, x_ilow, target_i_up, target_i_low, iup, ilow \
                = non_shrinked_svm_worker(start_index, stop_index, k, x_i_up, x_i_low, v_up, v_low, C, cls, type)

        min_gradient_list.append(min_gradient)
        max_gradient_list.append(max_gradient)
        x_i_up_list.append(x_iup)
        x_i_low_list.append(x_ilow)
        target_i_up_list.append(target_i_up)
        target_i_low_list.append(target_i_low)
        i_up_list.append(iup)
        i_low_list.append(ilow)

        # increment the index
        start_index = start_index + q
        stop_index = stop_index + q

    # get global min and max values
    global_min_g = tf.reduce_min(min_gradient_list)
    global_max_g = tf.reduce_max(max_gradient_list)

    # corresponding worker index for global min and max gradients
    i_min = tf.arg_min(min_gradient_list, 0)  # this will be a value between 1 and p
    i_max = tf.arg_max(max_gradient_list, 0)  # this will be a value between 1 and p

    output = [i_up_list, i_low_list,
              global_min_g, global_max_g,
              i_min, i_max,
              x_i_up_list, x_i_low_list,
              target_i_up_list, target_i_low_list]

    return output


def single_node_svm_worker(X, Y, gradient, alpha, C):

    fn = lambda pair: tf.case(
        pred_fn_pairs=[
            (tf.logical_and(tf.greater(pair[1], 0.0), tf.less(pair[1], C)), lambda: tf.constant(0)),
            (tf.logical_and(tf.equal(pair[0], 1.0), tf.equal(pair[1], 0.0)), lambda: tf.constant(1)),
            (tf.logical_and(tf.equal(pair[0], -1.0), tf.equal(pair[1], C)), lambda: tf.constant(2)),
            (tf.logical_and(tf.equal(pair[0], 1.0), tf.equal(pair[1], C)), lambda: tf.constant(3)),
            (tf.logical_and(tf.equal(pair[0], -1.0), tf.equal(pair[1], 0.0)), lambda: tf.constant(4))
        ], default=lambda: tf.constant(-1))

    set_indexes = tf.map_fn(fn, (Y, alpha), dtype=tf.int32)

    # we need the gradients separated based on:
    #  I0 + I1 + I2/# I0 + I3 + I4
    i_0 = tf.where(tf.equal(set_indexes, 0))
    i_1 = tf.where(tf.equal(set_indexes, 1))
    i_2 = tf.where(tf.equal(set_indexes, 2))
    i_3 = tf.where(tf.equal(set_indexes, 3))
    i_4 = tf.where(tf.equal(set_indexes, 4))

    i0_i1_i2 = tf.concat([i_0, i_1, i_2], 0)
    i0_i3_i4 = tf.concat([i_0, i_3, i_4], 0)

    gradient_up = tf.gather(gradient, i0_i1_i2)
    gradient_low = tf.gather(gradient, i0_i3_i4)

    min_g = tf.reduce_min(gradient_up)
    max_g = tf.reduce_max(gradient_low)

    # get the actual indexes from the gradient cache
    select_min_index = tf.reshape(tf.where(tf.equal(gradient, min_g)), [-1])
    select_max_index = tf.reshape(tf.where(tf.equal(gradient, max_g)), [-1])

    index_up = tf.reshape(tf.where(tf.equal(i0_i1_i2, select_min_index)), [-1])
    index_low = tf.reshape(tf.where(tf.equal(i0_i3_i4, select_max_index)), [-1])

    i_local_up = tf.gather(i0_i1_i2, index_up)
    i_local_low = tf.gather(i0_i3_i4, index_low)

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


def single_node_gradient_compute(X, gradient, x_i_up, x_i_low, v_up, v_low):

    # compute the gradient
    kernel_up = tf.map_fn(lambda x: kernel_rbf(x_i_up, x), X)
    kernel_low = tf.map_fn(lambda x: kernel_rbf(x_i_low, x), X)
    term_up = tf.multiply(v_up, kernel_up)
    term_low = tf.multiply(v_low, kernel_low)
    term = tf.add(term_up, term_low)
    gradient_new = tf.add(gradient, term)

    return gradient_new
