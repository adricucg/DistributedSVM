from non_shrinkingSVM_worker import *
from util import *
from sklearn.metrics.pairwise import rbf_kernel
import datetime


def fit_SVM_parallel(sess, cls, p, q, type):

    tf.logging.set_verbosity(tf.logging.DEBUG)

    C = 10
    e = 0.001
    gamma = 0.125
    eps = 1e-20
    start_index = 0
    stop_index = q
    min_gradient_list = []
    max_gradient_list = []
    x_i_low_list = []
    x_i_up_list = []
    target_i_up_list = []
    target_i_low_list = []
    i_low_list = []
    i_up_list = []
    beta_up = -1
    beta_low = 1
    k_up = 0
    k_low = 0

    print('Start time: {0}'.format(datetime.datetime.now()))

    with tf.variable_scope("svm_worker") as scope:

        x_i_up = tf.placeholder(tf.float32)
        x_i_low = tf.placeholder(tf.float32)
        v_up = tf.placeholder(tf.float32, None)
        v_low = tf.placeholder(tf.float32, None)
        b_sum = tf.Variable(0.0)
        b_count = tf.Variable(0.0)
        tf.get_variable('total', initializer=tf.constant(0.0))
        tf.get_variable('x_test', initializer=tf.constant(0.0), validate_shape=False)
        tf.get_variable('y_test', initializer=tf.constant(0), validate_shape=False)

        # TODO these need to be randomly selected ?
        # does the data need to be shuffled?
        t_data_X, t_data_y = load_data(0, 100, cls, type)

        i_up = np.where(t_data_y == 1)[0][0]
        i_low = np.where(t_data_y == -1)[0][0]

        x_up = t_data_X[i_up]
        x_low = t_data_X[i_low]
        y_up = t_data_y[i_up]  #  1
        y_low = t_data_y[i_low] # -1

        for k in range(0, p):
            with tf.device("/job:job1/task:" + str(k)):
                # pin the compute expensive operations to different servers in the cluster
                # this is the actual gradient calculation for each subsample
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

        scope.reuse_variables()

        for k in range(0, p):
            with tf.device("/job:job1/task:" + str(k)):
                g_sum, no_elems = compute_b_local(k, C)
            b_sum = tf.add(b_sum, g_sum)
            b_count = tf.add(b_count, tf.to_float(no_elems))

        global_min_g = tf.reduce_min(min_gradient_list)
        global_max_g = tf.reduce_max(max_gradient_list)

        i_min = tf.arg_min(min_gradient_list, 0)  # this will be a value between 1 and p
        i_max = tf.arg_max(max_gradient_list, 0)  # this will be a value between 1 and p

        beta = tf.divide(b_sum, b_count)

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # run the optimization loop
        epoch = 0
        while beta_up + 2 * e <= beta_low:

            print('Iteration: ', epoch)

            if i_up == i_low and k_up == k_low:
                print('i_up and i_low are the same!')
                break

            if k_up != k_low:

                with tf.device("/job:job1/task:" + str(k_up)):
                    alpha_up_t = tf.get_variable('alpha_' + str(k_up))

                with tf.device("/job:job1/task:" + str(k_low)):
                    alpha_low_t = tf.get_variable('alpha_' + str(k_low))

                alpha_up = sess.run(alpha_up_t)
                alpha_low = sess.run(alpha_low_t)

                alpha_up_old = alpha_up[i_up]
                alpha_low_old = alpha_low[i_low]

            else:

                with tf.device("/job:job1/task:" + str(k_up)):
                    alpha_t = tf.get_variable('alpha_' + str(k_up))

                alpha = sess.run(alpha_t)

                alpha_up_old = alpha[i_up]
                alpha_low_old = alpha[i_low]

            print('alpha up old: ', alpha_up_old)
            print('alpha low old: ', alpha_low_old)

            # compute eta
            eta = 2 * \
                  rbf_kernel(x_low.reshape(1, -1), x_up.reshape(1, -1), gamma=gamma) \
                  - rbf_kernel(x_up.reshape(1, -1), x_up.reshape(1, -1), gamma=gamma) \
                  - rbf_kernel(x_low.reshape(1, -1), x_low.reshape(1, -1), gamma=gamma)

            eta = eta[0][0]

            # compute the new alpha values
            alpha_up_new, alpha_low_new = \
                compute_alpha(alpha_up_old, alpha_low_old, y_up, y_low, beta_up, beta_low, eta, C, eps)

            # the change in alphas is very small so exit
            if alpha_up_new == 0 and alpha_low_new == 0:
                print('Change in alphas is very small so exit')
                break

            print('alpha up new: ', alpha_up_new)
            print('alpha low new: ', alpha_low_new)

            if k_up != k_low:

                alpha_up[i_up] = alpha_up_new
                alpha_low[i_low] = alpha_low_new

                # assign the new values
                sess.run(alpha_up_t.assign(alpha_up))
                sess.run(alpha_low_t.assign(alpha_low))

            else:
                # assign the new values

                alpha[i_up] = alpha_up_new
                alpha[i_low] = alpha_low_new

                sess.run(alpha_t.assign(alpha))

            # compute v_low and v_up
            vup = y_up * (alpha_up_new - alpha_up_old)
            vlow = y_low * (alpha_low_new - alpha_low_old)

            # need here some sort of reduce so you run one session only but the underlying operations are distributed
            result = \
                sess.run(
                    [
                        i_up_list, i_low_list,
                        global_min_g, global_max_g,
                        i_min, i_max,
                        x_i_up_list, x_i_low_list,
                        target_i_up_list, target_i_low_list,
                    ],
                    feed_dict={x_i_low: x_low, x_i_up: x_up, v_up: vup, v_low: vlow})

            all_i_up = result[0]
            all_i_low = result[1]
            beta_up = result[2]
            beta_low = result[3]
            k_up = result[4]
            k_low = result[5]
            all_x_up = result[6]
            all_x_low = result[7]
            all_target_up = result[8]
            all_target_low = result[9]

            # update values for the next iteration
            x_up = all_x_up[k_up][0][0]
            x_low = all_x_low[k_low][0][0]
            y_up = all_target_up[k_up][0][0]
            y_low = all_target_low[k_low][0][0]
            i_up = all_i_up[k_up][0][0]
            i_low = all_i_low[k_low][0][0]

            print('beta up:', beta_up)
            print('beta low: ', beta_low)
            print('worker corresponding to beta up:', k_up)
            print('worker corresponding to beta low:', k_low)
            print('target for x_i_up:', y_up)
            print('target for x_i_low:', y_low)
            print('index i_up:', i_up)
            print('index i_low:', i_low)
            print('..........................................')

            epoch = epoch + 1

        # compute b and return it
        b = sess.run(beta)

        print('b count', sess.run(b_count))
        print('b is', b)

    print('Finish time: {0}'.format(datetime.datetime.now()))

    return b


def predict_SVM_parallel(sess, b, x_new):

    with tf.variable_scope("svm_worker") as scope:
        scope.reuse_variables()
        total = tf.get_variable('total')

    p = 5  # number of workers
    for k in range(0, p):
        with tf.device("/job:job1/task:" + str(k)):
            with tf.variable_scope("svm_worker") as scope:
                scope.reuse_variables()

                alpha = tf.get_variable('alpha_' + str(k))
                X = tf.get_variable('X_' + str(k))
                Y = tf.get_variable('Y_' + str(k))

            kernel = tf.map_fn(lambda x: kernel_rbf(x_new, x), X)
            term = tf.multiply(tf.multiply(alpha, Y), kernel)
            sum = tf.reduce_sum(term)

        total = tf.add(total, sum)

    alpha_y_kernel = sess.run(total)

    if(alpha_y_kernel - b) >= 0:
        return 1

    return -1


def test_SVM_parallel(sess, b, cls, p, type):

    X_test, Y_test = load_test_data(cls, type)
    X_test = X_test.astype(np.float32)

    batch_size = 500
    epochs = int(len(X_test)/batch_size)

    with tf.variable_scope("svm_worker") as scope:
        scope.reuse_variables()
        total = tf.get_variable('total')

        x_test = tf.get_variable('x_test', validate_shape=False)
        y_test = tf.get_variable('y_test', dtype=tf.int32, validate_shape=False)

        for k in range(0, p):
            with tf.device("/job:job1/task:" + str(k)):
                alpha = tf.get_variable('alpha_' + str(k))
                X = tf.get_variable('X_' + str(k))
                Y = tf.get_variable('Y_' + str(k))

                alpha_mult_Y = tf.multiply(alpha, Y)
                kernel = matrix_kernel_rbf(x_test, X)
                term = tf.matmul(kernel, tf.expand_dims(alpha_mult_Y, 1))

            total = tf.add(total, term)

    decision_fn = tf.subtract(total, b)

    fn = lambda pair: tf.case(
        pred_fn_pairs=[
            (tf.logical_and(tf.greater_equal(pair[0][0], 0.0), tf.equal(pair[1], 1)), lambda: tf.constant(1)),
            (tf.logical_and(tf.less(pair[0][0], 0.0), tf.equal(pair[1], -1)), lambda: tf.constant(1))
        ], default=lambda: tf.constant(0))

    pred_grid = tf.map_fn(fn, (decision_fn, y_test), dtype=tf.int32)

    correct_pred = tf.reduce_sum(pred_grid)

    accuracy = tf.divide(tf.to_float(correct_pred), tf.to_float(tf.size(pred_grid)))

    print('Running predictions')

    total_accuracy = 0
    for i in range(0, epochs):

        _x = X_test[i * batch_size:(i + 1) * batch_size]
        _y = Y_test[i * batch_size:(i + 1) * batch_size]
        result = sess.run(accuracy, feed_dict={x_test: _x, y_test: _y})

        total_accuracy = total_accuracy + result

    final_accuracy = round((total_accuracy/epochs)*100, 2)

    print('Accuracy for class {0}: {1}%'.format(cls, final_accuracy))



