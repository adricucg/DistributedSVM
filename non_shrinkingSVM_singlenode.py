from non_shrinkingSVM_worker import *
from util import *
from sklearn.metrics.pairwise import rbf_kernel
import datetime
import numpy as np


def fit_SVM_single_worker(sess, cls, q, type):

    tf.logging.set_verbosity(tf.logging.DEBUG)

    C = 10
    e = 0.001
    eps = 1e-20
    gamma = 0.02
    start_index = 0
    stop_index = q
    beta_up = -1
    beta_low = 1

    print('Start time: {0}'.format(datetime.datetime.now()))

    with tf.variable_scope("svm_worker") as scope:

        x_i_up = tf.placeholder(tf.float32)
        x_i_low = tf.placeholder(tf.float32)
        v_up = tf.placeholder(tf.float32, None)
        v_low = tf.placeholder(tf.float32, None)

        t_data_X, t_data_y = load_data(0, 100, cls, type)

        i_up = np.where(t_data_y == 1)[0][0]
        i_low = np.where(t_data_y == -1)[0][0]

        # TODO these need to be randomly selected ?
        # does the data need to be shuffled?
        x_up = t_data_X[i_up]
        x_low = t_data_X[i_low]
        y_up = t_data_y[i_up]  #  1
        y_low = t_data_y[i_low] # -1

        min_gradient, max_gradient, x_iup, x_ilow, y_i_up, y_i_low, iup, ilow = \
            non_shrinked_svm_worker(start_index, stop_index, 0, x_i_up, x_i_low, v_up, v_low, C, cls, type)

        scope.reuse_variables()

        b_sum, b_count = compute_b_local(0, C)
        beta = tf.divide(b_sum, tf.to_float(b_count))

        # start session
        sess.run(tf.global_variables_initializer())

        epoch = 0
        while beta_up + 2 * e <= beta_low:

            print('Iteration: ', epoch)

            if i_up == i_low:
                print('i_up and i_low are the same!')
                break

            # get the existing values for alpha
            # alpha = sess.run(tf.gather(alpha_t, [i_up, i_low]))
            alpha_t = tf.get_variable('alpha_0')

            alpha = sess.run(alpha_t)

            print('alpha: ', alpha)

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

            # If examples can't be optimized within epsilon (eps), skip this pair
            if alpha_up_new == 0 and alpha_low_new == 0:
                print('Change in objective function is very small so we reached optimality')
                break

            print('alpha up new: ', alpha_up_new)
            print('alpha low new: ', alpha_low_new)

            # assign the new values
            #sess.run(tf.scatter_nd_update(alpha_t, [[i_up], [i_low]], [alpha_up_new, alpha_low_new]))

            alpha[i_up] = alpha_up_new
            alpha[i_low] = alpha_low_new

            sess.run(alpha_t.assign(alpha))

            # compute v_low and v_up
            vup = y_up * (alpha_up_new - alpha_up_old)
            vlow = y_low * (alpha_low_new - alpha_low_old)

            # need here some sort of reduce so you run one session only
            result = \
                sess.run(
                    [
                        min_gradient, max_gradient,
                        x_iup, x_ilow,
                        y_i_up, y_i_low,
                        iup, ilow
                    ],
                    feed_dict={x_i_low: x_low, x_i_up: x_up, v_up: vup, v_low: vlow})

            beta_up = result[0]
            beta_low = result[1]
            all_x_up = result[2]
            all_x_low = result[3]
            all_target_up = result[4]
            all_target_low = result[5]
            all_i_up = result[6]
            all_i_low = result[7]

            # update values for the next iteration
            x_up = all_x_up[0][0]
            x_low = all_x_low[0][0]
            y_up = all_target_up[0][0]
            y_low = all_target_low[0][0]
            i_up = all_i_up[0][0]
            i_low = all_i_low[0][0]

            print('beta up:', beta_up)
            print('beta low: ', beta_low)
            print('target for x_i_up:', y_up)
            print('target for x_i_low:', y_low)
            print('index i_up:', i_up)
            print('index i_low:', i_low)
            print('..........................................')

            epoch = epoch + 1

        b = sess.run(beta)

        print('b_count: ', sess.run(b_count))
        print('b is', b)

    print('Finish time: {0}'.format(datetime.datetime.now()))

    return b


def predict_SVM_single_worker(sess, b, x_new):

    with tf.variable_scope("svm_worker") as scope:
        scope.reuse_variables()

        alpha = tf.get_variable('alpha_0')
        Y = tf.get_variable('Y_0')
        X = tf.get_variable('X_0')

    kernel = tf.map_fn(lambda x: kernel_rbf(x_new, x), X)
    term = tf.multiply(tf.multiply(alpha, Y), kernel)
    sum = tf.reduce_sum(term)

    alpha_y_kernel = sess.run(sum)

    if(alpha_y_kernel - b) >= 0:
        return 1

    return -1


def test_SVM_single_worker(sess, b, cls, type):

    X_test, y_test = load_test_data(cls, type)

    with tf.variable_scope("svm_worker") as scope:
        scope.reuse_variables()

        alpha = tf.get_variable('alpha_0')
        Y = tf.get_variable('Y_0')
        X = tf.get_variable('X_0')

    alpha_mult_Y = tf.multiply(alpha, Y)

    kernel = matrix_kernel_rbf(X_test, X)
    term = tf.matmul(kernel, tf.expand_dims(alpha_mult_Y, 1))
    decision_fn = tf.subtract(term, b)

    fn = lambda pair: tf.case(
        pred_fn_pairs=[
            (tf.logical_and(tf.greater_equal(pair[0][0], 0.0), tf.equal(pair[1], 1)), lambda: tf.constant(1)),
            (tf.logical_and(tf.less(pair[0][0], 0.0), tf.equal(pair[1], -1)), lambda: tf.constant(1))
        ], default=lambda: tf.constant(0))

    pred_grid = tf.map_fn(fn, (decision_fn, y_test), dtype=tf.int32)

    correct_pred = tf.reduce_sum(pred_grid)

    accuracy = tf.divide(tf.to_float(correct_pred), tf.to_float(tf.size(pred_grid)))

    print('decision: ', sess.run(decision_fn))
    print('pred_grid: ', sess.run(pred_grid))
    print('test vector: ', y_test)
    print('correct pred: ', sess.run(correct_pred))

    result = sess.run(accuracy)

    print('Accuracy for class {0}: {1}'.format(cls, result))




