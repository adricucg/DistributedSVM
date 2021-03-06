from shrinkedSVM_worker import *
from non_shrinkingSVM_worker import *
from util import *
from sklearn.metrics.pairwise import rbf_kernel
import datetime


def fit_shrinking_SVM_parallel(sess, cls, p, q, shrinking_counter, type):

    tf.logging.set_verbosity(tf.logging.DEBUG)

    C = 10
    e = 0.001
    gamma = 0.125
    eps = 1e-20
    start_index = 0
    stop_index = q

    beta_up = -1
    beta_low = 1
    k_up = 0
    k_low = 0
    alpha_dictionary = {}

    shrink = False

    shrinked_indexes = []

    counter = shrinking_counter

    print('Start time: {0}'.format(datetime.datetime.now()))

    with tf.variable_scope("svm_worker") as scope:

        x_i_up = tf.placeholder(tf.float32)
        x_i_low = tf.placeholder(tf.float32)
        v_up = tf.placeholder(tf.float32, None)
        v_low = tf.placeholder(tf.float32, None)
        b_sum = tf.Variable(0.0)
        b_count = tf.Variable(0.0)
        new_threshold = tf.Variable(0)
        temp_gradient = tf.Variable(0.0)
        tf.get_variable('total', initializer=tf.constant(0.0))

        # does the data need to be shuffled?
        t_data_X, t_data_y = load_data(0, 100, cls, type)
        #t_data_X, t_data_y = load_data(0, 9, cls)

        i_up = np.where(t_data_y == 1)[0][0]
        i_low = np.where(t_data_y == -1)[0][0]

        x_up = t_data_X[i_up]
        x_low = t_data_X[i_low]
        y_up = t_data_y[i_up]  #  1
        y_low = t_data_y[i_low] # -1

        optimize_all_samples \
                = prepare_non_shrinked_workers(p, q, start_index, stop_index, x_i_up, x_i_low, v_up, v_low, C, cls, type)

        scope.reuse_variables()

        # b threshold calculation...this runs after we converged
        for k in range(0, p):
            with tf.device("/job:job1/task:" + str(k)):
                g_sum, no_elems = compute_b_local(k, C)
            b_sum = tf.add(b_sum, g_sum)
            b_count = tf.add(b_count, tf.to_float(no_elems))

        beta = tf.divide(b_sum, b_count)

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        epoch = 0

        # run the optimization loop
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

            if shrinking_counter == 0:

                print('..........................................')
                print('Attempting to shrink the dataset')

                # active_indexes is a list, for each worker the list of active indexes
                # shrinked_indexes is a list, for each worker the list of shrinked indexes
                shrinked_index_list, active_index_list, threshold \
                    = prepare_active_sets(sess, C, beta_low, beta_up, p, new_threshold)

                sess.run(tf.local_variables_initializer())

                # use the number of active samples as the next shrinking threshold
                shrinking_result = sess.run([shrinked_index_list, active_index_list, threshold])

                shrinked_indexes = shrinking_result[0]
                active_indexes = shrinking_result[1]
                shrinked_indexes_counter = shrinking_result[2]

                if shrinked_indexes_counter > 0:

                    print('Shrinking samples found: ', shrinked_indexes_counter)

                    shrink = True

                    optimize_active_samples = \
                            prepare_shrinked_svm_workers(sess,p, active_indexes, x_i_up, x_i_low, v_up, v_low, C)

                    sess.run(tf.local_variables_initializer())

                else:
                    print('No shrinking samples found')

                shrinking_counter = counter

                print('..........................................')

            if shrink:
                result = sess.run(optimize_active_samples, feed_dict={x_i_low: x_low, x_i_up: x_up, v_up: vup, v_low: vlow})
            else:
                # do the normal session call with all the samples
                result = sess.run(optimize_all_samples, feed_dict={x_i_low: x_low, x_i_up: x_up, v_up: vup, v_low: vlow})

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

            # sync the gradient when close to convergence
            if (beta_low - beta_up) < 20 * e and shrink:
                print('Reconstructing the gradient')

                gradient_reconstruct_list = []

                for k in range(0, p):
                    if len(shrinked_indexes[k]) > 0:
                        with tf.device("/job:job1/task:" + str(k)):
                            gradient_reconstruct = reconstruct_gradient(sess, k, p, shrinked_indexes[k], temp_gradient)

                    gradient_reconstruct_list.append(gradient_reconstruct)

                # initialize local variables
                sess.run(tf.local_variables_initializer())

                sess.run(gradient_reconstruct_list)

                # the beta_up and beta_low should be found using all samples not just the shrinked ones
                shrink = False

                print('..........................................')

            shrinking_counter = shrinking_counter - 1

            epoch = epoch + 1

        # reconstruct it one more time since we converged
        if shrink:
            print('Final gradient reconstruction')

            final_gradient_reconstruct_list = []

            for k in range(0, p):
                if len(shrinked_indexes[k]) > 0:
                    with tf.device("/job:job1/task:" + str(k)):
                        gradient_reconstruct = reconstruct_gradient(sess, k, p, shrinked_indexes[k], temp_gradient)

                    final_gradient_reconstruct_list.append(gradient_reconstruct)

            # initialize local variables
            sess.run(tf.local_variables_initializer())

            sess.run(final_gradient_reconstruct_list)

            print('..........................................')

        # compute b and return it
        b = sess.run(beta)

        print('b count', sess.run(b_count))
        print('b is', b)

        print('Finish time: {0}'.format(datetime.datetime.now()))

    return b





