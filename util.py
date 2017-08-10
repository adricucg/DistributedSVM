import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math as math
import pandas as pd
from sklearn.model_selection import train_test_split

# Declare function to do reshape/batch multiplication
def reshape_matmul(mat, batch_size):

    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [10, batch_size, 1])

    return tf.matmul(v2, v1)


def prepare_mnist():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    batch_size = 55000
    X = mnist.train.images[0:batch_size]
    labels = mnist.train.labels[0:batch_size]

    X_test = mnist.test.images
    labels_test = mnist.test.labels

    y = np.zeros([10, batch_size], dtype = 'int32')
    for j in range(0, 10):
        for i in range(0, batch_size):
            if labels[i,j] > 0:
                y[j, i] = 1
            else:
                y[j, i] = -1

    y_test = np.zeros([10, len(X_test)], dtype = 'int32')
    for j in range(0, 10):
        for i in range(0,len(X_test)):
            if labels_test[i,j] > 0:
                y_test[j, i] = 1
            else:
                y_test[j, i] = -1

    return X, X_test, y, y_test


def prepare_forest():

    data = pd.read_csv('../FOREST_data/forest.csv', header=None)

    print('count: ', len(data))
    print(data[:][54])

    data_y = data[[54]]
    data_X = data.drop([54], axis=1)
    size = len(data)
    # size = 100
    #
    y = np.zeros(shape=(7, size))
    for i in range(0, size):
         print('iteration: ', i)
         for j in range(0, 7):
             if data_y.loc[i, 54] == (j + 1):
                 y[j, i] = 1
             else:
                 y[j, i] = -1

    for j in range(0, 7):
         data_X.loc[:, 54] =  pd.Series(y[j])
         data_X.to_csv("../FOREST_data/covtype_" + str(j + 1) + ".csv", index = False, header = False)


def load_forest(cls):

    data = pd.read_csv("FOREST_data/covtype_" + str(cls) + ".csv", header=None)
    data_y = data[[54]]
    data_X = data.drop([54], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test


# starts a Tensorflow server for a specific job name and task
def start_worker_server(jobname, workers, taskindex):

    cluster = tf.train.ClusterSpec({jobname: workers})
    server = tf.train.Server(cluster, job_name=jobname, task_index=taskindex)
    server.start()
    server.join()


# reads data either from local or HDFS and returns the samples between start and stop index
def load_data(start_index, stop_index, cls, type):

    if type == 'mnist':
        X, X_test, y, y_test = prepare_mnist()

        data_X = X[start_index : stop_index]
        data_y = y[cls, start_index: stop_index]

        # _x = np.array([[5, 1, 3], [1, 2, 0], [4, 4, 8],
        #                 [1, 1, 1], [0.5, 0.5, 0.5], [4, 2, 3],
        #                 [2, 1, 0],[1, 2, 0], [6, 7, 9]])
        #
        #
        # _y = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1])
        #
        # data_X = _x[start_index : stop_index]
        # data_y = _y[start_index : stop_index]

    if type == 'forest':
        X, X_test, y, y_test = load_forest(cls)

        data_X = (X[start_index: stop_index]).values
        data_y = (((y[start_index: stop_index]).values).reshape(-1, stop_index - start_index))[0]

    return data_X, data_y


def load_test_data(cls, type):

    if type == 'mnist':
        X, X_test, y, y_test = prepare_mnist()

        data_X_test = X_test
        data_y_test = y_test[cls]

    if type == 'forest':
        X, X_test, y, y_test = load_forest(cls)

        data_X_test = X_test
        data_y_test = y_test

    return data_X_test, data_y_test


def kernel_rbf(x_i_data, x_j_data):

    # Gaussian (RBF) kernel
    gamma = tf.constant(-0.02)
    rA = tf.reduce_sum(tf.square(x_i_data))
    rB = tf.reduce_sum(tf.square(x_j_data))
    pairwise = tf.reduce_sum(tf.multiply(2., tf.multiply(x_i_data, x_j_data)))
    sq_dists = tf.add(tf.subtract(rA, pairwise), rB)
    kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    return kernel


def matrix_kernel_rbf(X, Y):

    gamma = tf.constant(-0.02)
    rA_matrix = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1, 1])
    rB_matrix = tf.reshape(tf.reduce_sum(tf.square(Y), 1), [-1, 1])
    pred_sq_dist = tf.add(tf.subtract(rA_matrix, tf.multiply(2., tf.matmul(X, tf.transpose(Y)))),
                          tf.transpose(rB_matrix))
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

    return pred_kernel


def compute_alpha(alpha_up, alpha_low, y_up, y_low, beta_up, beta_low, eta, C, eps):

    ZERO = 1e-12
    if math.fabs(y_up - y_low) > 0:
        L = (alpha_low - alpha_up) if (alpha_low - alpha_up) > ZERO else 0.0
        H = (C + alpha_low - alpha_up) if (C + alpha_low - alpha_up) < C else C
    else:
        L = (alpha_low + alpha_up - C) if (alpha_low + alpha_up - C) > ZERO else 0.0
        H = (alpha_low + alpha_up) if (alpha_low + alpha_up < C) else C

    print('L: ', L)
    print('H: ', H)

    if math.fabs(L - H) < ZERO:
        print(' L and H smaller than 0 {0}, {1}'.format(L, H))
        return 0,0

    if eta < 0:
        alpha_low_new = alpha_low - y_low * (beta_up - beta_low) / eta

        if alpha_low_new < L:
            print('alpha_low_new {0} is smaller than L {1}!'.format(alpha_low_new, L))
            alpha_low_new = L
        elif alpha_low_new > H:
            print('alpha_low_new {0} is bigger than H {1}!'.format(alpha_low_new, H))
            alpha_low_new = H
    else:
        print('Eta is positive!: {0}'.format(eta))
        return 0,0
        # compart = y_low * (beta_up - beta_low) - eta * alpha_low
        # Lobj = 0.5 * eta * L * L + compart * L
        # Hobj = 0.5 * eta * H * H + compart * H
        #
        # if Lobj > Hobj + eps:
        #     alpha_low_new = L
        # elif Lobj < Hobj - eps:
        #     alpha_low_new = H
        # else:
        #     alpha_low_new = alpha_low

    # Push alpha_low_new to 0 or C if very close
    if alpha_low_new < ZERO:
        alpha_low_new = 0.0
    elif alpha_low_new > (C - ZERO):
        alpha_low_new = C

    delalpha_low = alpha_low_new - alpha_low

    if math.fabs(delalpha_low) < eps * (alpha_low_new + alpha_low + eps):
        print('fabs quit')
        print('delta alpha low:{0} alpha_low_new:{1} alpha_low:{2} L:{3} H:{4}'.format(
              delalpha_low, alpha_low_new, alpha_low, L, H))

        return 0,0

    # alpha up new value
    alpha_up_new = alpha_up + y_up * y_low * (alpha_low - alpha_low_new)

    if alpha_up_new < ZERO:
        print('alpha up new is negative: ', alpha_up_new)
        alpha_low_new = alpha_low_new + y_up * y_low * alpha_up_new
        alpha_up_new = 0.0

    elif alpha_up_new > (C - ZERO):
        print('alpha up new is bigger than C: ', alpha_up_new)
        t = alpha_up_new - C
        alpha_low_new = alpha_low_new + y_up * y_low * t
        alpha_up_new = C

    return alpha_up_new, alpha_low_new


def compute_b_local(worker_index, C):

    gradient = tf.get_variable('gradient_old_' + str(worker_index))
    alpha = tf.get_variable('alpha_' + str(worker_index))

    fn = lambda a: tf.case(
        pred_fn_pairs=[
            (tf.logical_and(tf.greater(a, 0.0), tf.less(a, C)), lambda: tf.constant(0))
        ], default=lambda: tf.constant(-1))

    set_indexes = tf.map_fn(fn, alpha, dtype=tf.int32)

    i_0 = tf.where(tf.equal(set_indexes, 0))
    gradient_i_0 = tf.gather(gradient, i_0)

    gradient_sum = tf.reduce_sum(gradient_i_0)

    count = tf.size(i_0)

    return gradient_sum, count
