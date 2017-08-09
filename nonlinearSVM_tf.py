import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import *

# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.Session()

X_train, X_test, y_train, y_test = prepare_mnist()

# Declare batch size
batch_size = 50

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_target = tf.placeholder(shape=[10, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 784], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[10,batch_size]))

# Gaussian (RBF) kernel
gamma = tf.constant(-10.0)

# dist is x1*x1 + x2*x2 + ...
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
rbf_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Compute SVM Model
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target, batch_size)
second_term = tf.reduce_sum(tf.multiply(rbf_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                      tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
prediction = tf.arg_max(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))

# Declare optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
batch_accuracy = []
for i in range(500):
    rand_index = np.random.choice(len(X_train), size=batch_size)
    rand_x = X_train[rand_index]
    rand_y = y_train[:, rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid: rand_x})
    batch_accuracy.append(acc_temp)

    if (i + 1) % 25 == 0:
        print('Step #' + str(i + 1))
        print('Loss = ' + str(temp_loss))


print(batch_accuracy)


print('the svm prediction:' ,
      prediction.eval(feed_dict={
          x_data: np.reshape(mnist.test.images[1000], (1,784)),
          y_target: y_test[:, 1000],
          prediction_grid:  np.reshape(mnist.test.images[1000], (1,784)) },
          session = sess))
print('the actual label: ', mnist.test.labels[1000])