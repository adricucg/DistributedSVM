from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

rbf_svc = svm.SVC(kernel='rbf')

print(type(mnist.train.images[1,:]))
print(mnist.train.images[1,:])

print('...........................')
print(mnist.train.labels[0:10])

batch_size = 55000
X = mnist.train.images[0:batch_size]
labels = mnist.train.labels[0:batch_size]

X_test = mnist.test.images
labels_test = mnist.test.labels

y = np.zeros(batch_size, dtype = 'int32')
for i in range(0,batch_size):
    for j in range(0,10):
        if labels[i,j] > 0:
            y[i] = j


y_test = np.zeros(10000, dtype = 'int32')
for i in range(0,10000):
    for j in range(0,10):
        if labels_test[i,j] > 0:
            y_test[i] = j

rbf_svc.fit(X, y)

print('mean score is: ', rbf_svc.score(X_test, y_test))
print('the svm prediction:' , rbf_svc.predict(mnist.test.images[1050]))
print('the actual label: ', mnist.test.labels[1050])


