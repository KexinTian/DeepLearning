import numpy as np
np.random.seed(123)
import pandas as pd
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 10
import tensorflow as tf
tf.set_random_seed(123)
import keras
MY_DATASETSLIB_HOME ='F:/my_datasets'
import sys
if not MY_DATASETSLIB_HOME in sys.path:
    sys.path.append(MY_DATASETSLIB_HOME)
print("测试:数据集保存的根目录"+MY_DATASETSLIB_HOME)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(os.path.join(MY_DATASETSLIB_HOME, 'mnist'),
                                  one_hot=True)
print("测试:mnist数据集保存位置"+os.path.join(MY_DATASETSLIB_HOME, 'mnist'))
X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels

num_outputs = 10  # 0-9 digits
num_inputs = 784  # total pixels


def mlp(x, num_inputs, num_outputs, num_layers, num_neurons):
    w = []
    b = []
    for i in range(num_layers):
        # weights
        w.append(tf.Variable(tf.random_normal(
            [num_inputs if i == 0 else num_neurons[i - 1],
             num_neurons[i]]),
            name="w_{0:04d}".format(i)
        ))
        # biases
        b.append(tf.Variable(tf.random_normal(
            [num_neurons[i]]),
            name="b_{0:04d}".format(i)
        ))
    w.append(tf.Variable(tf.random_normal(
        [num_neurons[num_layers - 1] if num_layers > 0 else num_inputs,
         num_outputs]), name="w_out"))
    b.append(tf.Variable(tf.random_normal([num_outputs]), name="b_out"))

    # x is input layer
    layer = x
    # add hidden layers
    for i in range(num_layers):
        layer = tf.nn.relu(tf.matmul(layer, w[i]) + b[i])
    # add output layer
    layer = tf.matmul(layer, w[num_layers]) + b[num_layers]

    return layer


def mnist_batch_func(batch_size=100):
    X_batch, Y_batch = mnist.train.next_batch(batch_size)
    return [X_batch, Y_batch]


def tensorflow_classification(n_epochs, n_batches,
                              batch_size, batch_func,
                              model, optimizer, loss, accuracy_function,
                              X_test, Y_test):
    with tf.Session() as tfs:
        tfs.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in range(n_batches):
                X_batch, Y_batch = batch_func(batch_size)
                feed_dict = {x: X_batch, y: Y_batch}
                _, batch_loss = tfs.run([optimizer, loss], feed_dict)
                epoch_loss += batch_loss
            average_loss = epoch_loss / n_batches
            loss_epochs[epoch]=average_loss
            print("epoch: {0:04d}   loss = {1:0.6f}".format(
                epoch, average_loss))
            feed_dict = {x: X_test, y: Y_test}
            average_accuracy = tfs.run(accuracy_function, feed_dict=feed_dict)
            accuracy_epochs[epoch]=average_accuracy
            print("epoch: {0:04d}   accuracy = {1:0.6f}".format(
                epoch, average_accuracy))

        feed_dict = {x: X_test, y: Y_test}
        accuracy_score = tfs.run(accuracy_function, feed_dict=feed_dict)
        print("accuracy={0:.8f}".format(accuracy_score))


tf.reset_default_graph()#用于清除默认图形堆栈并重置全局默认图形，只适用于当前线程。
## 当一个tf.Session或者tf.InteractiveSession激活时调用这个函数会导致未定义的行为。调用此函数后使用任何以前创建的tf.Operation或tf.Tensor对象将导致未定义的行为。

num_layers = 0
num_neurons = []
learning_rate = 0.01
n_epochs = 50
loss_epochs=np.empty(shape=[n_epochs],dtype=float)
accuracy_epochs=np.empty(shape=[n_epochs],dtype=float)
batch_size = 100
n_batches = int(mnist.train.num_examples / batch_size)

# input images
x = tf.placeholder(dtype=tf.float32, name="x", shape=[None, num_inputs])
# target output
y = tf.placeholder(dtype=tf.float32, name="y", shape=[None, num_outputs])

model = mlp(x=x,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_layers=num_layers,
            num_neurons=num_neurons)

# loss function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
# optimizer function
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(loss)

predictions_check = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy_function = tf.reduce_mean(tf.cast(predictions_check, tf.float32))

tensorflow_classification(n_epochs=n_epochs,
                          n_batches=n_batches,
                          batch_size=batch_size,
                          batch_func=mnist_batch_func,
                          model=model,
                          optimizer=optimizer,
                          loss=loss,
                          accuracy_function=accuracy_function,
                          X_test=mnist.test.images,
                          Y_test=mnist.test.labels
                          )
plt.figure(figsize=(14, 8))

plt.axis([0, n_epochs, 0, np.max(loss_epochs)])
plt.plot(loss_epochs, label='Loss on X_train')
plt.title('LOSE in Iterations')
plt.xlabel('# Epoch')
plt.ylabel('LOSE ')
plt.legend()
plt.show()

plt.figure(figsize=(14, 8))
plt.axis([0, n_epochs, 0, 1])
plt.plot(accuracy_epochs, label='ACCURACY on X_test')
plt.title('ACCURACY in Iterations')
plt.xlabel('# Epoch')
plt.ylabel('ACCURACY')
plt.legend()
plt.show()