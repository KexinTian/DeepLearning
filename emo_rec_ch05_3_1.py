from time import strftime, gmtime

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import utils  

FLAGS=tf.app.flags.FLAGS
import sys
from absl import  app
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

tf.app.flags.DEFINE_string("MY_DATASETSLIB_HOME", "../my_datasets", "数据集的名称，与MY_DATASETSLIB_HOME中的子文件夹相对应")
tf.app.flags.DEFINE_string("dataset", "casiadatabase", "数据集的名称，与MY_DATASETSLIB_HOME中的子文件夹相对应")
tf.app.flags.DEFINE_integer("random_seed", "123","情感类别数目")
tf.app.flags.DEFINE_integer("num_classes", "6","情感类别数目")
tf.app.flags.DEFINE_integer("num_features", "384","情感特征维数")


if not os.path.isdir( FLAGS.MY_DATASETSLIB_HOME):
    print(FLAGS.MY_DATASETSLIB_HOME,"不存在")
filename = os.path.join(
        os.getcwd(), "log-"+strftime("%Y-%m-%d", gmtime()) + "-ch05_3_1.txt")
sys.stdout = utils.Logger(filename)#原来sys.stdout指向控制台，现在重定向到文件
"""
 打开语音数据集
 """
dataset_path = os.path.join(FLAGS.MY_DATASETSLIB_HOME, FLAGS.dataset)
print(" 数据集保存位置: " + dataset_path)
loc = os.path.join(dataset_path,   FLAGS.dataset+"_data.txt")# 这个是提取出来的标签和语音特征所在的txt文件
data = np.loadtxt(loc, skiprows=0)

"""
生成训练集和测试集
"""
y = data[:, 0]
x = data[:, 1:]
"""
对情感语音数据集进行预处理
 """
x = np.array(x).reshape(len(x), 384)
x=preprocessing.MinMaxScaler().fit_transform(x)
x.mean(axis=0)
"""
one_hot编码
"""
y=np.eye(FLAGS.num_classes)[y.astype(int)]
np.random.seed(FLAGS.random_seed)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

num_outputs = FLAGS.num_classes  # 输出类别数
num_inputs = FLAGS.num_features  # total pixels


tf.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形，只适用于当前线程。
## 当一个tf.Session或者tf.InteractiveSession激活时调用这个函数会导致未定义的行为。调用此函数后使用任何以前创建的tf.Operation或tf.Tensor对象将导致未定义的行为。

num_layers = 0
num_neurons = []
learning_rate = 0.01
n_epochs = 50
batch_size = 100
n_batches = int(len(X_train) / batch_size)
loss_epochs = np.empty(shape=[n_epochs], dtype=float)
accuracy_epochs = np.empty(shape=[n_epochs], dtype=float)

# input images
x = tf.placeholder(dtype=tf.float32, name="x", shape=[None, num_inputs])
# target output
y = tf.placeholder(dtype=tf.float32, name="y", shape=[None, num_outputs])

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



def batch_func(start,end):
    """
    需要自定义批处理函数
    :param start: 每batch开始的位置
    :param: end:   每batch结束的位置
    :return:X_batch, Y_batch，其中Y_batch为热编码以后
    """
    X_batch=X_train[start:end]
    Y_batch=Y_train[start:end]
    print ("Y_train.shape: {0};    Y_batch.shape: {1}".format(Y_train.shape,Y_batch.shape))


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
                start=batch*batch_size
                end=min((batch+1)*batch_size,len(X_train))
                X_batch, Y_batch = batch_func(start,end)
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

def main(argv):

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
                              batch_func=batch_func,
                              model=model,
                              optimizer=optimizer,
                              loss=loss,
                              accuracy_function=accuracy_function,
                              X_test=X_test,
                              Y_test=Y_test
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

if __name__ == "__main__":
  app.run(main)