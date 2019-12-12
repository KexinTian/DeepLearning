from time import strftime, gmtime

import numpy as np
import os
import tensorflow as tf

from utils import utils
FLAGS=tf.app.flags.FLAGS
import sys
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import tflearn


tf.app.flags.DEFINE_string("dataset", "casiadatabase", "数据集的名称，与MY_DATASETSLIB_HOME中的子文件夹相对应")
tf.app.flags.DEFINE_integer("random_seed", "123","情感类别数目")
tf.app.flags.DEFINE_integer("num_classes", "6","情感类别数目")
tf.app.flags.DEFINE_integer("num_features", "384","情感特征维数")

MY_DATASETSLIB_HOME ="F:/my_datasets"
if not MY_DATASETSLIB_HOME in sys.path:
    sys.path.append(MY_DATASETSLIB_HOME)
print(" 数据集保存的根目录"+MY_DATASETSLIB_HOME)
filename = os.path.join(
        os.getcwd(), "log-"+strftime("%Y-%m-%d", gmtime()) + "-ch05_3_3.txt")
sys.stdout = utils.Logger(filename)#原来sys.stdout指向控制台，现在重定向到文件
"""
 打开语音数据集
 """
dataset_path = os.path.join(MY_DATASETSLIB_HOME, FLAGS.dataset)
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


tf.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形，只适用于当前线程。
## 当一个tf.Session或者tf.InteractiveSession激活时调用这个函数会导致未定义的行为。调用此函数后使用任何以前创建的tf.Operation或tf.Tensor对象将导致未定义的行为。
num_outputs = FLAGS.num_classes  # 输出类别数
num_inputs = FLAGS.num_features  # total pixels
num_layers = 2
num_neurons = []
for i in range(num_layers):
    num_neurons.append(256)
learning_rate = 0.01
n_epochs = 50
batch_size = 100


input_layer = tflearn.input_data(shape=[None, num_inputs])
dense1 = tflearn.fully_connected(input_layer, num_neurons[0], activation='relu')
dense2 = tflearn.fully_connected(dense1, num_neurons[1], activation='relu')
softmax = tflearn.fully_connected(dense2, num_outputs, activation='softmax')
optimizer = tflearn.SGD(learning_rate=learning_rate)
net = tflearn.regression(softmax, optimizer=optimizer,
                         metric=tflearn.metrics.Accuracy(),
                         loss='categorical_crossentropy')
model = tflearn.DNN(net)


model.fit(X_train,Y_train,
          n_epoch=n_epochs, batch_size=batch_size,
          show_metric=True, run_id='dense_model')
score=model.evaluate(X_test,Y_test)
print("Test accuracy:{0}".format(score[0]))

