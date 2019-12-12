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
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

tf.app.flags.DEFINE_string("dataset", "casiadatabase", "数据集的名称，与MY_DATASETSLIB_HOME中的子文件夹相对应")
tf.app.flags.DEFINE_integer("random_seed", "123","情感类别数目")
tf.app.flags.DEFINE_integer("num_classes", "6","情感类别数目")
tf.app.flags.DEFINE_integer("num_features", "384","情感特征维数")

MY_DATASETSLIB_HOME ="F:/my_datasets"
if not MY_DATASETSLIB_HOME in sys.path:
    sys.path.append(MY_DATASETSLIB_HOME)
print(" 数据集保存的根目录"+MY_DATASETSLIB_HOME)
filename = os.path.join(
        os.getcwd(), "log-"+strftime("%Y-%m-%d", gmtime()) + "-ch05_3_2.txt")
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


model=Sequential()#创建一个顺序模型
model.add(Dense(units=num_neurons[0],activation='relu',input_shape=(num_inputs,)))#添加一个隐藏层
model.add(Dense(units=num_neurons[1], activation='relu' ) ) # 添加第二层隐藏层
model.add(Dense(units=num_outputs, activation='softmax'))  # 添加具有激活函数的输出层
model.summary()#输出模型的详细信息
model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=learning_rate),
                  metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=batch_size,epochs=n_epochs)
score=model.evaluate(X_test,Y_test)
print("Test loss:{0};   Test accuracy:{1}".format(score[0],score[1]))

