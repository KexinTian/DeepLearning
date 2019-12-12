import os
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import RMSprop,SGD

MY_DATASETSLIB_HOME ='F:/my_datasets'

if not MY_DATASETSLIB_HOME in sys.path:
    sys.path.append(MY_DATASETSLIB_HOME)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(os.path.join(MY_DATASETSLIB_HOME, 'mnist'),
                                  one_hot=True)
print("测试:mnist数据集保存位置"+os.path.join(MY_DATASETSLIB_HOME, 'mnist'))
X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels
n_classes=10
"""
以为数据784像素转换成二维结构28×28，模拟28个时间步长，每个步长具有28个像素
"""
X_train=X_train.reshape(-1,28,28)
X_test=X_test.reshape(-1,28,28)
"""创建和拟合 SimpleRNN 模型
"""
model=Sequential()
model.add(SimpleRNN(units=16,activation='relu',input_shape=(28,28)))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])
model.summary()

"""训练模型并输出测试数据集的分类精度
"""
model.fit(X_train, Y_train,
          batch_size=100, epochs=20)
score=model.evaluate(X_test, Y_test )
print("test  loss: {0} \n".format(score[0]))
print("test  accuracy: {0} \n".format(score[1]))


