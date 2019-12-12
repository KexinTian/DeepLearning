import os
import sys
import keras
from keras.models import  Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Reshape
from keras.optimizers import SGD
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

"""定义每一层的滤波器数量"""
n_filters=[32,64]
"""其他超参数"""
n_classes=10
n_width=28
n_height=28
n_depth=1
n_input=n_width*n_height*n_depth#总像素784
learning_rate=0.01
n_epochs=20
batch_size=100

"""定义序列模型:卷积层-》池化层—》卷积层-》池化层-》全连接层—》输出层"""
model=Sequential()

model.add(Reshape(target_shape=(n_width,n_height,n_depth),input_shape=(n_input,)))
model.add(Conv2D(filters=n_filters[0],kernel_size=4,padding="SAME",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=n_filters[1],kernel_size=4,padding="SAME",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())#Flatten，把多维的输入一维化，常用在卷基层到全连接层的过渡
model.add(Dense(units=1024,activation="relu"))

model.add(Dense(units=n_classes,activation="softmax"))

model.summary()
"""编译、训练、评估"""
model.compile(loss="categorical_crossentropy",
              optimizer=SGD(lr=learning_rate),
              metrics=["accuracy"])
model.fit(X_train,Y_train,batch_size=batch_size,epochs=n_epochs)
score=model.evaluate(X_test,Y_test)
print("Test loss: ",score[0])
print("Test Accuracy: ",score[1])