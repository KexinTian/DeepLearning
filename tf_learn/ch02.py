import tensorflow as tf
# 用于清除默认图形堆栈并重置全局默认图形
tf.reset_default_graph()

import tflearn
import tflearn.datasets.mnist as mnist
import os

batch_size = 100
n_classes = 10
n_epochs = 10

X_train, Y_train, X_test, Y_test = mnist.load_data(
    data_dir=os.path.join('.', 'mnist'), one_hot=True)

# Build deep neural network
input_layer = tflearn.input_data(shape=[None, 784])
layer1 = tflearn.fully_connected(input_layer,
                                 10,
                                 activation='relu'
                                 )
layer2 = tflearn.fully_connected(layer1,
                                 10,
                                 activation='relu'
                                 )
output = tflearn.fully_connected(layer2,
                                 n_classes,
                                 activation='softmax'
                                 )

net = tflearn.regression(output,
                         optimizer='adam',
                         metric=tflearn.metrics.Accuracy(),
                         loss='categorical_crossentropy'
                         )
model = tflearn.DNN(net)

model.fit(
    X_train,
    Y_train,
    n_epoch=n_epochs,
    batch_size=batch_size,
    show_metric=True,
    run_id='dense_model')

score = model.evaluate(X_test, Y_test)
print('Test accuracy:', score[0])