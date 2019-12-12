import os
import sys
import tensorflow as tf

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
n_width=28
n_height=28
n_depth=1
n_input=n_width*n_height*n_depth#总像素784
learning_rate=0.001
n_epochs=10
batch_size=100
n_batchs=int(mnist.train.num_examples/batch_size)
""""
输入图形形状,并reshape为模型需要的形状
"""
x=tf.placeholder(dtype=tf.float32,name="x",shape=[None,n_input])
x_=tf.reshape(x,shape=[-1,n_width,n_height,n_depth])
"""
输出标签,因为onehot编码，所以有10位
"""
y=tf.placeholder(dtype=tf.float32,name="y",shape=[None,n_classes])
"""
使用32个4×4大小的核定义第一个卷积层，得到32个特征图
（核参数通过正态分布初始化；卷基层添加了relu激活函数）
第一个卷积层输出32×28×28*1
"""
layer1_w=tf.Variable(tf.random_normal(shape=[4,4,n_depth,32],stddev=0.1),name="11_w")
layer1_b=tf.Variable(tf.random_normal([32]),name="11_b")
layer1_conv=tf.nn.relu(tf.nn.conv2d(x_,layer1_w,strides=[1,1,1,1],padding="SAME")+layer1_b)
"""
定义第一个池化层,ksize表示在2×2×1的区域上进行池化。strides=ksize保证区域彼此不重叠
第一个池化层的输出32×14×14×1
"""
layer1_pool=tf.nn.max_pool(layer1_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
"""
定义第二个卷积层，64个核。生成64个特征图
输出64×14×14×1
"""
layer2_w=tf.Variable(tf.random_normal(shape=[4,4,32,64],stddev=0.1),name="12_w")
layer2_b=tf.Variable(tf.random_normal([64]),name="12_b")
layer2_conv=tf.nn.relu(tf.nn.conv2d(layer1_pool,layer2_w,strides=[1,1,1,1],padding="SAME")+layer2_b)
"""
定义第二个池化层
输出64×7×7×1
"""
layer2_pool=tf.nn.max_pool(layer2_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
""""
定义全连接层，全连接层有1024个神经元
在输入进全连接层之前，需要将其拉伸为大小为1024的向量
"""
layer3_w=tf.Variable(tf.random_normal(shape=[64*7*7*1,1024],stddev=0.1),name="13_w")
layer3_b=tf.Variable(tf.random_normal([1024]),name="13_b")
layer3_fc=tf.nn.relu(tf.matmul(tf.reshape(layer2_pool,[-1,64*7*7*1]),layer3_w)+layer3_b)
"""
全连接层后面与线型输出层相连
这一层没有使用softmax，因为损失函数会将softmax自动应用于输出
"""
layer4_w=tf.Variable(tf.random_normal(shape=[1024,n_classes],stddev=0.1),name="1")
layer4_b=tf.Variable(tf.random_normal([n_classes]),name="14_b")
layer4_out=tf.matmul(layer3_fc,layer4_w)+layer4_b
model=layer4_out


entropy=tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=y)
loss=tf.reduce_mean(entropy)
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as tfs:
    tf.global_variables_initializer().run()
    for epoch in range (n_epochs):
        total_loss=0.0
        for batch in range(n_batchs):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed_dict = {x: batch_x, y: batch_y}
            batch_loss, _ = tfs.run([loss, optimizer], feed_dict=feed_dict)
            total_loss += batch_loss
            print("epoch:{0} test:{1} \n".format(epoch,batch) )
        average_loss=total_loss/n_batchs
        print("Epoch: {0:04d} loss= {1:0.6f} ".format(epoch,average_loss))
    print("Model Trained")

    predictions_check=tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(predictions_check, tf.float32))
    feed_dict={x:mnist.test.images, y:mnist.test.labels}
    print("Accuracy: ",accuracy.eval(feed_dict=feed_dict))
