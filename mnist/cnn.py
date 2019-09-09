import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('data/',one_hot=True)

class_num=10

tf.reset_default_graph()
sess=tf.InteractiveSession()


# None x height x width x color channels
X=tf.placeholder('float',shape=[None,28,28,1])
# None x number of classes
y=tf.placeholder('float',shape=[None,class_num])


def conv2d(X,W):
    '''
    strides:batch size x height x width x channels
    padding:SAME 补充 VALID 忽略
    :param X:
    :param W:
    :return:
    '''
    return tf.nn.conv2d(input=X,filter=W,strides=[1,1,1,1],padding='SAME')


def max_pool_2x2(X):
    '''
    ksize:batch size x height x width x channels
    :param X:
    :return:
    '''
    return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# first conv and pool layer
# 卷积核：filter x filter x input channels x output channels(feature map num)
W_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
# b must match output channels of the filter
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))
# 非线性映射
h_conv1=tf.nn.relu(conv2d(X,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

# second conv and pool layer
W_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

# first full connected layer
# 经过两次卷积和池化 原始输入由 28*28*1 变为 7*7*64
W_fc1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))
# 将输入拉长
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# dropout layer
keep_prob=tf.placeholder('float')
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

# second fully connected layer
W_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))

# final layer
y_pred=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

# 交叉熵损失
cross_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pred))
train_step=tf.train.AdamOptimizer().minimize(cross_entropy_loss)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

sess.run(tf.global_variables_initializer())

batch_size=50
for i in range(1000):
    batch=mnist.train.next_batch(batch_size)
    training_input=batch[0].reshape([batch_size,28,28,1])
    training_label=batch[1]
    train_step.run(session=sess, feed_dict={X:training_input,y:training_label,keep_prob:1.0})
    if i%100==0:
        train_accuracy=accuracy.eval(session=sess,feed_dict={X:training_input,y:training_label,keep_prob:1.0})
        print('step  %d , training accuracy %g '  % (i,train_accuracy))
