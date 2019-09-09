import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

sess=tf.Session()
mnist=input_data.read_data_sets('data/',one_hot=True)
print(mnist.train.images.shape)

lr=1e-3
# 每行输入大小
input_size=28
# 行数即为序列数
timestep_size=28
hidden_layer_units=256
layer_num=2
class_num=10

X=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,class_num])

batch_size=tf.placeholder(tf.int32,[])
keep_prob=tf.placeholder(tf.float32,[])

_X=tf.reshape(X,[-1,28,28])


def lstm_cell():
    cell=rnn.LSTMCell(hidden_layer_units,reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)

mlstm_cell=tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)],state_is_tuple=True)

# 用 0 初始化
init_state=mlstm_cell.zero_state(batch_size,dtype=tf.float32)

outputs=[]
state=init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
            # _X[batch size,row,col]
        (cell_output,state)=mlstm_cell(_X[:,timestep,:],state)
        outputs.append(cell_output)
# 最后一次递归的结果
h_state=outputs[-1]

# softmax 层参数
W = tf.Variable(tf.truncated_normal([hidden_layer_units,class_num],stddev=0.1))
bias=tf.Variable(tf.constant(0.1,shape=[class_num]),dtype=tf.float32)
y_pre=tf.nn.softmax(tf.matmul(h_state,W)+bias)

cross_entropy=-tf.reduce_mean(y*tf.log(y_pre))
train_op=tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

# train
# init
def train():
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)
        sess.run(train_op, feed_dict={
            X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})
        if (i + 1) % 200 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
            print('iter %d , step %d, training accuracy %g' %
                  (mnist.train.epochs_completed, (i + 1), train_accuracy))

    # test
    print('test accuracy %g' % sess.run(accuracy, feed_dict={
        X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size: mnist.test.images.shape[0]}))


def single():
    '''
    单个图像每层的结果
    :return:
    '''
    sess.run(tf.global_variables_initializer())

    _batch_size = 5
    X_batch, y_batch = mnist.test.next_batch(_batch_size)
    print(X_batch.shape, y_batch.shape)
    _outputs, _state = sess.run([outputs, state], feed_dict={
        X: X_batch, y: y_batch, keep_prob: 1.0, batch_size: _batch_size})
    # _outputs: timestep x batch size x feature
    print('_outputs.shape =', np.asarray(_outputs).shape)

    print(mnist.train.labels[4])
    X3=mnist.train.images[4]
    img3=X3.reshape([28,28])
    plt.imshow(img3,cmap='gray')
    plt.show()

    X3.shape=[-1,784]
    y_batch=mnist.train.labels[0]
    y_batch.shape=[-1,class_num]
    X3_outputs=np.array(sess.run(outputs,feed_dict={X:X3,y:y_batch,keep_prob:1.0,batch_size:1}))
    print(X3_outputs.shape)
    X3_outputs.shape=[28,hidden_layer_units]
    print(X3_outputs.shape)

    h_W=sess.run(W,feed_dict={
        X:X3,y:y_batch,keep_prob:1.0,batch_size:1})
    h_bias=sess.run(bias,feed_dict={
        X:X3,y:y_batch,keep_prob:1.0,batch_size:1})
    h_bias.shape=[-1,10]

    bar_index=range(class_num)
    for i in range(X3_outputs.shape[0]):
        plt.subplot(7,4,i+1)
        X3_h_state=X3_outputs[i,:].reshape([-1,hidden_layer_units])
        pro=sess.run(tf.nn.softmax(tf.matmul(X3_h_state,h_W)+h_bias))
        plt.bar(bar_index,pro[0],width=0.2,align='center')
        plt.axis('off')
    plt.show()



if __name__=='__main__':
    # train()
    single()