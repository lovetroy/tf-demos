from numpy.core.multiarray import ndarray
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('data/', one_hot=True)

class_num = 10
input_size = 784
training_iter = 2000
batch_size = 64

X = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, class_num])

W1 = tf.Variable(tf.random_normal([input_size, class_num], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [class_num])

y_pred = tf.nn.softmax(tf.matmul(X, W1) + B1)
loss = tf.reduce_mean(tf.square(y - y_pred))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


def train():
    # train
    for i in range(training_iter):
        batch=mnist.train.next_batch(batch_size)
        batch_input=batch[0]
        batch_label=batch[1]
        _,training_loss=sess.run([opt,loss],feed_dict={X:batch_input,y:batch_label})
        if i%1000==0:
            train_accuracy=accuracy.eval(session=sess,feed_dict={X:batch_input,y:batch_label})
            print('step %d, training accuracy %g' % (i,train_accuracy))


def test_batch():
    # test
    batch=mnist.test.next_batch(batch_size)
    test_accuracy=sess.run(accuracy,feed_dict={X:batch[0],y:batch[1]})
    print('test accuracy %g' % (test_accuracy))


def test():
    # test
    for i in range(1,5):
        cur_img = np.reshape(mnist.test.images[i, :], (28, 28))
        cur_label = np.argmax(mnist.test.labels[i, :])
        plt.matshow(cur_img, cmap=plt.get_cmap('gray'))
        y_pred_vec = sess.run(y_pred,feed_dict={X:np.array([mnist.test.images[i,:]])})
        print('y_pred_vec',y_pred_vec)
        print(str(i), ' th label is ',str(cur_label),' predict is ', str(np.argmax(y_pred_vec)))
        # plt.show()

# TODO softmax 向量归一化

def test_argmax():
    '''
    输出某个维度最大值索引
    :return:
    '''
    # argmax
    data = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
    # axis=0 输出每列最大值索引 [3 3 1]
    print(np.argmax(data, 0))
    # axis=1输出每行最大值索引 [2 2 0 0]
    print(np.argmax(data, 1))


def test_equal():
    '''
    逐个元素进行判断，如果相等就是True，不相等，就是False
    :return: [[ True False  True]
              [False  True False]]
    '''
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[1, 0, 3], [1, 5, 1]]
    with tf.Session() as sess:
        print(sess.run(tf.equal(a, b)))


def test_softmax():
    a=[1.0,2.0,3.0]
    with tf.Session() as sess:
        print(sess.run(tf.nn.softmax(a)))


def test_reduce_mean():
    '''
    计算张量tensor沿着指定的数轴（tensor的某一维度）上的平均值
    reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None)
    :return:
    '''
    pass

if __name__=='__main__':
    # test_argmax()
    # test_equal()
    test_softmax()