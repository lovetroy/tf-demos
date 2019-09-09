import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('data/',one_hot=True)


def train_one_layer():
    '''
    one layer neural network
    :return:
    '''
    # parameter settings
    class_num = 10
    input_size = 784
    hidden_layer_units = 50
    training_iter = 10000
    batch_size = 100

    X = tf.placeholder(tf.float32, shape=[None, input_size])
    y = tf.placeholder(tf.float32, shape=[None, class_num])

    # parameter init
    W1 = tf.Variable(tf.truncated_normal([input_size, hidden_layer_units], stddev=0.1))
    B1 = tf.Variable(tf.constant(0.1), [hidden_layer_units])
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_units, class_num], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1), [class_num])

    # neural network structure
    hidden_layer_output = tf.matmul(X, W1) + B1
    hidden_layer_output = tf.nn.relu(hidden_layer_output)
    final_output = tf.matmul(hidden_layer_output, W2) + B2
    final_output = tf.nn.relu(final_output)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=final_output))
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(final_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(training_iter):
        batch = mnist.train.next_batch(batch_size)
        batch_input = batch[0]
        batch_label = batch[1]
        _, training_loss = sess.run([opt, loss], feed_dict={X: batch_input, y: batch_label})
        if i % 1000 == 0:
            train_accuray = accuracy.eval(session=sess, feed_dict={X: batch_input, y: batch_label})
            print('step %d , training accuracy %g' % (i, train_accuray))


def train_two_layer():
    # parameter settings
    class_num = 10
    input_size = 784
    hidden_layer1_units = 50
    hidden_layer2_units = 100
    training_iter = 10000
    batch_size = 100

    X = tf.placeholder(tf.float32, shape=[None, input_size])
    y = tf.placeholder(tf.float32, shape=[None, class_num])

    # parameter init
    W1 = tf.Variable(tf.truncated_normal([input_size, hidden_layer1_units], stddev=0.1))
    B1 = tf.Variable(tf.constant(0.1), [hidden_layer1_units])
    W2 = tf.Variable(tf.truncated_normal([hidden_layer1_units, hidden_layer2_units], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1), [hidden_layer2_units])
    W3 = tf.Variable(tf.truncated_normal([hidden_layer2_units, class_num], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1), [class_num])

    # neural network structure
    hidden_layer1_output = tf.matmul(X, W1) + B1
    hidden_layer1_output = tf.nn.relu(hidden_layer1_output)
    hidden_layer2_output = tf.matmul(hidden_layer1_output,W2)+B2
    hidden_layer2_output = tf.nn.relu(hidden_layer2_output)
    final_output = tf.matmul(hidden_layer2_output, W3) + B3

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=final_output))
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(final_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(training_iter):
        batch = mnist.train.next_batch(batch_size)
        batch_input = batch[0]
        batch_label = batch[1]
        _, training_loss = sess.run([opt, loss], feed_dict={X: batch_input, y: batch_label})
        if i % 1000 == 0:
            train_accuray = accuracy.eval(session=sess, feed_dict={X: batch_input, y: batch_label})
            print('step %d , training accuracy %g' % (i, train_accuray))

    # test
    test_input=mnist.test.images
    test_label=mnist.test.labels
    test_accuracy=accuracy.eval(session=sess,feed_dict={X:test_input,y:test_label})
    print('test accuracy ',test_accuracy)


if __name__=='__main__':
    # train_one_layer()
    train_two_layer()