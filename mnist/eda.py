import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('data/',one_hot=True)

print('data set type %s' % (type(mnist)))
print('training data %d' % (mnist.train.num_examples))
print('testing data %d' % (mnist.test.num_examples))

print('data type %s' % (type(mnist.train.images)))
print('label type %s' % (type(mnist.train.labels)))
print('training data shape %s' % (mnist.train.images.shape,))
print('training label shape %s' % (mnist.train.labels.shape,))

nsample=5
randids=np.random.randint(mnist.train.images.shape[0],size=nsample)

for i in randids:
    cur_img=np.reshape(mnist.train.images[i,:],(28,28))
    cur_label=np.argmax(mnist.train.labels[i,:])
    plt.matshow(cur_img,cmap=plt.get_cmap('gray'))
    print(str(i),'th label is',str(cur_label))
    plt.show()

# Batch Learning
batch_size=100
batch_xs,batch_ys=mnist.train.next_batch(batch_size)
print('Batch data %s' % (type(batch_xs)))
print('Batch label %s' % (type(batch_ys)))
print('Batch data shape %s' % (batch_xs.shape,))
print('Batch label shape %s' % (batch_ys.shape,))