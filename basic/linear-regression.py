import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_points=1000
vectors_set=[]
for i in range(num_points):
    x1=np.random.normal(0.00,0.55)
    y1=x1*0.1+0.3+np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

x=[v[0] for v in vectors_set]
y=[v[1] for v in vectors_set]

plt.scatter(x, y, c='r')
plt.show()

W=tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')
b=tf.Variable(tf.zeros([1]),name='b')
y_pred= W * x + b

loss=tf.reduce_mean(tf.square(y-y_pred),name='loss')
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss,name='train')

sess=tf.Session()

init=tf.global_variables_initializer()
sess.run(init)

print('W=',sess.run(W),'b=',sess.run(b),'loss=',sess.run(loss))

for step in range(20):
    sess.run(train)
    print('W=',sess.run(W),'b=',sess.run(b),'loss=',sess.run(loss))

y_pred=sess.run(y_pred)

plt.scatter(x,y,c='r')
plt.plot(x,y_pred)
plt.show()