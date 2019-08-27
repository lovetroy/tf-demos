import tensorflow as tf
import numpy as np

# https://www.lfd.uci.edu/~gohlke/pythonlibs

# print(tf.__version__)

# 创建遍量
a=3.0
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
b = tf.Variable([a])
y=tf.matmul(w,x)
# y=tf.add(y,b)

a1=tf.zeros([3,4],tf.int32)
a2=tf.ones([2,3],tf.int32)
a3=tf.zeros_like(a2)
a4=tf.ones_like(a3)
a5=tf.constant([1,2,3])
a6=tf.constant(-1.0,shape=[2,3])
a7=tf.linspace(10.0,12.0,3,name='linspace')
a8=tf.range(0,8,2)

norm=tf.random_normal([2,3],mean=-1,stddev=4)
shuff=tf.random_shuffle(a5)

state = tf.Variable(0)
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

n=np.zeros((3,3))
print(n)
t=tf.convert_to_tensor(n)

a10=tf.constant(5.0)
a11=tf.constant(10.0)
a12=tf.add(a10,a11,name='add')
a13=tf.div(a10,a11,name='divide')
a14=a10+a11
a15=a10/a11

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # print(y.eval())
    # print(b.eval())

    print(a1.eval())
    print(a2.eval())
    print(a3.eval())
    print(a4.eval())
    print(a5.eval())
    print(a6.eval())
    print(a7.eval())
    print(a8.eval())

    print(norm.eval())
    print(shuff.eval())

    print(sess.run(state),sess.run(new_value))
    sess.run(update)
    print(sess.run(state),sess.run(new_value))

    print(sess.run(t))

    print('a10=',sess.run(a10))
    print('a11=',sess.run(a11))
    print('a10+a11=',sess.run(a12))
    print('a10+a11=', sess.run(a14))
    print('a10/a11=',sess.run(a13))
    print('a10/a11=',sess.run(a15))

    print(sess.run([output],feed_dict={input1:[7.0],input2:[2.0]}))