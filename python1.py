import tensorflow as tf
import numpy as np

x_data=np.random.rand(100).astype(np.float32)
print('1--',x_data)
y_data=x_data*0.1+0.3
print('2--',y_data)
sess=tf.Session()
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
print('3--',Weights)
biases=tf.Variable(tf.zeros([1]))
print('4--',biases)

y=Weights*x_data+biases
print('5--',y)

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init = tf.initialize_all_variables()

#with tf.Session() as sess:
sess.run(init)
for _ in range(200):
    sess.run(train)
    if _ % 20 ==0:
        print(_,sess.run(Weights),sess.run(biases))

