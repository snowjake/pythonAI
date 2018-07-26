import tensorflow as tf
import numpy as np
#add layer
def add_layer(inputs,in_size,out_size,activation_function=None):    
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs


x_data=np.linspace(-1,1,300)[:,np.newaxis]
#print('1--',x_data)
noise =np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise
#print('2--',y_data)
with tf.name_scope('input'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')
#one layer
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
#two layer
prediction=add_layer(l1,10,1,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess=tf.Session()
writer=tf.summary.FileWriter("d://logs/",graph=sess.graph)
#writer.add_graph(sess.graph)
init = tf.initialize_all_variables()
sess.run(init)

