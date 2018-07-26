import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# 下载MNIST数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)#download mnist.train(6W) and mnist.test(1W) data

#add layer  | 构造神经网络(回归模型):tf.matmul(xs,W)+b
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

# 计算准确率
#首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，
#它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
#由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，
#比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，
#我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#define placeholder for input to network  | 构造输入输出数据结构
xs=tf.placeholder(tf.float32,[None,784],name='x_input')#28*28
ys=tf.placeholder(tf.float32,[None,10],name='y_input')
#keep_prob = tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1])
#print(x_image.shape) #[n_samples,28,28,chnnel(1)黑白色就一层]

#conv1 layer
W_conv1=weight_variable([5,5,1,32])#patch 5x5 ,in size 1,out size 32  卷积取一部分  1是图片厚度 32是长度
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#hidden conv2d layer one #output size 28x28x32
h_pool1=max_pool_2x2(h_conv1)#hidden pooling layer one #output size 14x14x32
#conv2 layer
W_conv2=weight_variable([5,5,32,64])#patch 5x5 ,in size 32,out size 64  卷积取一部分  32是图片厚度 64是长度
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#hidden conv2d layer one #output size 14x14x64
h_pool2=max_pool_2x2(h_conv2)#hidden pooling layer one #output size 7x7x64
#func1 layer
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#[n_samples,7,7,64] ->>[n_samples,7*7*64]

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#func2 layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)


#add output layer | 计算预测值(使用softmax回归) use:tf.nn.softmax(tf.matmul(xs,W)+b)  xs为输入 W为权重 b为偏差值
#prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)

#the error between prediction and real data 添加训练模型 使用loss函数:cross-entropy交叉熵
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))#loss
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#使用反向传播算法(bp算法)中的梯度下降算法(gd算法)确定最小化的成本值(使用GradientDescentOptimizer优化器)

#remember to define the same dtype and shape when restore 
W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32)
b=tf.Variable([1,2,3],dtype=tf.float32)

saver=tf.train.Saver()

sess=tf.Session()
#sess=tf.InteractiveSession()
#important step
init = tf.initialize_all_variables()
sess.run(init)

for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)#使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

save_path=saver.save(sess,"my_net/save_net.ckpt")
