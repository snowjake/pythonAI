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


#define placeholder for input to network  | 构造输入输出数据结构
xs=tf.placeholder(tf.float32,[None,784],name='x_input')#28*28
ys=tf.placeholder(tf.float32,[None,10],name='y_input')



#add output layer | 计算预测值(使用softmax回归) use:tf.nn.softmax(tf.matmul(xs,W)+b)  xs为输入 W为权重 b为偏差值
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)

#the error between prediction and real data 添加训练模型 使用loss函数:cross-entropy交叉熵
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))#loss
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#使用反向传播算法(bp算法)中的梯度下降算法(gd算法)确定最小化的成本值(使用GradientDescentOptimizer优化器)

sess=tf.Session()
#important step
init = tf.initialize_all_variables()
sess.run(init)

for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)#使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
        
