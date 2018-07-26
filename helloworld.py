import tensorflow as tf

hello = tf.constant('Hello world, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))