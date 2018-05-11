import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))