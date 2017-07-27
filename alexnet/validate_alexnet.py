import tensorflow as tf
from alexnet import AlexNet

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

## initialize the weight and bias variables in tensorflow
## the layers except for fc7 and fc8 will be replaced with actual variables
train_layers = ['fc7', 'fc8']
var_list = [v for v in tf.trainable_variables()
            if v.name.split("/")[0] in train_layers]


model = AlexNet(x, 1000)

score = model.fc8

# batch_size = 128
# number_of_classes = 2
# learning_rate = 0.001

# x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
# y = tf.placeholder(tf.float32, [None, number_of_classes])

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=score,
#     labels=y))



# gradients = tf.gradients(loss, var_list)
# gradients = list(zip(gradients, var_list))


# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    model.load_initial_weights(sess)
    
