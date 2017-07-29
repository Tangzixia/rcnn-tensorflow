import numpy as np
from datetime import datetime

train_file = "train.txt"
validation_file = "validation.txt"


train_layers = ['fc7', 'fc8']



import os
import tensorflow as tf

batch_size = 128
number_of_classes = 2
learning_rate = 0.001

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, number_of_classes])

#keep_prob = tf.placeholder(tf.float32)

from alexnet import AlexNet

model = AlexNet(x, number_of_classes, train_layers)

score = model.fc8
var_list = [v for v in tf.trainable_variables()
            if v.name.split("/")[0] in train_layers]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=score,
    labels=y))



gradients = tf.gradients(loss, var_list)
gradients = list(zip(gradients, var_list))


optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.apply_gradients(grads_and_vars=gradients)



correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




from reader import Imgdata

train_generator = Imgdata(train_file)
val_generator = Imgdata(validation_file) 


# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(len(train_generator.instances) / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(len(val_generator.instances) / batch_size).astype(np.int16)

num_epochs = 25

test_generator = Imgdata("test_competition.txt", mode="TEST")
test_batches_per_epoch = np.floor(len(test_generator.instances) / batch_size).astype(np.int16)

test_scores = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    model.load_initial_weights(sess)


    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        step = 1
        
        while step < train_batches_per_epoch:
            
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.get_batch(batch_size)
            
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys})
            step += 1

        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.get_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty})
            test_acc += acc
            test_count += 1
            
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        val_generator.reset_pointer()
        train_generator.reset_pointer()

    import ipdb
    ipdb.set_trace()
    
    


    for _ in range(test_batches_per_epoch):
        batch_tx, batch_ty = test_generator.get_batch(batch_size)
        batch_ty = one_hot_labels = np.zeros((batch_size, 2))
        scores = sess.run(score, feed_dict={x: batch_tx,
                                            y: batch_ty})
        test_scores.append(scores)
     



test_op = open('output.txt', 'w')
for item in test_scores:
  test_op.write("%s\n" % item)
