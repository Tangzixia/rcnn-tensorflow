import tensorflow as tf
from alexnet import AlexNet
from caffe_classes import class_names

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


## initialize the weight and bias variables in tensorflow
## the layers except for fc7 and fc8 will be replaced with actual variables
#train_layers = ['fc7', 'fc8']
train_layers = []
var_list = [v for v in tf.trainable_variables()
            if v.name.split("/")[0] in train_layers]


batch_size = 1
number_of_classes = 1000

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, number_of_classes])

model = AlexNet(x, 1000)
score = model.fc8

probs = tf.nn.softmax(score)


import os
import cv2
image_folder = "validation-images/"

image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]

images = [cv2.imread(f) for f in image_files]


import numpy as np
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)

    for idx, image in enumerate(images):
        image = cv2.resize(image.astype(np.float32), (227,227))
        image -= imagenet_mean
        image = image.reshape((1, 227, 227, 3))
        
        probabilities = sess.run(probs, feed_dict={x: image})
        class_name = class_names[np.argmax(probabilities)]
        print(class_name)
