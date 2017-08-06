import create_data
from base_model import BaseModel
from model import ObjectDetector
import tensorflow as tf

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


with tf.Session() as sess:
    coco, dataset = create_data.prepare_train_coco_data()
    model = ObjectDetector()
    sess.run(tf.global_variables_initializer())
    model.load(sess)
