import tensorflow as tf
import numpy as np
from base_model import BaseModel
import cv2

import nn

## original source : https://github.com/DeepRNN/object_detection/blob/master/model.py



class ObjectDetector(BaseModel):



    def build_basic_vgg16(self):
        """ Build the basic VGG16 net. """
        print("Building the basic VGG16 net...")
        bn = self.nn.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_1_feats = nn.convolution(imgs, 3, 3, 64, 1, 1, 'conv1_1')
        conv1_1_feats = nn.batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
        conv1_2_feats = nn.convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
        conv1_2_feats = nn.batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
        pool1_feats = nn.max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

        conv2_1_feats = nn.convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
        conv2_1_feats = nn.batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
        conv2_2_feats = nn.convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
        conv2_2_feats = nn.batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
        pool2_feats = nn.max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

        conv3_1_feats = nn.convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
        conv3_1_feats = nn.batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
        conv3_2_feats = nn.convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
        conv3_2_feats = nn.batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
        conv3_3_feats = nn.convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
        conv3_3_feats = nn.batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
        pool3_feats = nn.max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

        conv4_1_feats = nn.convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
        conv4_1_feats = nn.batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
        conv4_2_feats = nn.convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
        conv4_2_feats = nn.batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
        conv4_3_feats = nn.convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
        conv4_3_feats = nn.batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
        pool4_feats = nn.max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

        conv5_1_feats = nn.convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
        conv5_1_feats = nn.batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
        conv5_2_feats = nn.convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
        conv5_2_feats = nn.batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
        conv5_3_feats = nn.convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
        conv5_3_feats = nn.batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')

        self.conv_feats = conv5_3_feats
        self.conv_feat_shape = [40, 40, 512]

        self.roi_warped_feat_shape = [16, 16, 512]
        self.roi_pooled_feat_shape = [8, 8, 512]

        self.imgs = imgs
        self.is_train = is_train
        print("Basic VGG16 net built.")
