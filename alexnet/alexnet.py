import tensorflow as tf
import numpy as np

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

class AlexNet(object):
    
    def __init__(self, X, number_of_classes, skip_layers=[], weights_path="bvlc_alexnet.npy"):
        self.X = X
        self.weights_path = weights_path
        self.dropout_ratio = 0.4
        self.number_of_classes = number_of_classes
        self.skip_layer = skip_layers
        self.create()

    def create(self):
        '''
        Create the network graph
        '''
        conv1 = self.convolution(self.X, 11, 11, 96, 4, 4, padding="VALID", name="conv1")
        pool1 = self.max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = self.normalization(pool1, 2, 2e-05, 0.75, name="norm1")

        conv2 = self.convolution(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = self.max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = self.normalization(pool2, 2, 2e-05, 0.75, name="norm2")

        conv3 = self.convolution(norm2, 3, 3, 384, 1, 1, name='conv3')
        
        conv4 = self.convolution(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        
        conv5 = self.convolution(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='VALID',  name='pool5')

        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = self.fully_connected(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = self.dropout(fc6, self.dropout_ratio)
        
        fc7 = self.fully_connected(dropout6, 4096, 4096, name='fc7')
        dropout7 = self.dropout(fc7, self.dropout_ratio)

        self.fc8 = self.fully_connected(dropout7, 4096,
                                        self.number_of_classes, name='fc8')
        
    def load_initial_weights(self, session):
        
        weights_dict = np.load(self.weights_path, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.skip_layer:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        try:
#                        ipdb.set_trace()
                        # Biases
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases', trainable=False)
                                session.run(var.assign(data))

                                # Weights
                            else:
                                var = tf.get_variable('weights', trainable=False)
                                session.run(var.assign(data))
                        except:
                            import ipdb
                            ipdb.set_trace()




    def convolution(self, _input, filter_height, filter_width,
                    num_filters, strideX, strideY, padding="SAME", groups=1, name="error"):

        input_channels = int(_input.get_shape()[-1])

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        input_channels/groups,
                                                        num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

        convolve = lambda input_group, weight_group: tf.nn.conv2d(
                                            input_group,
                                            weight_group,
                                            strides=[1, strideX, strideY, 1],
                                            padding=padding)

        if groups == 1:
            conv = convolve(_input, weights)
        
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=_input)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=weights)
            
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            conv = tf.concat(axis=3, values=output_groups)
            
        scores = tf.reshape(tf.nn.bias_add(conv, biases),
                            conv.get_shape().as_list())

        relu = tf.nn.relu(scores, name=scope.name)
        return relu

    def fully_connected(self, _input, fc_in, fc_out, name):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights',
                                      shape=[fc_in, fc_out],
                                      trainable=True)
            biases = tf.get_variable('biases', [fc_out], trainable=True)
            
        scores = tf.nn.xw_plus_b(_input, weights, biases, name=scope.name)
#        scores = tf.matmul(_input, weights) + biases
        return tf.nn.relu(scores)
        
    def max_pool(self, _input, filter_height, filter_width, strideY,
                 strideX, padding='SAME', name="error"):
        return tf.nn.max_pool(_input, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, strideY, strideX, 1],
                              padding=padding, name=name)

    def normalization(self, _input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(_input, depth_radius=radius,
                                              alpha=alpha, beta=beta, bias=bias,
                                              name=name)

    def dropout(self, _input, keep_prob):
        return tf.nn.dropout(_input, keep_prob)
                                          
