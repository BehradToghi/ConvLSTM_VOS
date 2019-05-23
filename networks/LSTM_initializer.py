#######################################################################################################################
import tensorflow as tf
import os
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

class LSTM_initializer:
    '''
    Based on the VGG-16 implementation in TensorFlow:
    https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py

    Using pre-trained Numpy weight values from:
    https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
    '''
    def __init__(self):

        vgg16_path = os.path.join("models", "VGG16", "vgg16.npy")

        self.vgg16_weights = np.load(vgg16_path, encoding='latin1', allow_pickle=True).item()
        print("Reading VGG-16 weights from: ", vgg16_path)

    def build(self, mrgb):
        with tf.variable_scope("initializer_scope"):

            print("INFO: Building LSTM Initializer Network Started!")
            print("#############################", type(mrgb))
            mrgb_scaled = mrgb * 255.0

            # Convert RGB to BGR
            mask, red, green, blue = tf.split(axis=3, num_or_size_splits=4, value=mrgb_scaled)

            assert red.get_shape().as_list()[1:] == [256, 448, 1]
            assert green.get_shape().as_list()[1:] == [256, 448, 1]
            assert blue.get_shape().as_list()[1:] == [256, 448, 1]
            assert mask.get_shape().as_list()[1:] == [256, 448, 1]

            self.mbgr = tf.concat(axis=3, values=[
                mask,
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])

            self.conv1_1 = self.first_conv_layer(self.mbgr, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            self.pool5 = self.max_pool(self.conv5_3, 'pool5')

            self.h0 = self.new_conv_layer(self.pool5, 1, 512, 512, "conv_h0")
            self.c0 = self.new_conv_layer(self.pool5, 1, 512, 512, "conv_c0")

            self.vgg16_weights = None

            print("build model finished: %ds")
            print(self.mbgr, "initializer_input")
            print(self.conv1_1, "initializer_conv1_1")
            print(self.conv1_2, "initializer_conv1_2")
            print(self.pool1, "initializer_pool1")
            print(self.conv2_1, "initializer_conv2_1")
            print(self.conv2_2, "initializer_conv2_2")
            print(self.pool2, "initializer_pool2")
            print(self.conv3_1, "initializer_conv3_1")
            print(self.conv3_2, "initializer_conv3_2")
            print(self.conv3_3, "initializer_conv3_3")
            print(self.pool3, "initializer_pool3")
            print(self.conv4_1, "initializer_conv4_1")
            print(self.conv4_2, "initializer_conv4_2")
            print(self.conv4_3, "initializer_conv4_3")
            print(self.pool4, "initializer_pool4")
            print(self.conv5_1, "initializer_conv5_1")
            print(self.conv5_2, "initializer_conv5_2")
            print(self.conv5_3, "initializer_conv5_3")
            print(self.pool5, "initializer_pool5")
            print(self.h0, "initializer_h0")
            print(self.c0, "initializer_c0")

    def max_pool(self, bottom, name):
        with tf.variable_scope("initializer_" + name):
            pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        return pool

    def new_conv_layer(self, bottom, filter_size, in_channels, out_channels, name):
        # builds a new convolutional layer initialized by Xavier or Normal weights
        with tf.variable_scope("initializer_"+name):

            filt = tf.get_variable(name+"_filter", shape=[filter_size, filter_size, in_channels, out_channels],
                                   initializer=tf.contrib.layers.xavier_initializer())

            init_biases = tf.truncated_normal([out_channels], .01, .001)
            conv_biases = tf.Variable(init_biases, name=name+"_biases")

            print("filt", filt)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', name="initializer_"+name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def first_conv_layer(self, input, name):
        # Read weights for the first conv layer with a 4-channel input as mbgr
        with tf.variable_scope("initializer_"+name):

            weights_3channels = self.vgg16_weights[name][0]
            weights_channel4 = np.mean(weights_3channels, axis=2)
            weights_channel4 = np.reshape(weights_channel4, (3, 3, 1, 64))
            weights = np.concatenate((weights_channel4, weights_3channels), axis=2)

            filt = tf.Variable(weights, name=name+"_filter")
            conv_biases = tf.Variable(self.vgg16_weights[name][1], name=name+"_biases")
            print("initializer_filt", filt)

            conv = tf.nn.conv2d(input, filt, [1, 1, 1, 1], padding='SAME', name="initializer_"+name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def conv_layer(self, bottom, name):
        # Read weights for the convolutional layer and biases from the pretrained VGG16 weights
        with tf.variable_scope("initializer_"+name):

            filt = tf.Variable(self.vgg16_weights[name][0], name=name+"_filter")
            conv_biases = tf.Variable(self.vgg16_weights[name][1], name=name+"_biases")
            print ("initializer_filt", filt)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', name="initializer_"+name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

