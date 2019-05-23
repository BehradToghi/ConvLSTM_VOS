#######################################################################################################################
import tensorflow as tf
import os
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

class Encoder:
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

    def build(self, rgb):
        with tf.variable_scope("encoder_scope"):


            print("INFO: Building Encoder Network Started!")
            rgb_scaled = rgb * 255.0

            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

            assert red.get_shape().as_list()[1:] == [256, 448, 1]
            assert green.get_shape().as_list()[1:] == [256, 448, 1]
            assert blue.get_shape().as_list()[1:] == [256, 448, 1]

            self.bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],

            ])

            self.conv1_1 = self.conv_layer(self.bgr, "conv1_1")
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
            self.conv6 = self.new_conv_layer(self.pool5, 1, 512, 512, "conv_6")


            self.vgg16_weights = None

            print(self.bgr, "encoder_input")
            print(self.conv1_1, "encoder_conv1_1")
            print(self.conv1_2, "encoder_conv1_2")
            print(self.pool1, "encoder_pool1")
            print(self.conv2_1, "encoder_conv2_1")
            print(self.conv2_2, "encoder_conv2_2")
            print(self.pool2, "encoder_pool2")
            print(self.conv3_1, "encoder_conv3_1")
            print(self.conv3_2, "encoder_conv3_2")
            print(self.conv3_3, "encoder_conv3_3")
            print(self.pool3, "encoder_pool3")
            print(self.conv4_1, "encoder_conv4_1")
            print(self.conv4_2, "encoder_conv4_2")
            print(self.conv4_3, "encoder_conv4_3")
            print(self.pool4, "encoder_pool4")
            print(self.conv5_1, "encoder_conv5_1")
            print(self.conv5_2, "encoder_conv5_2")
            print(self.conv5_3, "encoder_conv5_3")
            print(self.pool5, "encoder_pool5")
            print(self.conv6, "encoder_conv6")
            # print(self.h0, "h0")
            # print(self.c0, "c0")
            print("Encoder build model finished")

    def max_pool(self, bottom, name):
        with tf.variable_scope("encoder_"+name):
            pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        return pool

    def new_conv_layer(self, bottom, filter_size, in_channels, out_channels, name):
        # builds a new convolutional layer initialized by Xavier or Normal weights
        with tf.variable_scope("encoder_"+name):
            #
            # init_weights = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
            # filt = tf.Variable(init_weights, name=name+"_filter")

            filt = tf.get_variable(name + "_filter", shape=[filter_size, filter_size, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())

            init_biases = tf.truncated_normal([out_channels], .01, .001)
            conv_biases = tf.Variable(init_biases, name=name+"_biases")
            print("encoder_filt", filt)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', name="encoder_"+name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def conv_layer(self, bottom, name):
        # Read weights for the convolutional layer and biases from the pretrained VGG16 weights
        with tf.variable_scope("encoder_"+name):

            filt = tf.Variable(self.vgg16_weights[name][0], name=name+"_filter")
            conv_biases = tf.Variable(self.vgg16_weights[name][1], name=name+"_biases")
            print ("encoder_filt", filt)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', name="encoder_"+name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu
