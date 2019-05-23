import tensorflow as tf
import numpy as np

#######################################################################################################################
class Decoder:
    '''
    Up-convolutional decoder network
    '''
    def __init__(self):
        print("Decoder")

    def build(self, yy):
        with tf.variable_scope("decoder_scope"):
            print("INFO: Building Decoder Network Started!")

            self.deconv1 = self.deconv_layer(yy, 5, 512, 512, "deconv1")
            self.deconv2 = self.deconv_layer(self.deconv1, 5, 512, 256, "deconv2")
            self.deconv3 = self.deconv_layer(self.deconv2, 5, 256, 128, "deconv3")
            self.deconv4 = self.deconv_layer(self.deconv3, 5, 128, 64, "deconv4")
            self.deconv5 = self.deconv_layer(self.deconv4, 5, 64, 64, "deconv5")
            self.y_hat = self.new_conv_layer(self.deconv5, 5, 64, 1, "conv_6")
            self.mask_out = tf.nn.sigmoid(self.y_hat)


            print(self.deconv1, "decoder_deconv1")
            print(self.deconv2, "decoder_deconv2")
            print(self.deconv3, "decoder_deconv3")
            print(self.deconv4, "decoder_deconv4")
            print(self.deconv5, "decoder_deconv5")
            print(self.y_hat, "decoder_y_hat")

            print("Decoder build model finished")


    def deconv_layer(self, bottom, filter_size, in_channels, out_channels, name):
        # builds a new transpose convolution layer
        with tf.variable_scope("decoder_"+name):
            filt = tf.get_variable(name + "_filter", shape=[filter_size, filter_size, out_channels, in_channels],
                                   initializer=tf.contrib.layers.xavier_initializer())

            init_biases = tf.truncated_normal([out_channels], .01, .001)
            conv_biases = tf.Variable(init_biases, name=name+"_biases")
            print("decoder_filt", filt)

            in_shape = np.array(bottom.get_shape().as_list(), dtype=np.int32)
            output_shape = np.array([in_shape[0], 2*in_shape[1], 2*in_shape[2], out_channels], dtype=np.int32)

            deconv = tf.nn.conv2d_transpose(bottom, filt, output_shape,[1, 2, 2, 1], padding='SAME', name="decoder_"+name)
            bias = tf.nn.bias_add(deconv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def new_conv_layer(self, bottom, filter_size, in_channels, out_channels, name):
        # builds a new convolutional layer initialized by Xavier or Normal weights
        with tf.variable_scope("decoder" + name):

            filt = tf.get_variable(name + "_filter", shape=[filter_size, filter_size, in_channels, out_channels],
                                   initializer=tf.contrib.layers.xavier_initializer())

            init_biases = tf.truncated_normal([out_channels], .01, .001)
            conv_biases = tf.Variable(init_biases, name=name + "_biases")
            print("decoder_filt", filt)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', name="decoder" + name)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias
