'''
Sequence-to-sequence Video Object Segmentation
Using YouTube-VOS dataset:

https://arxiv.org/abs/1809.00461

Author: Behrad Toghi
'''

import utils.VOS_data_loader as data_loader
from utils import rnn as rnn_net
from utils import rnn_cell as rnn_cell
from utils.rnn_cell_carlthome import ConvLSTMCell
# from utils import data_loader as data_generator

import math
import os
import tensorflow as tf
# from tensorflow import keras
import numpy as np
import time
import cv2

'''
NOTICE: PLEASE SET THE DATASET PATH AND OTHER HYPERPARAMETERS HERE!
'''
#
# # This is the path to the original dataset
# original_dataset_path = "/home/toghi/Toghi_WS/PyWS/data/VOS/"
#
# # Proprocess dataset and save it to a new directory as new_dataset_small
# data_generator.build_new_dataset(original_dataset_path)
# data_generator.build_new_valid_dataset(original_dataset_path)

MAIN_DATASET_PATH = "../new_dataset_small/train"
# MAIN_DATASET_PATH = "/home/toghi/Toghi_WS/PyWS/CAP6412/new_dataset_small/train"

VGG_MEAN = [103.939, 116.779, 123.68]


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
            start_time = time.time()

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

            print(("Decoder build model finished: %ds" % (time.time() - start_time)))


    def deconv_layer(self, bottom, filter_size, in_channels, out_channels, name):
        # builds a new transpose convolution layer
        with tf.variable_scope("decoder_"+name):
            #
            # init_weights = tf.truncated_normal([filter_size, filter_size, out_channels, in_channels], 0.0, 0.001)
            # filt = tf.Variable(init_weights, name=name+"_filter")

            filt = tf.get_variable(name + "_filter", shape=[filter_size, filter_size, out_channels, in_channels], initializer=tf.contrib.layers.xavier_initializer())

            init_biases = tf.truncated_normal([out_channels], .01, .001)
            conv_biases = tf.Variable(init_biases, name=name+"_biases")
            print("decoder_filt", filt)

            in_shape = np.array(bottom.get_shape().as_list(), dtype=np.int32)
            # print(name, "***in_shape",  in_shape)
            output_shape = np.array([in_shape[0], 2*in_shape[1], 2*in_shape[2], out_channels], dtype=np.int32)
            # print(name, "***output_shape",  output_shape)

            deconv = tf.nn.conv2d_transpose(bottom, filt, output_shape,[1, 2, 2, 1], padding='SAME', name="decoder_"+name)
            bias = tf.nn.bias_add(deconv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def new_conv_layer(self, bottom, filter_size, in_channels, out_channels, name):
        # builds a new convolutional layer initialized by Xavier or Normal weights
        with tf.variable_scope("decoder" + name):
            # init_weights = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
            # filt = tf.Variable(init_weights, name=name + "_filter")

            filt = tf.get_variable(name + "_filter", shape=[filter_size, filter_size, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())

            init_biases = tf.truncated_normal([out_channels], .01, .001)
            conv_biases = tf.Variable(init_biases, name=name + "_biases")
            print("decoder_filt", filt)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', name="decoder" + name)
            bias = tf.nn.bias_add(conv, conv_biases)
            # relu = tf.nn.relu(bias)

            return bias

#######################################################################################################################
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

            start_time = time.time()
            print("INFO: Building Encoder Network Started!")
            rgb_scaled = rgb * 255.0

            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            self.bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
            # print(self.bgr.shape)
            # self.conv1_1 = self.first_conv_layer(self.mbgr, "encoder_conv1_1")

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

            # self.h0 = self.new_conv_layer(self.pool5, 3, 512, 512, "conv_h0")
            # self.c0 = self.new_conv_layer(self.pool5, 3, 512, 512, "conv_c0")


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
            print(("Encoder build model finished: %ds" % (time.time() - start_time)))

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

#######################################################################################################################
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

            start_time = time.time()
            print("INFO: Building LSTM Initializer Network Started!")
            print("#############################", type(mrgb))
            mrgb_scaled = mrgb * 255.0

            # Convert RGB to BGR
            mask, red, green, blue = tf.split(axis=3, num_or_size_splits=4, value=mrgb_scaled)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            # assert mask.get_shape().as_list()[1:] == [224, 224, 1]

            # assert red.get_shape().as_list()[1:] == [256, 448, 1]
            # assert green.get_shape().as_list()[1:] == [256, 448, 1]
            # assert blue.get_shape().as_list()[1:] == [256, 448, 1]
            # assert mask.get_shape().as_list()[1:] == [256, 448, 1]

            self.mbgr = tf.concat(axis=3, values=[
                mask,
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])

            # assert mbgr.get_shape().as_list()[1:] == [224, 224, 4]
            #
            # self.conv1_1 = self.conv_layer(bgr, "conv1_1")
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

            print(("build model finished: %ds" % (time.time() - start_time)))
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

            # init_weights = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)

            # filt = tf.Variable(init_weights, name=name+"_filter")

            filt = tf.get_variable(name+"_filter", shape=[filter_size, filter_size, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer())

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

#######################################################################################################################
class Unrolled_convLSTM:
    def __init__(self):
        '''
        This is a Conv LSTM RNN based on:

        Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
        https://arxiv.org/abs/1506.04214
        '''

        # print("LSTM RNN")

    def build(self, input_frames, c_0, h_0):
        with tf.variable_scope("convlstm_scope"):

            print("INFO: Building LSTM Network!")
            # Create an LSTM tuple for the initial state h0 and c0
            init_state = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)

            shape = [8, 14]
            filters = 512
            kernel = [5, 5]


            cell = rnn_cell.ConvLSTMCell(2, (8, 14, 512), 512, (5, 5), name="conv_lstm_cell")
            # cell = ConvLSTMCell(shape, filters, kernel)

            self.lstm_output, state = rnn_net.dynamic_rnn(cell, input_frames, initial_state=init_state, dtype=input_frames.dtype)
            # self.lstm_output, state = rnn_net.dynamic_rnn(cell, input_frames, dtype=input_frames.dtype)

            #
            # cell = rnn_cell.ConvLSTMCell(2, (8, 14, 512), 512, (3, 3), initializers=None, name="conv_lstm_cell")
            # output, out_state = rnn.dynamic_rnn(cell, input_frames, sequence_length=7, initial_state=init_state)




#######################################################################################################################
class My_network:
    def __init__(self, batch_size, LR):
        self.my_graph = tf.Graph()
        self.batch_size = batch_size
        self.lr = LR

        with self.my_graph.as_default():
            with tf.variable_scope("global_name_scope", reuse=tf.AUTO_REUSE):

                # Initialize placeholders
                self.input_images_initializer = tf.placeholder("float", [batch_size, 256, 448, 4])
                self.input_image_encoder = tf.placeholder("float", [batch_size, 7, 256, 448, 3])
                self.groundtruth_mask = tf.placeholder("float", [batch_size, 7, 256, 448, 1])

                # Initialize operations
                self.init_model()
                self.init_loss()
                self.init_optimizer()

                # Initialize the graph
                self.init = tf.global_variables_initializer()

                self.saver = tf.train.Saver() #(max_to_keep=0)


    def init_model(self):

        # Get Initial states for LSTM from the first image and its masked
        lstm_initializer = LSTM_initializer()
        lstm_initializer.build(self.input_images_initializer)
        h_0 = lstm_initializer.h0
        c_0 = lstm_initializer.c0

        # Get 7 frames and feed them to Encoder
        tmp_shape = self.input_image_encoder.shape
        input_image_encoder_unstacked = tf.reshape(self.input_image_encoder, [tmp_shape[0]*tmp_shape[1], tmp_shape[2], tmp_shape[3], tmp_shape[4]])

        encoder = Encoder()
        encoder.build(input_image_encoder_unstacked)
        encoder_output = encoder.conv6

        # This will be the set of B batches and F frames to be fed to ConvLSTM
        encoder_output_stacked = tf.reshape(encoder_output, [self.input_image_encoder.shape[0], self.input_image_encoder.shape[1], encoder_output.shape[1], encoder_output.shape[2], encoder_output.shape[3]])


        # Feed the output of encoder to ConvLSTM
        conv_lstm = Unrolled_convLSTM()
        conv_lstm.build(encoder_output_stacked, c_0, h_0)
        lstm_output = conv_lstm.lstm_output

        # This will be fed to decoder
        lstm_output_unstacked = tf.reshape(lstm_output, (lstm_output.shape[0]*lstm_output.shape[1], lstm_output.shape[2], lstm_output.shape[3], lstm_output.shape[4]))


        # Feed the output of ConvLSTM to decoder
        decoder = Decoder()
        decoder.build(lstm_output_unstacked)
        decoder_output = decoder.y_hat
        mask_output = decoder.mask_out

        self.decoder_output_unstacked = tf.reshape(decoder_output, (lstm_output.shape[0], lstm_output.shape[1], decoder_output.shape[1], decoder_output.shape[2], decoder_output.shape[3]))
        self.mask_output_unstacked = tf.reshape(mask_output, self.decoder_output_unstacked.shape)

    def init_loss(self):
        with tf.variable_scope("loss_scope"):
            #Calculate the binary cross entropy loss between y and y_hat (produced and groundtruth masks)
            assert self.groundtruth_mask.shape == self.decoder_output_unstacked.shape
            cross_ent_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.groundtruth_mask,
                                                                logits=self.decoder_output_unstacked,
                                                                name="binary_cross_entropy_loss")
            self.loss = tf.reduce_mean(cross_ent_loss)

    def init_optimizer(self):
        with tf.variable_scope("train_scope"):
            # Create optimizer and train it
            # SGD
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr, name='GradientDescent')
            # ADAM
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam')
            self.train = optimizer.minimize(self.loss)

            # # opt = tf.train.AdamOptimizer(args.lr)
            # grads = self.train.compute_gradients(loss)
            # grads2 = [(tf.where(tf.is_nan(grad), tf.zeros(grad.shape), grad), var) for grad, var in grads]
            # opt_op = self.train.apply_gradients(grads2)


    def graph_builder(self):
        return self.init, self.input_images_initializer, self.input_image_encoder, self.decoder_output_unstacked,\
               self.mask_output_unstacked, self.groundtruth_mask, self.loss, self.train, self.saver, self.my_graph



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# Build the graph
# SCENARIO_NAME = "out_masks_normal_13"
CHECKPOINT_PATH = "./runs/13epoch_loss/checkpoints_custom/model-13"
#######################################################################################################################

batch_size = 4

my_network = My_network(batch_size, LR=1e-6)
init, input_images_initializer, input_image_encoder, decoder_output, mask_output, groundtruth_mask, loss, train, saver, my_graph = my_network.graph_builder()
#
from tensorflow import ConfigProto
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.60)

# config = ConfigProto(gpu_options=gpu_options)
config = ConfigProto()
config.gpu_options.allow_growth = True

with tf.device("/device:GPU:0"):

    print("Reset the graph")
    tf.reset_default_graph()

    with tf.Session(config=config, graph=my_graph) as sess:

        # Initialize the graph
        # sess.run(init)
        # Save the graph
        # writer = tf.summary.FileWriter('./graphs', graph=my_graph)

        # # Restore variables from disk.
        saver.restore(sess, CHECKPOINT_PATH)
        print("Model restored from: ", CHECKPOINT_PATH)


        data_loader = data_loader.Data_loader(MAIN_DATASET_PATH)
        object_list = np.array(data_loader.get_object_list())
        n_batch = math.floor(len(object_list) / batch_size)
        ################
        # n_batch = 4
        ################
        print("INFO: Batch Size is {:d}, total number of batches: {:d} ".format(batch_size, n_batch))
        EPOCHS = 80

        loss_history_array = np.zeros([EPOCHS*n_batch])


        for epoch in range(EPOCHS):
            # print(object_list.shape)

            # shuffle the dataset at each epoch
            shuffled_object_list = object_list.copy()
            np.random.shuffle(shuffled_object_list)

            # print(shuffled_object_list.shape)
            # shuffled_object_list = object_list
            for idx in range(n_batch):
            # for idx in range(1):
                start_time = time.time()


                batch_object_list = np.array(shuffled_object_list[idx * batch_size:(idx + 1) * batch_size])
                assert len(batch_object_list) == batch_size

                # print(("1: %f s" % (time.time() - start_time)))
                # start_time = time.time()

                print("\n ***DEBUG:", batch_object_list)
                input_initializer, input_encoder, groundtruth = data_loader.get_data_batch(batch_object_list)
                eta_read = time.time() - start_time
                start_time = time.time()

                # print(("2: %f s" % (time.time() - start_time)))
                # start_time = time.time()

                # print("input_initializer.shape", input_initializer.shape)
                # print("input_encoder.shape", input_encoder.shape)
                # print("groundtruth.shape", groundtruth.shape)



                # print("running session")
                history = sess.run([train, loss], feed_dict={input_images_initializer: input_initializer, input_image_encoder: input_encoder, groundtruth_mask: groundtruth})

                # print(("3: %f s" % (time.time() - start_time)))

                eta_learn = time.time() - start_time


                print(" ###INFO: ETA: read:{:f}s, train:{:f}s, Epoch {:d}/{:d} Progress = %{:f}, Batch #{:d} ===> Loss = ".format(eta_read, eta_learn, epoch, EPOCHS, 100*idx/n_batch, idx), history[1])

                # Keep the track of loss
                loss_history_array[epoch*n_batch+idx] = history[1]


                # # DEBUG VISUALIZATION
                # for b in range(len(input_encoder)):
                #     for f in range(len(input_encoder[b])):
                #         image = input_encoder[b][f]
                #         mask = groundtruth[b][f]
                #
                #
                #         cv2.imshow('img', image)
                #         cv2.waitKey(0)
                #         cv2.destroyAllWindows()
                #
                #         cv2.imshow('msk', mask)
                #         cv2.waitKey(0)
                #         cv2.destroyAllWindows()

            # save the loss history to file
            np.save("./loss_history_finetune_normal.npy", loss_history_array)


            save_path = saver.save(sess, "./checkpoints_finetune_normal/model", global_step=epoch)
            # print("End of epoch")
            print("Model saved in path: %s" % save_path)


    print("\n ***DEBUG: TRAINING DONE!")
    # print("Reset the graph")
    # tf.reset_default_graph()
    #
    # with tf.Session(config=config, graph=my_graph) as sess:
    #     # Restore variables from disk.
    #     saver.restore(sess, "./checkpoints/model-79")
    #     print("Model restored.")
    #
    #     idx = 2
    #     batch_object_list = np.array(object_list[idx * batch_size:(idx + 1) * batch_size])
    #     print(batch_object_list)
    #
    #     input_initializer, input_encoder, groundtruth = data_loader.get_data_batch(batch_object_list)
    #     print("===>INFO: Showing sample results")
    #     val_out = sess.run([mask_output, decoder_output], feed_dict={input_images_initializer: input_initializer, input_image_encoder: input_encoder, groundtruth_mask: groundtruth})
    #
    #
    #
    #     # assert len(test_out[0]) == len(test_out[1])
    #     np.save("./validation_output.npy", val_out)
    #     #
    #     # for batch in range(len(test_out[0])):
    #     #     for frame in range(len(test_out[0][batch])):
    #     #         image = test_out[batch][frame]
    #     #         print("image.shape", image.shape)
    #     #         cv2.imshow("img", image)
    #     #         cv2.waitKey(0)
    #     #         cv2.destroyAllWindows()
    #     #
    #
    #







            #
            # print("decoder_output.shape", decoder_output.shape)
            #
            # batch1 = history[2][0]
            #
            # for nn in range(len(batch1)):
            #     image = batch1[nn]
            #     print("image.shape", image.shape)
            #     cv2.imshow("img", image)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()



            # print("input_initializer.shape", input_initializer.shape)
            # print("input_encoder.shape", input_encoder.shape)
            # print("groundtruth.shape", groundtruth.shape)
            #
            # image = input_encoder[0][1]
            # cv2.imshow("imgbatch0", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #
            # image = input_encoder[1][1]
            # cv2.imshow("imgbatch1", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            # decoder_out = sess.run(decoder_output, feed_dict={input_image_encoder: batch_7})
            #
            # save_path = saver.save(sess, "./checkpoints/model.ckpt")
            # print("Model saved in path: %s" % save_path)





        #
        # print(decoder_out.shape)
        # print(type(decoder_out[0]))
        # print(np.ndarray.max(decoder_out[0]))
        # print(np.ndarray.min(decoder_out[0]))
        #
        # cv2.imshow("out", decoder_out[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        #
        #
        # encoder_out = sess.run(encoder_output, feed_dict={input_image_encoder: batch_7})
        #
        # print(encoder_output)
        # print(type(encoder_output))
        # print(encoder_output.shape)
#
#
#
# with tf.device("/device:GPU:0"):
#
#     with tf.Session() as sess:
#         print("Session Started.")
#
#
#
#
#
#
#
#
#
#         #
#         # my_network = my_net()
#         # input_tensor = tf.placeholder("float", [batch_size, 256, 448, 3])
#         # output = my_network.build()
#
#         # sess.run(my_network.input_decoder_batch, feed_dict={tf.convert_to_tensor(batch_1)})
#
#
#
#
#
#         #
#         # input_initializer_batch = tf.placeholder("float", [batch_size, 256, 448, 4])
#         # input_encoder_batch = tf.placeholder("float", [batch_size, 256, 448, 3])
#         # input_decoder_batch = tf.placeholder("float", [batch_size, 8, 14, 512])
#         #
#         #
#         # lstm_initializer = LSTM_initializer()
#         # with tf.name_scope("initializer_vgg"):
#         #     lstm_initializer.build(input_initializer_batch)
#         #
#         # encoder = Encoder()
#         # with tf.name_scope("encoder_vgg"):
#         #     encoder.build(input_encoder_batch)
#         #
#         # decoder = Decoder()
#         # with tf.name_scope("decoder_vgg"):
#         #     decoder.build(input_decoder_batch)
#         #
#         #     # decoder.build(encoder.conv6)
#         #
#         #
#         #
#         # init = sess.run(tf.global_variables_initializer())
#         # print("************ Session /device:GPU:0 ************")
#         # #
#         # # h0 = sess.run(lstm_initializer.h0, feed_dict={input_initializer_batch: batch_0})
#         # # c0 = sess.run(lstm_initializer.c0, feed_dict={input_initializer_batch: batch_0})
#         # #
#         # # print(type(batch_1))
#         # xx1 = sess.run(encoder.conv6, feed_dict={input_encoder_batch: batch_1})
#         # print(xx1.shape)
#         #
#         # y1 = sess.run(decoder.y, feed_dict={input_decoder_batch: xx1})
#         # print(y1.shape)
#         #
#         # #
#         # # y1 = sess.run(decoder.y, feed_dict={input_decoder_batch: encoder.conv6})
#         # # print(y1.shape)
#         #
#         # #
#         # #
#         # # print("h0.shape", h0.shape)
#         # # print("c0.shape", c0.shape)
#         # # print("xx1.shape", xx1.shape)
#         # #
#         # # #
#         # # #
#         # # # print(sess.run(lstm_initializer.pool5, feed_dict={input_batch: batch}).shape)
#         # # # print(sess.run(lstm_initializer.h0, feed_dict={input_batch: batch}).shape)
#         # # # print(sess.run(lstm_initializer.c0, feed_dict={input_batch: batch}).shape)
#         # # #
#         # # # print(sess.run(encoder.conv6, feed_dict={input_batch: batch}).shape)
#         # # #
#         # #
#         # # #
#         # # # # define loss and optimizer here!
#         # # #
#         # # # cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
#         # # # train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
#         # # # sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})
#         # #
#         # # # prob = sess.run(vgg.prob, feed_dict={images: batch})
#         # #
#         # #
#         # #
