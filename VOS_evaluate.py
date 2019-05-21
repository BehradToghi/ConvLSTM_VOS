'''
Sequence-to-sequence Video Object Segmentation
Using YouTube-VOS dataset:

https://arxiv.org/abs/1809.00461

Author: Behrad Toghi
'''

import utils.VOS_data_loader2 as data_loader
from utils import rnn as rnn_net
from utils import rnn_cell as rnn_cell
from utils.rnn_cell_carlthome import ConvLSTMCell
# from utils import data_loader as data_generator
# from utils import mask_saver
import math
import os
import tensorflow as tf
# from tensorflow import keras
import numpy as np
import time
import cv2

from PIL import Image
from PIL import ImageChops
import shutil


'''
NOTICE: PLEASE SET THE DATASET PATH AND OTHER HYPERPARAMETERS HERE!
'''
#
# # This is the path to the original dataset
original_dataset_path = "/home/toghi/Toghi_WS/PyWS/data/VOS/"
#
# # Proprocess dataset and save it to a new directory as new_dataset_small
# data_generator.build_new_dataset(original_dataset_path)

# MAIN_DATASET_PATH = "../new_dataset_small/train"
MAIN_DATASET_PATH = "../new_dataset_small/valid"
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

            init_biases = tf.truncated_normal([out_channels], .0, .001)
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

        self.vgg16_weights = np.load(vgg16_path, encoding='latin1').item()
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

        self.vgg16_weights = np.load(vgg16_path, encoding='latin1').item()
        print("Reading VGG-16 weights from: ", vgg16_path)

    def build(self, mrgb):
        with tf.variable_scope("initializer_scope"):

            start_time = time.time()
            print("INFO: Building LSTM Initializer Network Started!")
            # print("#############################", type(mrgb))
            mrgb_scaled = mrgb * 255.0

            # Convert RGB to BGR
            mask, red, green, blue = tf.split(axis=3, num_or_size_splits=4, value=mrgb_scaled)


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



            print("initializer_filt", filt)

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

                # self.input_images_initializer = tf.placeholder("float", [batch_size, 768, 1344, 4])
                # self.input_image_encoder = tf.placeholder("float", [batch_size, 7, 768, 1344, 3])
                # self.groundtruth_mask = tf.placeholder("float", [batch_size, 7, 768, 1344, 1])

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

        # decoder_output = tf.reshape(decoder_output, (28, 720, 1280, 1))

        mask_output = decoder.mask_out

        self.decoder_output_unstacked = tf.reshape(decoder_output, (lstm_output.shape[0], lstm_output.shape[1], decoder_output.shape[1], decoder_output.shape[2], decoder_output.shape[3]))
        self.mask_output_unstacked = tf.reshape(mask_output, self.decoder_output_unstacked.shape)

    def init_loss(self):
        with tf.variable_scope("loss_scope"):
            #Calculate the binary cross entropy loss between y and y_hat (produced and groundtruth masks)
            # print(self.groundtruth_mask.shape, self.decoder_output_unstacked.shape)
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

class Saver_mask:
    def __init__(self, name):

        print("Mask saver initialized")
        self.scenario_name = name
        # self.color_array = [[103, 95, 236], [87, 145, 249], [99, 200, 250], [148, 199, 153], [178, 179, 98]]

    def store_masks(self, output, info):
        self.batch_size = len(info)

        for batch in range(self.batch_size):
            object_info = info[batch]

            self.video_ID = object_info[0]
            self.object_ID = object_info[1]
            frames_list = object_info[2]
            for jj in range(len(frames_list)):
                self.frame = frames_list[jj]

                print(self.video_ID, self.object_ID, self.frame)
                source_path, existsMask = self.check_first_annotation()

                # Check if this is the initial annotation which alread exist in the dataset
                if existsMask:
                    print("###DEBUG: Annotation already exists, copying from: ", source_path)
                    save_path = os.path.join("..", self.scenario_name, self.video_ID)  # ,
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    target_path = os.path.join(save_path, self.frame[:-4] + ".png")
                    shutil.copy(source_path, target_path)

                else:
                    mask = output[0][batch][jj]
                    # Threshold
                    mask = np.array(255 * mask, dtype=np.uint8)
                    mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
                    ret, mask_thresh = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)
                    # Read color palette
                    self.test_image = Image.open("./sample.png")
                    # Apply color palette
                    temp_mask = np.array(int(self.object_ID) * mask_thresh, dtype=np.int8)

                    final_mask = Image.fromarray(temp_mask, mode='P')
                    final_mask.putpalette(self.test_image.getpalette())

                    self.save_image(final_mask)

    def save_image(self, img):
        save_path = os.path.join("..", self.scenario_name, self.video_ID) #,
        isMask, mask_new_shape = self.get_mask_shape()

        if isMask:
            # Create drectory
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, self.frame[:-4]+".png")

            # resize mask
            final_mask_large = img.resize((mask_new_shape[1], mask_new_shape[0]))
            # Check if the annotation already exists
            if not os.path.isfile(file_path):
                final_mask_large.save(file_path)

            else:
                # print("DEBUG: file already exists")
                old_mask_read = Image.open(file_path)
                old_mask = old_mask_read.copy()

                previous_mask = np.array(old_mask, dtype=np.int8)
                current_mask = np.array(final_mask_large, dtype=np.int8)

                diff = previous_mask - 10*current_mask
                diff[diff<0] = 0
                new_mask_array = diff + current_mask

                combined_mask = Image.fromarray(new_mask_array, mode='P')
                combined_mask.putpalette(self.test_image.getpalette())
                combined_mask.save(file_path)



    def check_first_annotation(self):
        annotation_path = os.path.join(original_dataset_path, "valid", "Annotations", self.video_ID, self.frame[:-4]+".png")
        return annotation_path, os.path.isfile(annotation_path)


    def get_mask_shape(self):

        sample_mask_path = os.path.join(original_dataset_path, "valid_submit_sample", "Annotations", self.video_ID, self.frame[:-4]+".png")
        flag = os.path.isfile(sample_mask_path)
        if flag:
            sample_mask = cv2.imread(sample_mask_path)
            size = sample_mask.shape
            # if sample_mask.shape != (720, 1280,3):
            #     print("###### ERROR: SAMPLE MASK SIZE IS NOT 720x1280", sample_mask_path)
        else:
            print("#Debug: file does not exist: ", sample_mask_path)
            size = (720, 1280, 3)
        return flag, size

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# Build the graph
SCENARIO_NAME = "Results_final_fine"
# CHECKPOINT_PATH = "./runs/Final_Run_Newton/checkpoints/model-42"
# CHECKPOINT_PATH = "./runs/Final_Run_Newton/checkpoints_custom/model-41"
CHECKPOINT_PATH = "./runs/Final_Run_Newton/checkpoints_finetune_normal/model-20"


#######################################################################################################################
batch_size = 4
saver_instance = Saver_mask(SCENARIO_NAME)

my_network = My_network(batch_size, LR=1e-5)
init, input_images_initializer, input_image_encoder, decoder_output, mask_output, groundtruth_mask, loss, train, saver, my_graph = my_network.graph_builder()
#
from tensorflow import ConfigProto
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)

# config = ConfigProto(gpu_options=gpu_options)
# config.gpu_options.allow_growth = True

config = ConfigProto()
config.gpu_options.allow_growth = True

with tf.device("/device:GPU:0"):

    print("Reset the graph")
    tf.reset_default_graph()

    with tf.Session(config=config, graph=my_graph) as sess:
        # # Restore variables from disk.
        saver.restore(sess, CHECKPOINT_PATH)
        print("Model restored from: ", CHECKPOINT_PATH)



        data_loader = data_loader.Data_loader(MAIN_DATASET_PATH)
        object_list = np.array(data_loader.get_val_object_list())
        n_batch = math.floor(len(object_list) / batch_size)+1 # +1 added for debug

        print("###DEBUG: Total object number is {:d}, n_batch is {:f}, batch size is: {:d}".format(len(object_list), n_batch, batch_size))

        # n_batch = 1

        for idx in range(n_batch):
            start_time = time.time()


            start_i = idx*batch_size
            end_i = (idx+1)*batch_size
            if end_i <= len(object_list):
                # print(idx * batch_size, (idx + 1) * batch_size)
                batch_val_object_list = object_list[start_i:end_i]
            else:
                # print(len(object_list)-4, len(object_list))
                batch_val_object_list = object_list[len(object_list)-4:len(object_list)]
            assert len(batch_val_object_list) == batch_size

            # print("\n ***DEBUG:", batch_val_object_list)

            # input_initializer, input_encoder = data_loader.get_val_data_batch(batch_val_object_list)
            input_initializer, input_encoder = data_loader.get_val_data_batch(batch_val_object_list)

            eta_read = time.time() - start_time
            start_time = time.time()


            val_out = sess.run([mask_output, decoder_output], feed_dict={input_images_initializer: input_initializer, input_image_encoder: input_encoder})
            eta_learn = time.time() - start_time
            print(" ###INFO: ETA: read:{:f}s, train:{:f}s, Progress = %{:f}, Batch #{:d}".format(eta_read, eta_learn, 100 * idx / n_batch, idx))

            # save the loss history to file
            # np.save("./out"+str(idx)+".npy", val_out)
            # np.save("./list" + str(idx) + ".npy", batch_val_object_list)

            saver_instance.store_masks(val_out, batch_val_object_list)



        print("done")





