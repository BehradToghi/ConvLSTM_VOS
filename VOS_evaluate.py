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
from tensorflow import ConfigProto

'''
NOTICE: PLEASE SET THE DATASET PATH AND OTHER HYPERPARAMETERS HERE!
'''
#
# # This is the path to the original dataset
# original_dataset_path = "/home/toghi/Toghi_WS/PyWS/data/VOS/"
#
# # Proprocess dataset and save it to a new directory as new_dataset_small
# data_generator.build_new_dataset(original_dataset_path)

# MAIN_DATASET_PATH = "../new_dataset_small/train"
# MAIN_DATASET_PATH = "../new_dataset_small/valid"
# MAIN_DATASET_PATH = "/home/toghi/Toghi_WS/PyWS/CAP6412/new_dataset_small/train"

# VGG_MEAN = [103.939, 116.779, 123.68]

from networks.Decoder import Decoder
from networks.LSTM_initializer import LSTM_initializer
from networks.Encoder import Encoder
from networks.Unrolled_convLSTM import Unrolled_convLSTM
from utils.Saver_mask import Saver_mask
from utils.VOS_data_loader import Data_loader
#######################################################################################################################
class My_network:
    # def __init__(self, batch_size, LR):
    def __init__(self, batch_size):
        self.my_graph = tf.Graph()
        self.batch_size = batch_size
        # self.lr = LR

        with self.my_graph.as_default():
            with tf.variable_scope("global_name_scope", reuse=tf.AUTO_REUSE):

                # Initialize placeholders
                self.input_images_initializer = tf.placeholder("float", [batch_size, 256, 448, 4])
                self.input_image_encoder = tf.placeholder("float", [batch_size, 7, 256, 448, 3])
                # self.groundtruth_mask = tf.placeholder("float", [batch_size, 7, 256, 448, 1])

                # Initialize operations
                self.init_model()
                # self.init_loss()
                # self.init_optimizer()

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
        input_image_encoder_unstacked = tf.reshape(self.input_image_encoder, [tmp_shape[0]*tmp_shape[1], tmp_shape[2],
                                                                              tmp_shape[3], tmp_shape[4]])

        encoder = Encoder()
        encoder.build(input_image_encoder_unstacked)
        encoder_output = encoder.conv6

        # This will be the set of B batches and F frames to be fed to ConvLSTM
        encoder_output_stacked = tf.reshape(encoder_output, [self.input_image_encoder.shape[0],
                                                             self.input_image_encoder.shape[1],
                                                             encoder_output.shape[1], encoder_output.shape[2],
                                                             encoder_output.shape[3]])


        # Feed the output of encoder to ConvLSTM
        conv_lstm = Unrolled_convLSTM()
        conv_lstm.build(encoder_output_stacked, c_0, h_0)
        lstm_output = conv_lstm.lstm_output

        # This will be fed to decoder
        lstm_output_unstacked = tf.reshape(lstm_output, (lstm_output.shape[0]*lstm_output.shape[1],
                                                         lstm_output.shape[2], lstm_output.shape[3],
                                                         lstm_output.shape[4]))


        # Feed the output of ConvLSTM to decoder
        decoder = Decoder()
        decoder.build(lstm_output_unstacked)
        decoder_output = decoder.y_hat

        mask_output = decoder.mask_out

        self.decoder_output_unstacked = tf.reshape(decoder_output, (lstm_output.shape[0], lstm_output.shape[1],
                                                                    decoder_output.shape[1], decoder_output.shape[2],
                                                                    decoder_output.shape[3]))
        self.mask_output_unstacked = tf.reshape(mask_output, self.decoder_output_unstacked.shape)
    #
    # def init_loss(self):
    #     with tf.variable_scope("loss_scope"):
    #         #Calculate the binary cross entropy loss between y and y_hat (produced and groundtruth masks)
    #         # print(self.groundtruth_mask.shape, self.decoder_output_unstacked.shape)
    #         assert self.groundtruth_mask.shape == self.decoder_output_unstacked.shape
    #         cross_ent_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.groundtruth_mask,
    #                                                             logits=self.decoder_output_unstacked,
    #                                                             name="binary_cross_entropy_loss")
    #         self.loss = tf.reduce_mean(cross_ent_loss)
    #
    # def init_optimizer(self):
    #     with tf.variable_scope("train_scope"):
    #         # Create optimizer and train it
    #         # SGD
    #         # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr, name='GradientDescent')
    #         # ADAM
    #         optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam')
    #         self.train = optimizer.minimize(self.loss)
    #
    #         # # opt = tf.train.AdamOptimizer(args.lr)
    #         # grads = self.train.compute_gradients(loss)
    #         # grads2 = [(tf.where(tf.is_nan(grad), tf.zeros(grad.shape), grad), var) for grad, var in grads]
    #         # opt_op = self.train.apply_gradients(grads2)


    def graph_builder(self):
        # return self.init, self.input_images_initializer, self.input_image_encoder, self.decoder_output_unstacked,\
        #        self.mask_output_unstacked, self.groundtruth_mask, self.loss, self.train, self.saver, self.my_graph

        return self.input_images_initializer, self.input_image_encoder, self.decoder_output_unstacked, \
               self.mask_output_unstacked, self.saver, self.my_graph



#######################################################################################################################
# Build the graph

def main(args):
    SCENARIO_NAME = args.scenario_name
    # CHECKPOINT_PATH = "./runs/Final_Run_Newton/checkpoints/model-42"
    # CHECKPOINT_PATH = "./runs/Final_Run_Newton/checkpoints_custom/model-41"
    # CHECKPOINT_PATH = "./runs/Final_Run_Newton/checkpoints_finetune_normal/model-20"
    CHECKPOINT_PATH = checkpoints_path
    batch_size = args.batch_size
    dataset_path = "../new_dataset_small/valid"
    saver_instance = Saver_mask(SCENARIO_NAME, original_dataset_path)


    # my_network = My_network(batch_size, LR=1e-5)
    my_network = My_network(batch_size)
    input_images_initializer, input_image_encoder, decoder_output, mask_output, saver, my_graph\
        = my_network.graph_builder()

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


            data_loader = Data_loader(dataset_path)
            # data_loader = data_loader.Data_loader(MAIN_DATASET_PATH)
            object_list = np.array(data_loader.get_val_object_list())
            n_batch = math.floor(len(object_list) / batch_size)+1 # +1 added for debug

            print("###DEBUG: Total object number is {:d}, n_batch is {:f}, batch size is: {:d}"
                  .format(len(object_list), n_batch, batch_size))

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


                val_out = sess.run([mask_output, decoder_output],
                                   feed_dict={input_images_initializer: input_initializer,
                                              input_image_encoder: input_encoder})
                eta_learn = time.time() - start_time
                print(" ###INFO: ETA: read:{:f}s, train:{:f}s, Progress = %{:f}, Batch #{:d}"
                      .format(eta_read, eta_learn, 100 * idx / n_batch, idx))

                # save the loss history to file
                # np.save("./out"+str(idx)+".npy", val_out)
                # np.save("./list" + str(idx) + ".npy", batch_val_object_list)

                saver_instance.store_masks(val_out, batch_val_object_list)



            print("done")

if __name__ == '__main__':
    import argparse
    from utils.config_reader import conf_reader

    # Reading configurations from the YAML file
    configs = conf_reader()
    original_dataset_path = configs["configs"]["path"] # Path to the original dataset
    checkpoints_path = configs["configs"]["checkpoints_path"] # Path to the pre-saved checkpoint files

    # Reading input arguments from command line
    parser = argparse.ArgumentParser(description='Youtube Video Object Segmentation.')
    # parser.add_argument('--n_epochs', type=int, default=1, help='Number of Epochs for Training.')
    parser.add_argument('--batch_size', type=int, default=2, help='Size of the mini-batch.')
    parser.add_argument('--scenario_name', type=str, default="validation_results", help='Validation scenario name.')
    # parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate.')
    args = parser.parse_args()

    # Proprocess dataset and save it to a new directory as new_dataset_small
    # data_generator.build_new_dataset(original_dataset_path)

    main(args)




