'''
After training your model with "VOS.py" file, you can evaluate the performance using this file.
This script creates the objects masks based on the trained model
'''

import math
import tensorflow as tf
import numpy as np
import time
from tensorflow import ConfigProto

from utils.saver_mask import Saver_mask
from utils.vos_data_loader import Data_loader
from utils import data_generator as data_generator
from networks.decoder import Decoder
from networks.lstm_initializer import LSTM_initializer
from networks.encoder import Encoder
from networks.unrolled_convLSTM import Unrolled_convLSTM
#######################################################################################################################


class My_network:
    def __init__(self, batch_size):
        self.my_graph = tf.Graph()
        self.batch_size = batch_size

        with self.my_graph.as_default():
            with tf.variable_scope("global_name_scope", reuse=tf.AUTO_REUSE):

                # Initialize placeholders
                self.input_images_initializer = tf.placeholder("float", [batch_size, 256, 448, 4])
                self.input_image_encoder = tf.placeholder("float", [batch_size, 7, 256, 448, 3])

                # Initialize operations
                self.init_model()

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


    def graph_builder(self):
        return self.input_images_initializer, self.input_image_encoder, self.decoder_output_unstacked, \
               self.mask_output_unstacked, self.saver, self.my_graph
#######################################################################################################################
# Build the graph


def main(args):
    dataset_path = "../new_dataset_small/valid"
    scenario_name = args.scenario_name
    batch_size = args.batch_size
    saver_instance = Saver_mask(scenario_name, original_dataset_path)
    my_network = My_network(batch_size)
    input_images_initializer, input_image_encoder, decoder_output, mask_output, saver, my_graph\
        = my_network.graph_builder()

    config = ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device("/device:GPU:0"):
        print("Reset the graph")
        tf.reset_default_graph()
        with tf.Session(config=config, graph=my_graph) as sess:
            # Restore variables from disk.
            saver.restore(sess, checkpoints_path)
            print("Model restored from: ", checkpoints_path)

            data_loader = Data_loader(dataset_path)
            object_list = np.array(data_loader.get_val_object_list())
            n_batch = math.floor(len(object_list) / batch_size)+1 # +1 added for debug

            print("###DEBUG: Total object number is {:d}, n_batch is {:f}, batch size is: {:d}"
                  .format(len(object_list), n_batch, batch_size))

            for idx in range(n_batch):
                start_time = time.time()

                start_i = idx*batch_size
                end_i = (idx+1)*batch_size
                if end_i <= len(object_list):
                    batch_val_object_list = object_list[start_i:end_i]
                else:
                    batch_val_object_list = object_list[len(object_list)-batch_size:len(object_list)]
                assert len(batch_val_object_list) == batch_size

                input_initializer, input_encoder = data_loader.get_val_data_batch(batch_val_object_list)

                eta_read = time.time() - start_time
                start_time = time.time()

                val_out = sess.run([mask_output, decoder_output],
                                   feed_dict={input_images_initializer: input_initializer,
                                              input_image_encoder: input_encoder})
                eta_learn = time.time() - start_time
                print(" ###INFO: ETA: read:{:f}s, train:{:f}s, Progress = %{:f}, Batch #{:d}"
                      .format(eta_read, eta_learn, 100 * idx / n_batch, idx))

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
    parser.add_argument('--batch_size', type=int, default=2, help='Size of the mini-batch.')
    parser.add_argument('--scenario_name', type=str, default="validation_results", help='Validation scenario name.')
    args = parser.parse_args()

    # Proprocess dataset and save it to a new directory as new_dataset_small
    data_generator.build_new_valid_dataset(original_dataset_path)

    # Run the evaluation to creat output masks.
    main(args)
