'''
This scripts runs the training phase. You can indicate the configurable variable (Learning rate, number of epochs and
batch size as terminal inputs. Please refer to Readme.md for more information.

Sequence-to-sequence Video Object Segmentation
Using YouTube-VOS dataset:

https://arxiv.org/abs/1809.00461

Author: Behrad Toghi
'''


import math
import tensorflow as tf
import numpy as np
import time
from tensorflow import ConfigProto
import argparse

from networks.decoder import Decoder
from networks.lstm_initializer import LSTM_initializer
from networks.encoder import Encoder
from networks.unrolled_convLSTM import Unrolled_convLSTM
from utils import vos_data_loader as loader
from utils import data_generator as data_generator
from utils.config_reader import conf_reader
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
        input_image_encoder_unstacked = tf.reshape(self.input_image_encoder, [tmp_shape[0]*tmp_shape[1],
                                                                              tmp_shape[2], tmp_shape[3], tmp_shape[4]])

        encoder = Encoder()
        encoder.build(input_image_encoder_unstacked)
        encoder_output = encoder.conv6

        # This will be the set of B batches and F frames to be fed to ConvLSTM
        encoder_output_stacked = tf.reshape(encoder_output, [self.input_image_encoder.shape[0],
                                                             self.input_image_encoder.shape[1], encoder_output.shape[1],
                                                             encoder_output.shape[2], encoder_output.shape[3]])

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
            # ADAM
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam')
            self.train = optimizer.minimize(self.loss)

    def graph_builder(self):
        return self.init, self.input_images_initializer, self.input_image_encoder, self.decoder_output_unstacked,\
               self.mask_output_unstacked, self.groundtruth_mask, self.loss, self.train, self.saver, self.my_graph
#######################################################################################################################
# Build the graph


def main(args, checkpoints_path=None):
    print("Using TensorFlow V.", tf.__version__)
    # Initialization
    dataset_path = "../new_dataset_small/train"
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr

    my_network = My_network(batch_size, LR=lr)
    init, input_images_initializer, input_image_encoder, decoder_output, mask_output,\
    groundtruth_mask, loss, train, saver, my_graph = my_network.graph_builder()

    # # To limit the GPU utilization
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.60)
    # config = ConfigProto(gpu_options=gpu_options)
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device("/device:GPU:0"):
        if checkpoints_path is not None:
            print("Reset the graph")
            tf.reset_default_graph()
        with tf.Session(config=config, graph=my_graph) as sess:
            if checkpoints_path is None:
                # Initialize the graph
                sess.run(init)
            else:
                # Restore the checkpoint from disk.
                saver.restore(sess, checkpoints_path)
                print("Model restored from: ", checkpoints_path)

            # Save the graph
            writer = tf.summary.FileWriter('./graphs', graph=my_graph)

            data_loader = loader.Data_loader(dataset_path)
            object_list = np.array(data_loader.get_object_list())
            n_batch = math.floor(len(object_list) / batch_size)
            print("###INFO: Batch Size is {:d}, total number of batches: {:d} ".format(batch_size, n_batch))

            loss_history_array = np.zeros([n_epochs*n_batch]) # Saves the history of loss over epochs

            for epoch in range(n_epochs):
                # shuffle the dataset at each epoch
                shuffled_object_list = object_list.copy()
                np.random.shuffle(shuffled_object_list)

                for idx in range(n_batch):
                    start_time = time.time()
                    batch_object_list = np.array(shuffled_object_list[idx * batch_size:(idx + 1) * batch_size])
                    assert len(batch_object_list) == batch_size

                    input_initializer, input_encoder, groundtruth = data_loader.get_data_batch(batch_object_list)
                    eta_read = time.time() - start_time
                    start_time = time.time()

                    # Run the session
                    history = sess.run([train, loss], feed_dict={input_images_initializer: input_initializer,
                                                                 input_image_encoder: input_encoder,
                                                                 groundtruth_mask: groundtruth})
                    eta_learn = time.time() - start_time

                    print(" ###INFO: ETA: read:{:.2f}s, train:{:.2f}s, Epoch {:d}/{:d} Progress = %{:.1f},"
                          " Batch #{:d} => Loss = ".format(eta_read, eta_learn, epoch, n_epochs,
                                                             100*idx/n_batch, idx), history[1])

                    # Keep the track of loss
                    loss_history_array[epoch*n_batch+idx] = history[1]

                # save the loss history to file
                np.save("./loss_history.npy", loss_history_array)

                # Save the checkpoints
                save_path = saver.save(sess, "./checkpoints/model", global_step=epoch)
                print("End of Epoch, model saved in path: %s" % save_path)
        print("\n ***DEBUG: TRAINING DONE!")

if __name__ == '__main__':

    # Reading configurations from the YAML file
    configs = conf_reader()
    original_dataset_path = configs["configs"]["path"] # Path to the original dataset
    checkpoints_path = configs["configs"]["checkpoints_path"] # Path to the pre-saved checkpoint files

    # Reading input arguments from command line
    parser = argparse.ArgumentParser(description='Youtube Video Object Segmentation.')
    parser.add_argument('--n_epochs', type=int, default=70, help='Number of Epochs for Training.')
    parser.add_argument('--batch_size', type=int, default=2, help='Size of the mini-batch.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate.')
    args = parser.parse_args()

    # # Proprocess dataset and save it to a new directory as new_dataset_small
    # data_generator.build_new_dataset(original_dataset_path)

    # Run the training
    main(args, checkpoints_path)

