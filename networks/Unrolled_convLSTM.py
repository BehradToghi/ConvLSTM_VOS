#######################################################################################################################
import tensorflow as tf
from utils import rnn as rnn_net
from utils import rnn_cell as rnn_cell
from utils.rnn_cell_carlthome import ConvLSTMCell

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

            # cell = ConvLSTMCell(shape, filters, kernel)
            cell = rnn_cell.ConvLSTMCell(2, (8, 14, 512), 512, (5, 5), name="conv_lstm_cell")

            self.lstm_output, state = rnn_net.dynamic_rnn(cell, input_frames, initial_state=init_state, dtype=input_frames.dtype)

