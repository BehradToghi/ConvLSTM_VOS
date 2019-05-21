import pickle
import os
import matplotlib
import math
import matplotlib.pyplot as plt
import numpy as np
import time

def plot_overall_loss(history_path):

    loss_history = np.load(history_path)

    # print(len(loss_history))

    # n_batch = 20
    n_batch = 5942
    n_epochs = math.floor(len(loss_history)/20)



    loss_history_epochs = np.zeros(n_epochs)
    for epoch in range(n_epochs):

        temp_history = loss_history[epoch*n_batch:(epoch+1)*n_batch]
        # print(temp_history)
        loss_epoch = np.mean(temp_history)

        # print(loss_epoch)
        loss_history_epochs[epoch] = loss_epoch


    # print(len(loss_history_epochs))
    plt.plot(range(len(loss_history)), loss_history, 'g.')
    # plt.plot(3961*range(len(loss_history_epochs)), loss_history_epochs, 'r^')
    plt.xlim(0,26000)
    # plt.ylim(0,1)
    plt.show()
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()






# plot_overall_loss("../runs/80epochs_custom/loss_history.npy")
plot_overall_loss("../runs/testrun2/loss_history.npy")
# plot_overall_loss("../loss_history.npy")
#
# while True:
#     plot_overall_loss("../loss_history.npy")