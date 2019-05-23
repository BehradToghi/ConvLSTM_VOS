import time
import math
import os
import json
import numpy as np
import cv2
from shutil import copyfile
from random import randint



class Data_loader:
    def __init__(self, dataset_path):
        print("Data loader initialized")
        self.path = dataset_path

    def get_object_list(self):
        temp_path = os.path.join(self.path, "Annotations")
        print("INFO: Reading object list from directory: ", temp_path)

        object_list = []
        for subdir, dirs, files in os.walk(self.path):
            # print(subdir)

            object_ID = os.path.split(subdir)[1]
            video_ID = os.path.split(os.path.split(subdir)[0])[1]
            category = os.path.split(os.path.split(os.path.split(subdir)[0])[0])[1]
            if len(object_ID) == 1:
                # print(category, video_ID, object_ID)
                object_list.append([category, video_ID, object_ID])
        return object_list


    def get_data_batch(self, batch_object_list):


        self.batch_size = len(batch_object_list)
        input_initializer = np.zeros((self.batch_size, 256, 448, 4))
        input_encoder = np.zeros((self.batch_size, 7, 256, 448, 3))
        groundtruth = np.zeros((self.batch_size, 7, 256, 448, 1))

        for n in range(self.batch_size):
            object = batch_object_list[n]

            # print(object)
            category = object[0]
            video_ID = object[1]
            object_ID = object[2]

            object_path = os.path.join(self.path, "Annotations", category, video_ID, object_ID)
            frame_list = self.get_frame_range(object_path)
            # print(" ###INFO: Reading object:", object_path)
            # check if there are at least 7+1 frames for the object
            if len(frame_list)>7:

                # get a random number
                rnd = randint(0, len(frame_list)-8)
                initial_frame = frame_list[rnd][0:-4]
                # print(initial_frame)

                # Read the first frame to be fed to the LSTM initializer
                input_initializer_tmp = self.load_initializer_input(category, video_ID, object_ID, initial_frame) #(256, 448, 4)
                input_initializer[n] = input_initializer_tmp

                # Read 7 frames to be fed to Encoder
                for ii in range(7):
                    frame = frame_list[rnd+1+ii][0:-4]
                    print(video_ID, object_ID, frame)
                    frame_tmp =  self.load_encoder_input(category, video_ID, object_ID, frame) # (256, 448, 3)
                    input_encoder[n][ii] = frame_tmp

                # Read 7 groundtruth masks to calculate loss
                for jj in range(7):
                    frame = frame_list[rnd+1+jj][0:-4]
                    mask_tmp = self.load_groundtruth(category, video_ID, object_ID, frame)  # (256, 448)
                    mask_tmp = mask_tmp.reshape(mask_tmp.shape+(1,))  # (256, 448, 1)
                    groundtruth[n][jj] = mask_tmp
            # else:
            #     print(" ***DEBUG: There are less than 8 frames, skipping:", object_path)

        return input_initializer, input_encoder, groundtruth


    def get_frame_range(self, object_path):
        frames_list = []
        for subdir, dirs, files in os.walk(object_path):
            for file in files:
                if file.endswith(".png") | file.endswith(".jpg"):
                    frames_list.append(file)
        return frames_list

    def load_groundtruth(self, category, video_ID, object_ID, frame_ID):
        mask_path = os.path.join(self.path, "Annotations", category, video_ID, object_ID, frame_ID + ".png")
        # print(" ###INFO: Reading groundtruth:", mask_path)
        img = cv2.imread(mask_path, 0)
        ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        img = img / 255.0
        # resized_img = skimage.transform.resize(img, (256, 448))
        return img

    def load_encoder_input(self, category, video_ID, object_ID, frame_ID):
        jpeg_path = os.path.join(self.path, "JPEGImages", category, video_ID, object_ID, frame_ID + ".jpg")
        # print(" ###INFO: Reading encoder in:", jpeg_path)
        img = cv2.imread(jpeg_path, 1)
        img = img / 255.0
        # resized_img = skimage.transform.resize(img, (256, 448))
        return img

    def load_initializer_input(self, category, video_ID, object_ID, frame_ID):
        jpeg_path = os.path.join(self.path, "JPEGImages", category, video_ID, object_ID, frame_ID + ".jpg")
        mask_path = os.path.join(self.path, "Annotations", category, video_ID, object_ID, frame_ID + ".png")
        # print(" ###INFO: Reading initializer:", jpeg_path, mask_path)
        # print(jpeg_path, mask_path)
        img = self.channel_adder(jpeg_path, mask_path)
        img = img / 255.0
        # resized_img = skimage.transform.resize(img, (256, 448))
        return img

    def channel_adder(self, jpeg_path, mask_path):
        jpeg = cv2.imread(jpeg_path, 1)
        mask = cv2.imread(mask_path, 0)

        # ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        #
        # cv2.imshow("img", jpeg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        #
        # cv2.imshow("img", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = np.reshape(mask, mask.shape + (1,))
        mixed_frame = np.concatenate((mask, jpeg), axis=2)
        return mixed_frame


    def get_val_object_list(self):
        temp_path = os.path.join(self.path, "JPEGImages")
        print("INFO: Reading object list from directory: ", temp_path)

        object_list = []
        for subdir, dirs, files in os.walk(temp_path):

            object_ID = os.path.split(subdir)[1]
            video_ID = os.path.split(os.path.split(subdir)[0])[1]

            if len(object_ID) == 1:
                # print(video_ID, object_ID)
                object_path = os.path.join(temp_path, video_ID, object_ID)
                # print("\n")
                # print(object_path)
                frame_list = self.get_frame_range(object_path)
                # print(video_ID, object_ID,)
                # print(frame_list)

                if len(frame_list)>7:
                    start = 0
                    isEnd = False
                    #
                    while not isEnd:

                        if start+7<len(frame_list):

                            temp_frame_list = frame_list[start:start+7]

                            start = start+7

                        elif start+7==len(frame_list):
                            temp_frame_list = frame_list[start:start + 7]
                            isEnd = True

                        elif start+7>len(frame_list):
                            temp_frame_list = frame_list[-7:]
                            isEnd = True

                        object_list.append([video_ID, object_ID, temp_frame_list])
                        # print([video_ID, object_ID, temp_frame_list])

                    # print(len(temp_frame_list))
                    assert len(temp_frame_list)==7
                else:
                    print(" ***DEBUG: There are less than 8 frames:", object_path, video_ID, object_ID, frame_list)


                # print(frame_list)
                # object_list.append([video_ID, object_ID, frame_list])
        # print(object_list[0])
        return object_list


    def get_val_data_batch(self, batch_object_list):


        batch_size = len(batch_object_list)
        input_initializer = np.zeros((batch_size, 256, 448, 4))
        input_encoder = np.zeros((batch_size, 7, 256, 448, 3))
        # input_initializer = np.zeros((batch_size, 768, 1344, 4))
        # input_encoder = np.zeros((batch_size, 7, 768, 1344, 3))

        # groundtruth = np.zeros((self.batch_size, 7, 256, 448, 1))

        for n in range(batch_size):
            object = batch_object_list[n]

            # print(object)

            video_ID = object[0]
            object_ID = object[1]
            frame_list = object[2]



            # object_path = os.path.join(self.path, "Annotations", video_ID, object_ID)
            # frame_list = self.get_frame_range(object_path)
            # print(" ###INFO: Reading object:", object_path)
            # check if there are at least 7+1 frames for the object

            # print(n, video_ID, object_ID, frame_list)
            if len(frame_list)>=7:

                initial_mask_path = os.path.join(self.path, "Annotations", video_ID, object_ID)


                initial_frame = os.listdir(initial_mask_path)[0][0:-4]
                # # get a random number
                # rnd = randint(0, len(frame_list)-8)
                # initial_frame = frame_list[rnd][0:-4]
                # # print(initial_frame)

                # Read the first frame to be fed to the LSTM initializer
                input_initializer_tmp = self.load_val_initializer_input(video_ID, object_ID, initial_frame) #(256, 448, 4)
                input_initializer[n] = input_initializer_tmp

                # Read 7 frames to be fed to Encoder
                for ii in range(7):
                    frame = frame_list[ii][0:-4]
                    # print(video_ID, frame)
                    frame_tmp =  self.load_val_encoder_input(video_ID, object_ID, frame) # (256, 448, 3)
                    input_encoder[n][ii] = frame_tmp

                # # Read 7 groundtruth masks to calculate loss
                # for jj in range(7):
                #     # frame = frame_list[rnd+1+jj][0:-4]
                #     frame = initial_frame
                #     mask_tmp = self.load_val_groundtruth(video_ID, object_ID, frame)  # (256, 448)
                #     mask_tmp = mask_tmp.reshape(mask_tmp.shape+(1,))  # (256, 448, 1)
                #     groundtruth[n][jj] = mask_tmp
            # else:
            #     print(" ***DEBUG: There are less than 8 frames")

        return input_initializer, input_encoder



    def load_val_initializer_input(self, video_ID, object_ID, frame_ID):
        jpeg_path = os.path.join(self.path, "JPEGImages", video_ID, object_ID, frame_ID + ".jpg")
        mask_path = os.path.join(self.path, "Annotations", video_ID, object_ID, frame_ID + ".png")

        # print(" ###INFO: Reading initializer:", jpeg_path, mask_path)
        # print(jpeg_path, mask_path)
        # print("#########", video_ID, object_ID, frame_ID)
        img = self.channel_adder(jpeg_path, mask_path)
        img = img / 255.0
        # resized_img = skimage.transform.resize(img, (256, 448))
        return img

    def load_val_encoder_input(self, video_ID, object_ID, frame_ID):
        jpeg_path = os.path.join(self.path, "JPEGImages", video_ID, object_ID, frame_ID + ".jpg")
        # print(" ###INFO: Reading encoder in:", jpeg_path)
        img = cv2.imread(jpeg_path, 1)
        #
        #
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img = img / 255.0
        # resized_img = skimage.transform.resize(img, (256, 448))
        return img





#
# MAIN_DATASET_PATH = "../../new_dataset_small/valid"
#
# data_loader = Data_loader(MAIN_DATASET_PATH)
# object_list = np.array(data_loader.get_val_object_list())
#
# # print(object_list[2])
#
# # print(type(object_list[1][2][1]))
#
#
# print(len(object_list))
# batch_size = 4
# idx = 15
#
# batch_val_object_list = object_list[idx * batch_size:(idx + 1) * batch_size]
# print(batch_val_object_list, "\n")
#
# # input_initializer, input_encoder = data_loader.get_val_data_batch(batch_val_object_list)
# input_initializer, input_encoder = data_loader.get_val_data_batch(batch_val_object_list)
#
# # print("input_initializer.shape", input_initializer.shape)
# # print("input_encoder.shape", input_encoder.shape)


#
# for idx in range(3526//batch_size):
#
#     print("\n NEW BATCH ")
#     batch_val_object_list = object_list[idx * batch_size:(idx + 1) * batch_size]
#     print(batch_val_object_list, "\n")
#
#     # input_initializer, input_encoder = data_loader.get_val_data_batch(batch_val_object_list)
#     data_loader.get_val_data_batch(batch_val_object_list)
#









#
# temp_batch_object_list = object_list[idx * batch_size:(idx + 1) * batch_size]
# isEnd = False
# start = 0
# while not isEnd:
#     if start+7<len()


# input_initializer, input_encoder, groundtruth = data_loader.get_val_data_batch(batch_object_list)
#
# data_loader = Data_loader("/home/toghi/Toghi_WS/PyWS/data/VOS/valid")
# data_loader.get_val_object_list()
# object_list = np.array(data_loader.get_val_object_list())
# print(object_list.shape)
#
# batch_size = 2
# print("INFO: Batch Size is: ", batch_size)
#
# data_loader = Data_loader("/home/toghi/Toghi_WS/PyWS/CAP6412/new_dataset_2/train")
# object_list = np.array(data_loader.get_object_list())
# n_batch = math.floor(len(object_list) / batch_size)
#
# for idx in range(1):
#     batch_object_list = np.array(object_list[idx * batch_size:(idx + 1) * batch_size])
#     assert len(batch_object_list) == batch_size
#
#     print("INFO: Reading batch  {:d}".format(idx))
#     input_initializer, input_encoder, groundtruth = data_loader.get_data_batch(batch_object_list)
#
#     print("input_initializer.shape", input_initializer.shape)
#     print("input_encoder.shape", input_encoder.shape)
#     print("groundtruth.shape", groundtruth.shape)
#
#     input_encoder_unstacked = input_encoder.reshape((input_encoder.shape[0]*input_encoder.shape[1], input_encoder.shape[2], input_encoder.shape[3], input_encoder.shape[4]))
#     print(input_encoder_unstacked.shape)
#
#     for i in range (14):
#
#         image = input_encoder_unstacked[i]
#         cv2.imshow("img", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     input_encoder_stacked = input_encoder_unstacked.reshape((7,2,256,448,3))
#
#     image = input_encoder_stacked[1][0]
#     cv2.imshow("imgbatch0", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     image = input_encoder_stacked[1][1]
#     cv2.imshow("imgbatch1", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
