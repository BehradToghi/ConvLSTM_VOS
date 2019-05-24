'''
Classes and objects for data loader module to load frames of individual objects (parsed from the main annotation file)
for both training and evaluation phases.
'''

import os
import numpy as np
import cv2
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
            object_ID = os.path.split(subdir)[1]
            video_ID = os.path.split(os.path.split(subdir)[0])[1]
            category = os.path.split(os.path.split(os.path.split(subdir)[0])[0])[1]
            if len(object_ID) == 1:
                object_list.append([category, video_ID, object_ID])
        return object_list

    def get_data_batch(self, batch_object_list):
        self.batch_size = len(batch_object_list)
        input_initializer = np.zeros((self.batch_size, 256, 448, 4))
        input_encoder = np.zeros((self.batch_size, 7, 256, 448, 3))
        groundtruth = np.zeros((self.batch_size, 7, 256, 448, 1))

        for n in range(self.batch_size):
            object = batch_object_list[n]
            category = object[0]
            video_ID = object[1]
            object_ID = object[2]

            object_path = os.path.join(self.path, "Annotations", category, video_ID, object_ID)
            frame_list = self.get_frame_range(object_path)
            if len(frame_list)>7:
                # get a random number
                rnd = randint(0, len(frame_list)-8)
                initial_frame = frame_list[rnd][0:-4]

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
        img = cv2.imread(mask_path, 0)
        ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        img = img / 255.0
        return img

    def load_encoder_input(self, category, video_ID, object_ID, frame_ID):
        jpeg_path = os.path.join(self.path, "JPEGImages", category, video_ID, object_ID, frame_ID + ".jpg")
        img = cv2.imread(jpeg_path, 1)
        img = img / 255.0
        return img

    def load_initializer_input(self, category, video_ID, object_ID, frame_ID):
        jpeg_path = os.path.join(self.path, "JPEGImages", category, video_ID, object_ID, frame_ID + ".jpg")
        mask_path = os.path.join(self.path, "Annotations", category, video_ID, object_ID, frame_ID + ".png")
        img = self.channel_adder(jpeg_path, mask_path)
        img = img / 255.0
        return img

    def channel_adder(self, jpeg_path, mask_path):
        jpeg = cv2.imread(jpeg_path, 1)
        mask = cv2.imread(mask_path, 0)
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
                object_path = os.path.join(temp_path, video_ID, object_ID)
                frame_list = self.get_frame_range(object_path)
                if len(frame_list)>7:
                    start = 0
                    isEnd = False
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
                    assert len(temp_frame_list)==7
                else:
                    print(" ***DEBUG: There are less than 8 frames:", object_path, video_ID, object_ID, frame_list)
        return object_list

    def get_val_data_batch(self, batch_object_list):
        batch_size = len(batch_object_list)
        input_initializer = np.zeros((batch_size, 256, 448, 4))
        input_encoder = np.zeros((batch_size, 7, 256, 448, 3))

        for n in range(batch_size):
            object = batch_object_list[n]
            video_ID = object[0]
            object_ID = object[1]
            frame_list = object[2]
            if len(frame_list)>=7:
                initial_mask_path = os.path.join(self.path, "Annotations", video_ID, object_ID)
                initial_frame = os.listdir(initial_mask_path)[0][0:-4]

                # Read the first frame to be fed to the LSTM initializer
                input_initializer_tmp = self.load_val_initializer_input(video_ID, object_ID, initial_frame)
                input_initializer[n] = input_initializer_tmp

                # Read 7 frames to be fed to Encoder
                for ii in range(7):
                    frame = frame_list[ii][0:-4]
                    frame_tmp =  self.load_val_encoder_input(video_ID, object_ID, frame) # (256, 448, 3)
                    input_encoder[n][ii] = frame_tmp
            # else:
            #     print(" ***DEBUG: There are less than 8 frames")
        return input_initializer, input_encoder

    def load_val_initializer_input(self, video_ID, object_ID, frame_ID):
        jpeg_path = os.path.join(self.path, "JPEGImages", video_ID, object_ID, frame_ID + ".jpg")
        mask_path = os.path.join(self.path, "Annotations", video_ID, object_ID, frame_ID + ".png")

        img = self.channel_adder(jpeg_path, mask_path)
        img = img / 255.0
        return img

    def load_val_encoder_input(self, video_ID, object_ID, frame_ID):
        jpeg_path = os.path.join(self.path, "JPEGImages", video_ID, object_ID, frame_ID + ".jpg")
        img = cv2.imread(jpeg_path, 1)
        img = img / 255.0
        return img
