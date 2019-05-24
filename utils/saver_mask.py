#######################################################################################################################
import os
import numpy as np
import cv2

from PIL import Image
import shutil


class Saver_mask:
    def __init__(self, name, original_dataset_path):

        print("Mask saver initialized")
        self.scenario_name = name
        self.original_dataset_path = original_dataset_path
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
        save_path = os.path.join("..", self.scenario_name, self.video_ID)
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
        annotation_path = os.path.join(self.original_dataset_path, "valid", "Annotations",
                                       self.video_ID, self.frame[:-4]+".png")
        return annotation_path, os.path.isfile(annotation_path)


    def get_mask_shape(self):

        sample_mask_path = os.path.join(self.original_dataset_path, "valid_submit_sample", "Annotations",
                                        self.video_ID, self.frame[:-4]+".png")
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
