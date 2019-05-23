import json
import numpy as np
import cv2
import os
from shutil import copyfile


# def channel_adder(jpeg_path, mask_path):
#
#     jpeg = cv2.imread(jpeg_path, 1)
#     mask = cv2.imread(mask_path, 0)
#
#     mask = np.reshape(mask, mask.shape+(1,))
#     mixed_frame = np.concatenate((mask, jpeg), axis=2)
#
#     return mixed_frame
#
#
# def object_parser(video):
#
#     for objectN in video["objects"]:
#         tmp_category = video["objects"][objectN]["category"]
#         tmp_frame_list = video["objects"][objectN]["frames"]

def object_counter(json_path):

    total_object_count = 0
    videos = json_reader(json_path)

    for video_ID in videos:
        video = videos[video_ID]
        cnt = 0
        for object_ID in video["objects"]:
            cnt += 1
        if cnt>5:
            print(video_ID, cnt)
        total_object_count += cnt
    return total_object_count



def json_reader(json_path):

    with open(json_path) as json_file:
        meta = json.load(json_file)
        videos = meta["videos"]

    return videos


def save_frame(original_dataset_path, image, video_type, video_ID, category, frame_ID, object_ID):



    dir_jpeg = os.path.join("..", "new_dataset_small", video_type, "JPEGImages", category, video_ID, str(object_ID))
    dir_annot = os.path.join("..", "new_dataset_small", video_type, "Annotations", category, video_ID, str(object_ID))

    if not os.path.exists(dir_jpeg):
        os.makedirs(dir_jpeg)

    if not os.path.exists(dir_annot):
        os.makedirs(dir_annot)

    annot_path = os.path.join(dir_annot, frame_ID+".png")
    cv2.imwrite(annot_path, image)

    jpeg_source_path = os.path.join(original_dataset_path, video_type, "JPEGImages", video_ID, frame_ID+".jpg")
    jpeg_target_path = os.path.join(dir_jpeg, frame_ID+".jpg")
    # print(jpeg_source_path, jpeg_target_path)
    resize_and_copy(jpeg_source_path, jpeg_target_path)

def resize_and_copy(source, target):
    # jpeg_path = os.path.join(self.path, "JPEGImages", category, video_ID, object_ID, frame_ID + ".jpg")
    img = cv2.imread(source, 1)
    # img = img / 255.0

    # resized_img = skimage.transform.resize(img, (256, 448))
    resized_img = cv2.resize(img, (448, 256))
    # resized_img = cv2.resize(img, (1344, 768))
    # resized_img = img
    # if np.count_nonzero(resized_img) == 0:
    #     print("ERROR")
    # cv2.imshow("img", resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(target, resized_img)


def parse_objects(original_dataset_path, video_type, video_ID, category, frame_ID, object_ID):

    colors = [[103, 95, 236], [87, 145, 249], [99, 200, 250], [148, 199, 153], [178, 179, 98]]
    unit_array = np.array([1, 1, 1])

    frame_path = os.path.join(original_dataset_path, video_type, "Annotations", video_ID, frame_ID+".png")
    img = cv2.imread(frame_path, 1)
    color = colors[object_ID-1]


    colorl = np.subtract(color, unit_array)
    colorh = np.add(color, unit_array)

    mask = cv2.inRange(img, colorl, colorh)

    if np.count_nonzero(mask) == 0:
        print("ERROR: no annotation in video ", video_ID, " object # ", object_ID, " frame # ", frame_ID)

    else:
        result = cv2.bitwise_and(img, img, mask=mask)

        # resized_img = skimage.transform.resize(result, (256, 448))
        resized_img = cv2.resize(result, (448, 256))
        save_frame(original_dataset_path, resized_img, video_type, video_ID, category, frame_ID, object_ID)


def build_new_dataset (original_dataset_path):
    video_type = "train"
    json_path = os.path.join(original_dataset_path, video_type, "meta.json")
    videos = json_reader(json_path)
    counter = 0

    for video_ID in videos:
        counter += 1
        print("Reading video #", counter, " video_ID = ", video_ID)
        video = videos[video_ID]
        for object_ID in video["objects"]:
            category = video["objects"][object_ID]["category"]
            frame_list = video["objects"][object_ID]["frames"]
            object_ID = int(object_ID, 10)
            if object_ID <6:
                for frame_ID in frame_list:
                    parse_objects(original_dataset_path, video_type, video_ID, category, frame_ID, object_ID)
            else:
                print("ERROR: more than 5 objects in video ", video_ID)


def build_new_valid_dataset (original_dataset_path):
    video_type = "valid"
    json_path = os.path.join(original_dataset_path, video_type, "meta.json")
    videos = json_reader(json_path)
    counter = 0
    for video_ID in videos:
        counter += 1
        print("Reading validation video #", counter, " video_ID = ", video_ID)
        video = videos[video_ID]
        for object_ID in video["objects"]:
            category = video["objects"][object_ID]["category"]
            frame_list = video["objects"][object_ID]["frames"]
            object_ID = int(object_ID, 10)
            if object_ID <6:
                for frame_ID in frame_list:
                    if frame_ID == frame_list[0]:
                        parse_valid_objects(original_dataset_path, video_type, video_ID, category, frame_ID, object_ID)
                        save_valid_image_frame(original_dataset_path, video_type, video_ID, frame_ID, object_ID)
                    else:
                        save_valid_image_frame(original_dataset_path, video_type, video_ID, frame_ID, object_ID)
            else:
                print("ERROR: more than 5 objects in video ", video_ID)

def parse_valid_objects(original_dataset_path, video_type, video_ID, category, frame_ID, object_ID):

    colors = [[103, 95, 236], [87, 145, 249], [99, 200, 250], [148, 199, 153], [178, 179, 98]]
    unit_array = np.array([1, 1, 1])


    frame_path = os.path.join(original_dataset_path, video_type, "Annotations", video_ID, frame_ID+".png")
    img = cv2.imread(frame_path, 1)
    color = colors[object_ID-1]

    colorl = np.subtract(color, unit_array)
    colorh = np.add(color, unit_array)
    mask = cv2.inRange(img, colorl, colorh)

    if np.count_nonzero(mask) == 0:
        print("ERROR: no annotation in video ", video_ID, " object # ", object_ID, " frame # ", frame_ID)

    else:
        result = cv2.bitwise_and(img, img, mask=mask)
        resized_mask = cv2.resize(result, (1344, 768))
        # resized_mask = cv2.resize(result, (448, 256))
        # resized_mask = result
        save_valid_mask_frame(resized_mask, video_type, video_ID, frame_ID, object_ID)
        # save_valid_image_frame(original_dataset_path, video_type, video_ID, frame_ID, object_ID)
    # else:
    #     save_valid_image_frame(original_dataset_path, video_type, video_ID, frame_ID, object_ID)


def save_valid_image_frame(original_dataset_path, video_type, video_ID, frame_ID, object_ID):

    dir_jpeg = os.path.join("..", "..", "new_dataset_large_valid", video_type, "JPEGImages", video_ID, str(object_ID))
    if not os.path.exists(dir_jpeg):
        os.makedirs(dir_jpeg)
    jpeg_source_path = os.path.join(original_dataset_path, video_type, "JPEGImages", video_ID, frame_ID+".jpg")
    jpeg_target_path = os.path.join(dir_jpeg, frame_ID+".jpg")
    resize_and_copy(jpeg_source_path, jpeg_target_path)

def save_valid_mask_frame(image, video_type, video_ID, frame_ID, object_ID):
    # debug += 1
    # print("debug")
    dir_annot = os.path.join("..", "..", "new_dataset_large_valid", video_type, "Annotations", video_ID, str(object_ID))
    if not os.path.exists(dir_annot):
        os.makedirs(dir_annot)
    exists = os.path.isfile(dir_annot)
    if exists:
        print("########## ERROR", dir_annot)
    annot_path = os.path.join(dir_annot, frame_ID+".png")
    cv2.imwrite(annot_path, image)
