"""
https://machinelearningspace.com/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1/

https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""

import tensorflow as tf
import win32api

from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import time
import numpy as np
from collections import Counter
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image
from yolov3 import YOLOv3Net
import os

# If you don't have enough GPU hardware device available in your machine, uncomment the following three lines:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

model_size = (416, 416, 3)
num_classes = 80
class_load = './data/coco.names'
max_output_size = 100
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'
vid_path = 'data/videos/traffic.mp4'
# cam_path = "rtsp://admin:admin12345@192.168.1.55:554/ISAPI/Streaming/channels/103/h265"
cam_path = "rtsp://admin:admin12345@192.168.1.55:554/ISAPI/Streaming/channels/102/httpPreview"
# cam_path = "rtsp://service:12345@192.168.0.11:554/video.h264"
# cam_path = "rtsp://service:12345@192.168.0.11:554"
# cam_path = "rtsp://admin:admin12345@192.168.1.100/axis-media/media.amp"

##
#   [person, bicycle, car, motorbike, bus, trunk]
needed_objects = [0, 1.0, 2.0, 3.0, 5.0, 8.0]
##
org_offset = 110
txt_offset = 2
fontScale = 1
# person_color = (253, 2, 48)
person_color = (0,0,255)
car_color = (255, 216, 0)
motorbike_color = (189, 46, 0)
truck_color = (0, 210, 255)
# bicycle_color = (255, 178, 2)
bicycle_color = (128, 128, 128)
bus_color = (82, 255, 0)
lineType = 2
##

def main():
    model = YOLOv3Net(cfgfile, model_size, num_classes)

    model.load_weights(weightfile)

    class_names = load_class_names(class_load)

    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    # specify the video input.
    # 0 means input from cam 0.
    # For video, just change the 0 to video path
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(vid_path)
    cap = cv2.VideoCapture(cam_path)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 20)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    vid_chk = cap.isOpened()
    if not vid_chk:
        cap.open(cam_path)

    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        while True:
            # start = time.time()
            # grab frames from the buffer
            cap.grab()
            ret, frame = cap.read()
            if not ret:
                print("Connection Lost!!")
                # break
                cap.open(cam_path)
                continue

            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

            pred = model.predict(resized_frame)

            boxes, scores, classes, nums = output_boxes(
                pred, model_size,
                max_output_size=max_output_size,
                max_output_size_per_class=max_output_size_per_class,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)
            # convert numpy arr to list then slicing to the valid detected objects
            lst_detected_classes_id = classes.numpy()[0].tolist()
            lst_detected_classes_id = lst_detected_classes_id[: int(nums)]
            # Sort the lst_detected_classes_id in ascending order
            lst_detected_classes_id = sorted(lst_detected_classes_id, key=float)
            # print("lst_detected_classes_id = {}".format(lst_detected_classes_id))
            detected_objects = list()
            i = 0
            while i < len(lst_detected_classes_id):
                if (int(lst_detected_classes_id[i]) in needed_objects):
                    detected_objects.append(class_names[int(lst_detected_classes_id[i])])
                i += 1

            print("detected_classes = {}".format(detected_objects))
            dict_objects_occurrence = dict(Counter(detected_objects))
            # Draw predicted boxes#
            img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
            ##

            result = np.asarray(frame)
            # result = frame
            # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            blank_image = np.zeros((130, result.shape[1], 3), np.uint8)
            result = np.concatenate((result, blank_image))
            ##
            ##
            font = cv2.FONT_HERSHEY_SIMPLEX
            org_x = 10
            org_y = 40 + int(img.shape[0])  # 40 + height of image

            ##

            if 'person' in dict_objects_occurrence:
                cv2.putText(result, str(dict_objects_occurrence["person"]), (org_x, org_y), font, fontScale,
                            person_color,
                            lineType, cv2.LINE_AA)
            else:
                cv2.putText(result, "0", (org_x, org_y), font, fontScale, person_color, lineType, cv2.LINE_AA)

            if 'bicycle' in dict_objects_occurrence:
                cv2.putText(result, str(dict_objects_occurrence['bicycle']), (org_x + (1 * org_offset), org_y), font,
                            fontScale, bicycle_color, lineType, cv2.LINE_AA)
            else:
                cv2.putText(result, "0", (org_x + (1 * org_offset), org_y), font, fontScale, bicycle_color, lineType,
                            cv2.LINE_AA)

            if 'motorbike' in dict_objects_occurrence:
                cv2.putText(result, str(dict_objects_occurrence['motorbike']), (org_x + (2 * org_offset), org_y), font,
                            fontScale, motorbike_color, lineType, cv2.LINE_AA)
            else:
                cv2.putText(result, "0", (org_x + (2 * org_offset), org_y), font, fontScale, motorbike_color, lineType,
                            cv2.LINE_AA)

            if 'car' in dict_objects_occurrence:
                cv2.putText(result, str(dict_objects_occurrence['car']), (org_x + (3 * org_offset), org_y), font,
                            fontScale,
                            car_color, lineType, cv2.LINE_AA)
            else:
                cv2.putText(result, "0", (org_x + (3 * org_offset), org_y), font, fontScale, car_color, lineType,
                            cv2.LINE_AA)

            if 'truck' in dict_objects_occurrence:
                cv2.putText(result, str(dict_objects_occurrence['truck']), (org_x + (4 * org_offset), org_y), font,
                            fontScale, truck_color, lineType, cv2.LINE_AA)
            else:
                cv2.putText(result, "0", (org_x + (4 * org_offset), org_y), font, fontScale, truck_color, lineType,
                            cv2.LINE_AA)

            if 'bus' in dict_objects_occurrence:
                cv2.putText(result, str(dict_objects_occurrence['bus']), (org_x + (5 * org_offset), org_y), font,
                            fontScale,
                            bus_color, lineType, cv2.LINE_AA)
            else:
                cv2.putText(result, "0", (org_x + (5 * org_offset), org_y), font, fontScale, bus_color, lineType,
                            cv2.LINE_AA)

            fontpath = "arial.ttf"  # <== https://www.freefontspro.com/14454/arial.ttf
            font = ImageFont.truetype(fontpath, 32)

            text_person = "أفراد"
            reshaped_text_person = arabic_reshaper.reshape(text_person)
            bidi_text_person = get_display(reshaped_text_person)
            img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)
            draw.text((org_x, org_y + txt_offset), bidi_text_person, font=font, fill=person_color)
            # result = np.array(img_pil_person)

            text_bicycle = "دراجة"
            reshaped_text_bicycle = arabic_reshaper.reshape(text_bicycle)
            bidi_text_bicycle = get_display(reshaped_text_bicycle)
            # img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)
            draw.text((org_x + (1 * org_offset - 5), org_y + txt_offset), bidi_text_bicycle, font=font,
                      fill=bicycle_color)
            # result = np.array(img_pil_bicycle)

            text_bicycle = "دراجة" + "\n" + "نارية"
            reshaped_text_bicycle = arabic_reshaper.reshape(text_bicycle)
            bidi_text_bicycle = get_display(reshaped_text_bicycle)
            # img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)
            draw.text((org_x + (2 * org_offset - 5), org_y + txt_offset), bidi_text_bicycle, font=font,
                      fill=motorbike_color)
            # result = np.array(img_pil_bicycle)

            text_bicycle = "سيارات"
            reshaped_text_bicycle = arabic_reshaper.reshape(text_bicycle)
            bidi_text_bicycle = get_display(reshaped_text_bicycle)
            # img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)
            draw.text((org_x + (3 * org_offset - 5), org_y + txt_offset), bidi_text_bicycle, font=font, fill=car_color)
            # result = np.array(img_pil_bicycle)

            text_bicycle = "شاحنات"
            reshaped_text_bicycle = arabic_reshaper.reshape(text_bicycle)
            bidi_text_bicycle = get_display(reshaped_text_bicycle)
            # img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)
            draw.text((org_x + (4 * org_offset - 5), org_y + txt_offset), bidi_text_bicycle, font=font,
                      fill=truck_color)
            # result = np.array(img_pil_bicycle)

            text_bicycle = "أوتوبيس"
            reshaped_text_bicycle = arabic_reshaper.reshape(text_bicycle)
            bidi_text_bicycle = get_display(reshaped_text_bicycle)
            # img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)
            draw.text((org_x + (5 * org_offset - 5), org_y + txt_offset), bidi_text_bicycle, font=font, fill=bus_color)
            result = np.array(img_pil)
            ##
            cv2.imshow(win_name, result)
            # Time:
            # stop = time.time()
            # seconds = stop - start
            # print("Time taken : {0} seconds".format(stop))
            # # Calculate frames per second
            # fps = 1 / seconds
            # print("Estimated frames per second : {0}".format(fps))

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('Detections have been performed successfully.')
        exit()


if __name__ == '__main__':
    main()
