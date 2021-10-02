from pygame import mixer  # Load the popular external library
##
import threading
import time
##
from collections import Counter

import arabic_reshaper
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
from bidi.algorithm import get_display

from utils import load_class_names, output_boxes, draw_outputs, resize_image
from yolov3 import YOLOv3Net
from threading_camera import VideoStream

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
person_color = (0, 0, 255)
car_color = (255, 216, 0)
motorbike_color = (189, 46, 0)
truck_color = (0, 210, 255)
# bicycle_color = (255, 178, 2)
bicycle_color = (128, 128, 128)
bus_color = (82, 255, 0)
lineType = 2
##
model = YOLOv3Net(cfgfile, model_size, num_classes)
model.load_weights(weightfile)
class_names = load_class_names(class_load)
##
mixer.init()
##
counter_thresh = 20  # seconds
danger_thresh = 10


def alarm_trigger(val):
    # win32api.Beep(570, 1000)
    if val == 1:
        # os.system("real-police-siren-ringtone.mp3")
        mixer.music.load("real-police-siren-ringtone.mp3")
        mixer.music.play()
    if val == 2:
        mixer.music.load("beep-08b.mp3")
        mixer.music.play(2)


# th = threading.Thread(target=alarm_trigger(0))


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = VideoStream(resolution=(1080, 720), cam_path=video_source)
        if not self.vid.stream.isOpened():
            raise ValueError("Unable to open video source", video_source)
        # Classes configuration variables
        self.dict_detection_activate_config = {
            # 'KEY' : bool:Activate_detect
            'person': True,
            'person_thresh': 100,
            'bike': True,
            'bike_thresh': 100,
            'motoBike': True,
            'motoBike_thresh': 100,
            'car': True,
            'car_thresh': 100,
            'trunk': True,
            'trunk_thresh': 100,
            'bus': True,
            'bus_thresh': 100,
            ##
            'alarm_state': False,
            'safe_state': False,

        }
        # Get video source width and height
        self.width = self.vid.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("\n", self.width, self.height, '\n')
        ##
        self.safe_flag = False
        self.counter_start = time.time()
        ##
        # th.start()
        print('\n 1 \n')

    @property
    def get_frame(self):
        if self.vid.stream.isOpened():
            print('\n 2 \n')
            ret, frame = self.vid.grabbed, self.vid.read()
            if ret:
                print('\n 3 \n')
                resized_frame = tf.expand_dims(frame, 0)
                resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

                pred = model.predict(resized_frame)

                boxes, scores, classes, nums = output_boxes(
                    pred, model_size,
                    max_output_size=max_output_size,
                    max_output_size_per_class=max_output_size_per_class,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold
                )
                # convert numpy arr to list then slicing to the valid detected objects
                lst_detected_classes_id = classes.numpy()[0].tolist()
                lst_detected_classes_id = lst_detected_classes_id[: int(nums)]
                # Sort the lst_detected_classes_id in ascending order
                lst_detected_classes_id = sorted(lst_detected_classes_id, key=float)
                # print("lst_detected_classes_id = {}".format(lst_detected_classes_id))
                detected_objects = list()
                i = 0
                while i < len(lst_detected_classes_id):
                    if int(lst_detected_classes_id[i]) in needed_objects:
                        detected_objects.append(class_names[int(lst_detected_classes_id[i])])
                    i += 1

                print("detected_classes = {}".format(detected_objects))
                dict_objects_occurrence = dict(Counter(detected_objects))
                # print('dict_objects_occurrence = {}'.format(dict_objects_occurrence))
                # Draw predicted boxes#
                img = draw_outputs(frame, boxes, scores, classes, nums, class_names,
                                   self.dict_detection_activate_config)
                ##

                # result = np.asarray(frame)
                # result = frame
                # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                blank_image = np.zeros((130, result.shape[1], 3), np.uint8)
                result = np.concatenate((result, blank_image))
                ##
                ##
                font = cv2.FONT_HERSHEY_SIMPLEX
                org_x = 10
                org_y = 40 + int(img.shape[0])  # 40 + height of image

                ##
                if self.dict_detection_activate_config['person']:
                    if 'person' in dict_objects_occurrence:
                        cv2.putText(result, str(dict_objects_occurrence["person"]), (org_x, org_y), font, fontScale,
                                    person_color,
                                    lineType, cv2.LINE_AA)
                    else:
                        cv2.putText(result, "0", (org_x, org_y), font, fontScale, person_color, lineType, cv2.LINE_AA)
                #
                if self.dict_detection_activate_config['bike']:
                    if 'bicycle' in dict_objects_occurrence:
                        cv2.putText(result, str(dict_objects_occurrence['bicycle']), (org_x + (1 * org_offset), org_y),
                                    font,
                                    fontScale, bicycle_color, lineType, cv2.LINE_AA)
                    else:
                        cv2.putText(result, "0", (org_x + (1 * org_offset), org_y), font, fontScale, bicycle_color,
                                    lineType,
                                    cv2.LINE_AA)
                #
                if self.dict_detection_activate_config['motoBike']:
                    if 'motorbike' in dict_objects_occurrence:
                        cv2.putText(result, str(dict_objects_occurrence['motorbike']),
                                    (org_x + (2 * org_offset), org_y),
                                    font,
                                    fontScale, motorbike_color, lineType, cv2.LINE_AA)
                    else:
                        cv2.putText(result, "0", (org_x + (2 * org_offset), org_y), font, fontScale, motorbike_color,
                                    lineType,
                                    cv2.LINE_AA)
                #
                if self.dict_detection_activate_config['car']:
                    if 'car' in dict_objects_occurrence:
                        cv2.putText(result, str(dict_objects_occurrence['car']), (org_x + (3 * org_offset), org_y),
                                    font,
                                    fontScale,
                                    car_color, lineType, cv2.LINE_AA)
                    else:
                        cv2.putText(result, "0", (org_x + (3 * org_offset), org_y), font, fontScale, car_color,
                                    lineType,
                                    cv2.LINE_AA)
                #
                if self.dict_detection_activate_config['trunk']:
                    if 'truck' in dict_objects_occurrence:
                        cv2.putText(result, str(dict_objects_occurrence['truck']), (org_x + (4 * org_offset), org_y),
                                    font,
                                    fontScale, truck_color, lineType, cv2.LINE_AA)
                    else:
                        cv2.putText(result, "0", (org_x + (4 * org_offset), org_y), font, fontScale, truck_color,
                                    lineType,
                                    cv2.LINE_AA)
                #
                if self.dict_detection_activate_config['bus']:
                    if 'bus' in dict_objects_occurrence:
                        cv2.putText(result, str(dict_objects_occurrence['bus']), (org_x + (5 * org_offset), org_y),
                                    font,
                                    fontScale,
                                    bus_color, lineType, cv2.LINE_AA)
                    else:
                        cv2.putText(result, "0", (org_x + (5 * org_offset), org_y), font, fontScale, bus_color,
                                    lineType,
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
                draw.text((org_x + (3 * org_offset - 5), org_y + txt_offset), bidi_text_bicycle, font=font,
                          fill=car_color)
                # result = np.array(img_pil_bicycle)

                text_bicycle = "شاحنات"
                reshaped_text_bicycle = arabic_reshaper.reshape(text_bicycle)
                bidi_text_bicycle = get_display(reshaped_text_bicycle)
                # img_pil = Image.fromarray(result)
                draw = ImageDraw.Draw(img_pil)
                draw.text((org_x + (4 * org_offset - 5), org_y + txt_offset), bidi_text_bicycle, font=font,
                          fill=truck_color)
                # result = np.array(img_pil_bicycle)

                text_bicycle = "حافلات"
                reshaped_text_bicycle = arabic_reshaper.reshape(text_bicycle)
                bidi_text_bicycle = get_display(reshaped_text_bicycle)
                # img_pil = Image.fromarray(result)
                draw = ImageDraw.Draw(img_pil)
                draw.text((org_x + (5 * org_offset - 5), org_y + txt_offset), bidi_text_bicycle, font=font,
                          fill=bus_color)
                # result = np.array(img_pil)
                result = img_pil
                # Triggering an alarm for urgent events
                chk_counter = time.time()
                temp_chk = chk_counter - self.counter_start
                #
                self.fire_alarm_trigger(
                    temp_chk, chk_counter, dict_objects_occurrence,
                    self.dict_detection_activate_config['person_thresh'],
                    self.dict_detection_activate_config['person']
                )
                #
                if self.dict_detection_activate_config['alarm_state']:
                    self.safe_flag = False
                    mixer.music.stop()
                    self.dict_detection_activate_config['alarm_state'] = False

                if self.dict_detection_activate_config['safe_state']:
                    self.safe_flag = True
                    mixer.music.stop()
                    self.dict_detection_activate_config['safe_state'] = False

                return ret, result
            else:
                return ret, None
        else:
            return None

    ##

    def fire_alarm_trigger(self, ex_thresh, update_time, dict_detect, num_chk, show_activate):
        if ex_thresh > counter_thresh:
            if 'person' in dict_detect:
                if dict_detect['person'] > num_chk and show_activate:
                    alarm_trigger(2) if self.safe_flag else alarm_trigger(1)
                    self.counter_start = update_time

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.stream.isOpened():
            self.vid.stream.release()
