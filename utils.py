"""
Written by: Rahmad Sadli
Website : https://machinelearningspace.com

I finally made this program simple and readable
Hopefully, this program will help some beginners like me to understand better object detection.
If you want to redistribute it, just keep the author's name.

In oder to execute this program, you need to install TensorFlow 2.0 and opencv 4.x

For more details about how this program works. I explained well about it, just click the link below:
https://machinelearningspace.com/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1/

Credit to:
Ayoosh Kathuria who shared his great work using pytorch, really appreaciated it.
https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""

import tensorflow as tf
import numpy as np
import cv2
import time

needed_objects = [0, 1.0, 2.0, 3.0, 5.0, 8.0]
# person_color = (253, 2, 48)
person_color = (0, 0, 255)
car_color = (255, 216, 0)
motorbike_color = (189, 46, 0)
truck_color = (0, 210, 255)
# bicycle_color = (255, 178, 2)
bicycle_color = (128, 128, 128)
bus_color = (82, 255, 0)


def resize_image(inputs, modelsize):
    inputs = tf.image.resize(inputs, modelsize)
    return inputs


def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class,
                 iou_threshold, confidence_threshold):
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0

    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y, confidence, classes], axis=-1)

    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size,
                                      max_output_size_per_class, iou_threshold, confidence_threshold)

    return boxes_dicts


##
def draw_output_with_color(img, dim1, dim2, color, name, iou):
    img = cv2.rectangle(img, dim1, dim2, color, 2)
    img = cv2.putText(img, '{} {:.1f}%'.format(name, iou*100), dim1,
                      cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    pass


##
def draw_outputs(img, boxes, objectness, classes, nums, class_names, dict_detection_activate):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes = np.array(boxes)
    # print("utils scope\n######")
    # print("boxes = ", boxes[0, 0:2])
    # print("boxes = ", boxes[0][0:2])
    # print("classes = ", classes.numpy()[:int(nums)].tolist())
    # print("nums = ", int(nums))  # print the integer of valid detections

    for i in range(nums):
        # print("Type i = ",type(i),"i=", i)
        if int(classes[i]) in needed_objects:
            x1y1 = tuple((boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
            x2y2 = tuple((boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))
            # img = cv2.rectangle(img, (x1y1), (x2y2), (255, 0, 0), 2)
            # img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]), (x1y1),
            #                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            x = classes[i]
            if x == 0 and dict_detection_activate['person']:  # person
                draw_output_with_color(img, x1y1, x2y2, person_color, class_names[int(classes[i])], objectness[i])
            elif x == 1 and dict_detection_activate['bike']:  # bicycle
                draw_output_with_color(img, x1y1, x2y2, bicycle_color, class_names[int(classes[i])], objectness[i])
            elif x == 2 and dict_detection_activate['car']:  # car
                draw_output_with_color(img, x1y1, x2y2, car_color, class_names[int(classes[i])], objectness[i])
            elif x == 3 and dict_detection_activate['motoBike']:  # motorbike
                draw_output_with_color(img, x1y1, x2y2, motorbike_color, class_names[int(classes[i])], objectness[i])
            elif x == 5 and dict_detection_activate['bus']:  # bus
                draw_output_with_color(img, x1y1, x2y2, bus_color, class_names[int(classes[i])], objectness[i])
            elif x == 8 and dict_detection_activate['trunk']:  # trunk
                draw_output_with_color(img, x1y1, x2y2, truck_color, class_names[int(classes[i])], objectness[i])
            else:
                pass
        else:
            pass
    # print("utils scope end !!!")
    return img


def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def non_max_suppression(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold, confidence_threshold):
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox = bbox / model_size[0]

    scores = confs * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections
