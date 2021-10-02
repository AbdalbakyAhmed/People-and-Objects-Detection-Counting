"""
https://machinelearningspace.com/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1/

https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""
import threading
import queue
import tensorflow as tf

from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import time

from yolov3 import YOLOv3Net

sys_que = queue.Queue()

# If you don't have enough GPU hardware device available in your machine, uncomment the following three lines:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##
model_size = (416, 416, 3)
num_classes = 80
class_name = './data/coco.names'
max_output_size = 100
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5
##
cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'
vid_path = 'data/videos/traffic.mp4'
# cam_path = "rtsp://admin:admin12345@192.168.1.55:554/ISAPI/Streaming/channels/103/h265"
cam_path = "rtsp://admin:admin12345@192.168.1.55:554/ISAPI/Streaming/channels/102/httpPreview"
##
cap = cv2.VideoCapture(cam_path)
##
model = YOLOv3Net(cfgfile, model_size, num_classes)
model.load_weights(weightfile)
class_names = load_class_names(class_name)
##
win_name = 'Yolov3 detection'
def receive():
    print("start Reveive")
    # specify the video input.
    # 0 means input from cam 0.
    # For video, just change the 0 to video path

    if cap.isOpened():
        ret, frame = cap.read()
        sys_que.put(frame)

    while True:
        if cap.isOpened():
            time.sleep(.01)
            ret, frame = cap.read()
            sys_que.put(frame)
        else:
            pass


def main():
    print("Start Displaying")

    cv2.namedWindow(win_name)

    # frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            if sys_que.empty() != True:
                start = time.time()
                frame = sys_que.get()
                print("len of frame :", len(frame))
                if frame is not None:
                    resized_frame = tf.expand_dims(frame, 0)
                    resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

                    pred = model.predict(resized_frame)

                    boxes, scores, classes, nums = output_boxes(
                        pred, model_size,
                        max_output_size=max_output_size,
                        max_output_size_per_class=max_output_size_per_class,
                        iou_threshold=iou_threshold,
                        confidence_threshold=confidence_threshold)

                    img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
                    cv2.imshow(win_name, img)
                    # print("img =", img)
                    # print("img type =", type(img))

                    stop = time.time()

                    seconds = stop - start
                    # print("Time taken : {0} seconds".format(seconds))

                    # Calculate frames per second
                    fps = 1 / seconds
                    print("Estimated frames per second : {0}".format(fps))
                else:
                    pass
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('Detections have been performed successfully.')


if __name__ == '__main__':
    # main()
    p1 = threading.Thread(target=receive)
    p2 = threading.Thread(target=main)
    p1.start()
    p2.start()


############################################################################################
############################################################################################
############################################################################################
"""
https://machinelearningspace.com/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1/

https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""
import time
import tensorflow as tf

from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import time

from yolov3 import YOLOv3Net
##

from threading import Thread
# Define VideoStream class to handle streaming of video from external camera in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(1280,720),framerate=30):
        # Initialize the PiCamera and the camera image stream

        self.stream = cv2.VideoCapture("rtsp://admin:admin12345@192.168.1.55:554/ISAPI/Streaming/channels/102/httpPreview")
        # self.stream = cv2.VideoCapture('./data/traffic2.mp4')

        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return
            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
## End of class

##
# If you don't have enough GPU hardware device available in your machine, uncomment the following three lines:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_size = (416, 416, 3)
num_classes = 80
class_name = './data/coco.names'
max_output_size = 100
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'
vid_path = 'data/videos/traffic.mp4'
# cam_path = "rtsp://admin:admin12345@192.168.1.55:554/ISAPI/Streaming/channels/103/h265"
cam_path = "rtsp://admin:admin12345@192.168.1.55:554/ISAPI/Streaming/channels/102/httpPreview"

cap = VideoStream(resolution=(1280,720),framerate=25).start()

def main():
    model = YOLOv3Net(cfgfile, model_size, num_classes)

    model.load_weights(weightfile)

    class_names = load_class_names(class_name)

    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    # specify the vidoe input.
    # 0 means input from cam 0.
    # For vidio, just change the 0 to video path
    # cap = cv2.VideoCapture(0)

    try:
        while True:
            start = time.time()

            frame = cap.read()
            if frame is not None:
                print("frame OK")
                resized_frame = tf.expand_dims(frame, 0)
                resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

                pred = model.predict(resized_frame)

                boxes, scores, classes, nums = output_boxes(
                    pred, model_size,
                    max_output_size=max_output_size,
                    max_output_size_per_class=max_output_size_per_class,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

                img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
                cv2.imshow(win_name, img)

                stop = time.time()

                seconds = stop - start
                # print("Time taken : {0} seconds".format(seconds))

                # Calculate frames per second
                fps = 1 / seconds
                print("Estimated frames per second : {0}".format(fps))
                time.sleep(.0001)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
            else:
                print("frame None")
                cap.update()
                pass
    finally:
        cv2.destroyAllWindows()
        cap.stop()
        print('Detections have been performed successfully.')


if __name__ == '__main__':
    main()

############################################################################################
############################################################################################
