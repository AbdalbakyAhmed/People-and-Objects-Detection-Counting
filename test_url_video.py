import numpy as np
import cv2

from threading import Thread


# Define VideoStream class to handle streaming of video from external camera in separate processing thread


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(1280, 720), cam_path=0, framerate=30):
        # Initialize the PiCamera and the camera image stream

        self.stream = cv2.VideoCapture(cam_path)
        # self.stream = cv2.VideoCapture('./data/traffic2.mp4')

        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
        # Variable to control when the camera is stopped
        self.stopped = False

    @property
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


# End of class


# Open a sample video available in sample-videos
# cam = 'rtsp://service:12345@192.168.0.11:554/video.h264'
# cam = 'rtsp://service:12345@192.168.0.11:554/h264_sd.sdp'
cam = 'rtsp://service:12345@192.168.0.11:554/out.h264'
# cam = 'rtsp://service:12345@192.168.0.11:554'
vcap = VideoStream(resolution=(1080, 720), cam_path=0, framerate=20).start
print("\n chk \n",vcap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
# vcap = cv2.VideoCapture(cam_path)
# if not vcap.isOpened():
#    print "File Cannot be Opened"

while True:
    # Capture frame-by-frame
    frame = vcap.read()
    # print cap.isOpened(), ret
    if frame is not None:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print("Frame is None")
        break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()
print("Video stop")
