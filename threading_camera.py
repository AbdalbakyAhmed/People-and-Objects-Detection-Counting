import cv2
from threading import Thread


# Define VideoStream class to handle streaming of video from external camera in separate processing thread


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(1280, 720), cam_path=0):
        # Initialize the PiCamera and the camera image stream

        self.stream = cv2.VideoCapture(cam_path)
        # self.stream = cv2.VideoCapture('./data/traffic2.mp4')

        # ret = self.stream.set(3, resolution[0])
        # ret = self.stream.set(4, resolution[1])
        # Read first frame from the stream
        self.grabbed, self.frame = self.stream.read()
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
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# End of class