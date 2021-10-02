## People and Objects Detection/Counting - (Python, keras, YOLOv3)
The project involved building a capstone system that performs object detection on an image or sequence of frames from a video/live camera by using the You Only Look Once `YOLO V3` Deep Neural Network.

*I was lucky enough to be responsible for building this project during my military service period, The system was tested and deployed on real live sites like highway roads, morning crowded people and road bridges.*
 
- User can customize the classes that only need to detect from the feed.
- System returns a frame/live videos with the pre-customized objects. `Live Detection`
- System returns how many detected objects that existing in image. `Live Counting`  
- On real time, system provide a gui that gives the user ability of control the detection process on specific class. User can turn ON/OFF live detection on class [ex. persons]. `Live Control`
- `Alarm system`, User can configure a maximum number for each class that would be in the live feed. 
    1. More than that number, system will fire an audio alarm for the supervisor to alert that there is break-out at your covered area.
    2. User can control the alarm system by configuring it's safe for exceeding the number of detection. `Safe Button`
    3. User can restart the alarm system by controllable gui. `Danger Button`
    4. User can change the alarm thresh number for each class. `Thresh control` 
    5. User can take a snapshot of live detection frame. `Snapshot`
- **`Arabic` is the main language of detection process and GUI control.**

[image 1.1]

### Requirements:
    Python 3.6 or higher
    tensorflow-gpu==2.3.0rc0
    opencv-python==4.1.1.26
    absl-py
    matplotlib
    pillow
    pygame
    python-bidi
    arabic-reshaper
    tkinter

### Execution Steps:
1. Download the "yolov3.weights", from: https://pjreddie.com/media/files/yolov3.weights to the weights folder.
2. Execute the convert_weights.py:
    1. python convert_weights.py
3. Before testing, make sure that the weights have been converted to TF2 format. The converted weights file is saved in the weights folder.

### Running the code
> Execute the gui_tkinter_place.py file

### Screenshots
[image 1]
[]
