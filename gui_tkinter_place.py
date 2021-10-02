import time
import tkinter as tk
from tkinter.messagebox import showerror

import PIL.Image
import PIL.ImageTk
import cv2

# from gui_video_read_thread import MyVideoCapture
from gui_video_read import MyVideoCapture


def dis_num_bike(num):
    # num = self.persons.get()
    if num is '':
        num = 0
    # Try convert a str to int
    # If unable eg. int('hello') or int('5.5')
    # then show an error.
    try:
        num = int(num)
    # ValueError is the type of error expected from this conversion
    except ValueError:
        # Display Error Window (Title, Prompt)
        showerror('Non-Int Error', 'Please enter an integer')
    else:
        print(num)


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.top = tk.Tk()
        self.top.title("CONTROL")
        self.top.geometry("400x500")
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height + 200)
        self.canvas.pack(side=tk.LEFT, anchor=tk.CENTER, expand=False)
        ##
        # Tkinter constant dimensions for Buttons
        self.x1_axis = 30
        self.x2_axis = 181
        self.x3_axis = 310
        self.y_axis = 50
        self.x_offset = 50
        self.y_offset = 40
        ##
        # CheckButtons DAO
        self.person_chk = tk.IntVar(self.top)
        self.bike_chk = tk.IntVar(self.top)
        self.motoBike_chk = tk.IntVar(self.top)
        self.car_chk = tk.IntVar(self.top)
        self.trunk_chk = tk.IntVar(self.top)
        self.bus_chk = tk.IntVar(self.top)
        ##
        self.person_get_thresh = tk.Entry(self.top)
        self.bike_get_thresh = tk.Entry(self.top)
        self.motoBike_get_thresh = tk.Entry(self.top)
        self.car_get_thresh = tk.Entry(self.top)
        self.trunk_get_thresh = tk.Entry(self.top)
        self.bus_get_thresh = tk.Entry(self.top)
        ##
        self.display_snapshot(self.top)
        self.display_person(self.top)
        self.display_bike(self.top)
        self.display_moto(self.top)
        self.display_car(self.top)
        self.display_truck(self.top)
        self.display_bus(self.top)
        #
        self.display_alarm_chk_system(self.top)
        #
        # self.display_snapshot()
        # self.display_person()
        # self.display_bike()
        # self.display_moto()
        # self.display_cars()
        # self.display_truck()
        # self.display_bus()
        ##
        # Disable Exit (or [ X ]) in tkinter Windows
        self.window.protocol("WM_DELETE_WINDOW", self.disable_event)
        self.top.protocol("WM_DELETE_WINDOW", self.disable_event)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 5  # old value = 15
        self.update()

        self.window.mainloop()

    ##
    def alarm_state_getter(self):
        self.vid.dict_detection_activate_config['alarm_state'] = True
        print("\nAlarm System is initiated again\n")

    def safe_state_getter(self):
        self.vid.dict_detection_activate_config['safe_state'] = True
        print("\nSafe Mode Activated!!\n")

    ##
    # CheckButtons Setter functions
    def person_chk_setter(self):
        self.vid.dict_detection_activate_config['person'] = self.person_chk.get()

    def bike_chk_setter(self):
        self.vid.dict_detection_activate_config['bike'] = self.bike_chk.get()

    def moto_bike_chk_setter(self):
        self.vid.dict_detection_activate_config['motoBike'] = self.motoBike_chk.get()

    def car_chk_setter(self):
        self.vid.dict_detection_activate_config['car'] = self.car_chk.get()

    def trunk_chk_setter(self):
        self.vid.dict_detection_activate_config['trunk'] = self.trunk_chk.get()

    def bus_chk_setter(self):
        self.vid.dict_detection_activate_config['bus'] = self.bus_chk.get()

    ##
    # Alarm thresh getters
    ##
    def person_thresh_getter(self):
        num = self.person_get_thresh.get()
        try:
            num = int(num)
        # ValueError is the type of error expected from this conversion
        except ValueError:
            # Display Error Window (Title, Prompt)
            showerror('Non-Int Error', 'Please enter an Integer Number')
        else:
            self.vid.dict_detection_activate_config['person_thresh'] = num

    def bike_thresh_getter(self):
        num = self.bike_get_thresh.get()
        try:
            num = int(num)
        # ValueError is the type of error expected from this conversion
        except ValueError:
            # Display Error Window (Title, Prompt)
            showerror('Non-Int Error', 'Please enter an Integer Number')
        else:
            self.vid.dict_detection_activate_config['bike_thresh'] = num

    def moto_bike_thresh_getter(self):
        num = self.motoBike_get_thresh.get()
        try:
            num = int(num)
        # ValueError is the type of error expected from this conversion
        except ValueError:
            # Display Error Window (Title, Prompt)
            showerror('Non-Int Error', 'Please enter an Integer Number')
        else:
            self.vid.dict_detection_activate_config['motoBike_thresh'] = num

    def car_thresh_getter(self):
        num = self.car_get_thresh.get()
        try:
            num = int(num)
        # ValueError is the type of error expected from this conversion
        except ValueError:
            # Display Error Window (Title, Prompt)
            showerror('Non-Int Error', 'Please enter an Integer Number')
        else:
            self.vid.dict_detection_activate_config['car_thresh'] = num

    def trunk_thresh_getter(self):
        num = self.trunk_get_thresh.get()
        try:
            num = int(num)
        # ValueError is the type of error expected from this conversion
        except ValueError:
            # Display Error Window (Title, Prompt)
            showerror('Non-Int Error', 'Please enter an Integer Number')
        else:
            self.vid.dict_detection_activate_config['trunk_thresh'] = num

    def bus_thresh_getter(self):
        num = self.bus_get_thresh.get()
        try:
            num = int(num)
        # ValueError is the type of error expected from this conversion
        except ValueError:
            # Display Error Window (Title, Prompt)
            showerror('Non-Int Error', 'Please enter an Integer Number')
        else:
            self.vid.dict_detection_activate_config['bus_thresh'] = num

    # def display_int(self):
    #     num = self.person_get_thresh.get()
    #     # if num is '':
    #     #     num = 0
    #     # Try convert a str to int
    #     # If unable eg. int('hello') or int('5.5')
    #     # then show an error.
    #     try:
    #         num = int(num)
    #     # ValueError is the type of error expected from this conversion
    #     except ValueError:
    #         # Display Error Window (Title, Prompt)
    #         showerror('Non-Int Error', 'Please enter an Integer Number')
    #     else:
    #         print(num)

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame

        if ret:
            # self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.photo = PIL.ImageTk.PhotoImage(image=frame)

            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            # self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)

    def display_snapshot(self, window=None):
        if window is None:
            window = self.window
        # Button that lets the user take a snapshot
        btn_snapshot = tk.Button(window, text="Snapshot", width=20, command=self.snapshot)
        btn_snapshot.place(x=30, y=50)
        ##

    def display_person(self, window=None):
        if window is None:
            window = self.window

        # approval button for submission num of persons
        persons_btn = tk.Button(window, text="عدد الأفراد", width=20, bg='blue',
                                command=self.person_thresh_getter)
        # self.persons_btn = tk.Button(window, text="عدد الأفراد", width=20, command=self.display_int)
        # self.persons_btn.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.CENTER, expand=True)
        persons_btn.place(x=self.x1_axis, y=(self.y_axis + self.y_offset))

        # Data Entry that lets the user enter num of 'الأفراد'
        self.person_get_thresh.place(x=self.x2_axis, y=(self.y_axis + self.y_offset))
        # Check point to display 'الأفراد'
        c1 = tk.Checkbutton(window, text="الأفراد", onvalue=1, variable=self.person_chk, offvalue=0,
                            bg='blue', command=self.person_chk_setter)
        c1.select()
        c1.place(x=self.x3_axis, y=(self.y_axis + self.y_offset))
        # self.c1.place(x=95, y=130)

    def display_bike(self, window=None):
        if window is None:
            window = self.window

        # approval button for submission num of bikes
        bike_btn = tk.Button(window, text="عدد دراجة", width=20, bg='gray',
                             command=self.bike_thresh_getter)
        bike_btn.place(x=self.x1_axis, y=(self.y_axis + 2 * self.y_offset))

        # Data Entry that lets the user enter num of 'دراجة'
        bike = tk.Entry(window)
        bike.place(x=self.x2_axis, y=(self.y_axis + 2 * self.y_offset))

        # Bikes display checkButton
        c2 = tk.Checkbutton(window, text="دراجة", onvalue=1, variable=self.bike_chk, offvalue=0,
                            bg='gray', command=self.bike_chk_setter)
        c2.select()
        c2.place(x=self.x3_axis, y=(self.y_axis + 2 * self.y_offset))

    def display_moto(self, window=None):
        if window is None:
            window = self.window

        # approval button for submission num of bikes
        moto_btn = tk.Button(window, text="عدد دراجة نارية", width=20, bg='orange',
                             command=self.moto_bike_thresh_getter)
        moto_btn.place(x=self.x1_axis, y=(self.y_axis + 3 * self.y_offset))
        # Data Entry that lets the user enter num of 'دراجة نارية'
        moto = tk.Entry(window)
        moto.place(x=self.x2_axis, y=(self.y_axis + 3 * self.y_offset))
        # Bikes display checkButton
        c3 = tk.Checkbutton(window, text="دراجة نارية", onvalue=1, variable=self.motoBike_chk, offvalue=0,
                            bg='orange', command=self.moto_bike_chk_setter)
        c3.select()
        c3.place(x=self.x3_axis, y=(self.y_axis + 3 * self.y_offset))

    def display_car(self, window=None):
        if window is None:
            window = self.window

        # approval button for submission num of bikes
        car_btn = tk.Button(window, text="عدد السيارات", width=20, bg='yellow',
                            command=self.car_thresh_getter)
        car_btn.place(x=self.x1_axis, y=(self.y_axis + 4 * self.y_offset))
        # Data Entry that lets the user enter num of 'سيارات'
        car = tk.Entry(window)
        car.place(x=self.x2_axis, y=(self.y_axis + 4 * self.y_offset))
        # Bikes display checkButton
        c4 = tk.Checkbutton(window, text="سيارات", onvalue=1, variable=self.car_chk, offvalue=0,
                            bg='yellow', command=self.car_chk_setter)
        c4.select()
        c4.place(x=self.x3_axis, y=(self.y_axis + 4 * self.y_offset))
        ##

    def display_truck(self, window=None):
        if window is None:
            window = self.window

        # approval button for submission num of bikes
        truck_btn = tk.Button(window, text="عدد الشاحنات", width=20, bg='cyan',
                              command=self.trunk_thresh_getter)
        truck_btn.place(x=self.x1_axis, y=(self.y_axis + 5 * self.y_offset))
        # Data Entry that lets the user enter num of 'شاحنات'
        truck = tk.Entry(window)
        truck.place(x=self.x2_axis, y=(self.y_axis + 5 * self.y_offset))
        # Bikes display checkButton
        c = tk.Checkbutton(window, text="شاحنات", onvalue=1, variable=self.trunk_chk, offvalue=0,
                           bg='cyan', command=self.trunk_chk_setter)
        c.select()
        c.place(x=self.x3_axis, y=(self.y_axis + 5 * self.y_offset))
        ##

    def display_bus(self, window=None):
        if window is None:
            window = self.window

        # approval button for submission num of bikes
        bus_btn = tk.Button(window, text="عدد الحافلات", width=20, bg='green',
                            command=self.bus_thresh_getter)
        bus_btn.place(x=self.x1_axis, y=(self.y_axis + 6 * self.y_offset))
        # Data Entry that lets the user enter num of 'حافلة'
        bus = tk.Entry(window)
        bus.place(x=self.x2_axis, y=(self.y_axis + 6 * self.y_offset))
        # Bikes display checkButton
        c = tk.Checkbutton(window, text="حافلات", onvalue=1, variable=self.bus_chk, offvalue=0,
                           bg='green', command=self.bus_chk_setter)
        c.select()
        c.place(x=self.x3_axis, y=(self.y_axis + 6 * self.y_offset))
        ##

    def display_alarm_chk_system(self, window=None):
        if window is None:
            window = self.window

        # Initiate Alarm again
        alarm_btn = tk.Button(window, text="Alarm", width=10, height=5, bg='red',
                              command=self.alarm_state_getter)
        alarm_btn.place(x=self.x1_axis + 50, y=(self.y_axis + 7 * self.y_offset))
        # Safe Button
        safe_btn = tk.Button(window, text="SAFE", width=10, height=5, bg='green',
                             command=self.safe_state_getter)
        safe_btn.place(x=self.x1_axis + 200, y=(self.y_axis + 7 * self.y_offset))

    def disable_event(self):
        pass


# cam_path = "rtsp://service:12345@192.168.0.11:554/video.h264"
# cam_path = "rtsp://service:12345@192.168.0.11:554"

# Create a window and pass it to the Application object
# App(tk.Tk(), "Detection System")
App(tk.Tk(), "Detection System", 'data/videos/traffic.mp4')
# App(tk.Tk(), "Detection System", 'data/videos/vid_2.mp4')
# App(tk.Tk(), "Detection System", cam_path)
