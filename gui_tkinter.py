import time
import tkinter as tk
from turtle import width
from tkinter.simpledialog import askstring, askinteger
from tkinter.messagebox import showerror

import PIL.Image
import PIL.ImageTk
import cv2

from gui_video_read_thread import MyVideoCapture


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
        self.top.geometry("400x600")
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height + 200)
        self.canvas.pack(side=tk.LEFT, anchor=tk.CENTER, expand=False)
        ##
        self.persons = tk.Entry(self.top)
        ##
        self.display_snapshot(self.top)
        self.display_person(self.top)
        self.display_bike(self.top)
        self.display_moto(self.top)
        self.display_cars(self.top)
        self.display_truck(self.top)
        self.display_bus(self.top)
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

    def print_chk1(self, chk1=0):
        if chk1 == 0:
            print("Check button OFF")
        else:
            print("Check button ON")

    def display_int(self, num=5):
        num = self.persons.get()
        # if num is '':
        #     num = 0
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
        btn_snapshot.pack(anchor=tk.CENTER, expand=False, pady=5)
        ##

    def display_person(self, window=None):
        if window is None:
            window = self.window

        # Data Entry that lets the user enter num of 'الأفراد'
        self.persons.pack(side=tk.TOP, anchor=tk.CENTER,
                          expand=False)  # expand = False to disable the buttons expansion
        # approval button for submission num of persons
        persons_btn = tk.Button(window, text="عدد الأفراد", width=20, height=2, bg='blue',
                                command=self.display_int)
        # self.persons_btn = tk.Button(window, text="عدد الأفراد", width=20, command=self.display_int)
        # self.persons_btn.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.CENTER, expand=True)
        persons_btn.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)

        # Check point to display 'الأفراد'
        var1 = tk.IntVar()
        c1 = tk.Checkbutton(window, text="الأفراد", onvalue=1, variable=var1, offvalue=0,
                            bg='blue', command=self.print_chk1)
        c1.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        # self.c1.place(x=95, y=130)

    def display_bike(self, window=None):
        if window is None:
            window = self.window

        # Data Entry that lets the user enter num of 'دراجة'
        bike = tk.Entry(window)
        bike.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)  # expand = False to disable the buttons expansion
        # approval button for submission num of bikes
        bike_btn = tk.Button(window, text="عدد دراجة", width=20, height=2, bg='gray',
                             command=dis_num_bike(bike.get()))
        bike_btn.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        # Bikes display checkButton
        var2 = tk.IntVar()
        c2 = tk.Checkbutton(window, text="دراجة", onvalue=1, variable=var2, offvalue=0,
                            bg='gray', command=self.print_chk1)
        c2.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)

    ##

    def display_moto(self, window=None):
        if window is None:
            window = self.window

        # Data Entry that lets the user enter num of 'دراجة نارية'
        moto = tk.Entry(window)
        moto.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)  # expand = False to disable the buttons expansion
        # approval button for submission num of bikes
        moto_btn = tk.Button(window, text="عدد دراجة نارية", width=20, height=2, bg='orange',
                             command=self.display_int)
        moto_btn.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        # Bikes display checkButton
        var3 = tk.IntVar()
        c3 = tk.Checkbutton(window, text="دراجة نارية", onvalue=1, variable=var3, offvalue=0,
                            bg='orange', command=self.print_chk1)
        c3.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)

    def display_cars(self, window=None):
        if window is None:
            window = self.window

        # Data Entry that lets the user enter num of 'سيارات'
        car = tk.Entry(window)
        car.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)  # expand = False to disable the buttons expansion
        # approval button for submission num of bikes
        car_btn = tk.Button(window, text="عدد السيارات", width=20, height=2, bg='yellow',
                            command=self.display_int)
        car_btn.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        # Bikes display checkButton
        var4 = tk.IntVar()
        c4 = tk.Checkbutton(window, text="سيارات", onvalue=1, variable=var4, offvalue=0,
                            bg='yellow', command=self.print_chk1)
        c4.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        ##

    def display_truck(self, window=None):
        if window is None:
            window = self.window

        # Data Entry that lets the user enter num of 'شاحنات'
        truck = tk.Entry(window)
        truck.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)  # expand = False to disable the buttons expansion
        # approval button for submission num of bikes
        truck_btn = tk.Button(window, text="عدد الشاحنات", width=20, height=2, bg='cyan',
                              command=self.display_int)
        truck_btn.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        # Bikes display checkButton
        var = tk.IntVar()
        c = tk.Checkbutton(window, text="شاحنات", onvalue=1, variable=var, offvalue=0,
                           bg='cyan', command=self.print_chk1)
        c.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        ##

    def display_bus(self, window=None):
        if window is None:
            window = self.window

        # Data Entry that lets the user enter num of 'حافلة'
        bus = tk.Entry(window)
        bus.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)  # expand = False to disable the buttons expansion
        # approval button for submission num of bikes
        bus_btn = tk.Button(window, text="عدد الحافلات", width=20, height=2, bg='green',
                            command=self.display_int)
        bus_btn.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        # Bikes display checkButton
        var = tk.IntVar()
        c = tk.Checkbutton(window, text="حافلات", onvalue=1, variable=var, offvalue=0,
                           bg='green', command=self.print_chk1)
        c.pack(side=tk.TOP, anchor=tk.CENTER, expand=False)
        ##

    def disable_event(self):
        pass


# Create a window and pass it to the Application object
App(tk.Tk(), "Detection System")
