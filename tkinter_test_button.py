import tkinter as tk
from tkinter.simpledialog import askstring, askinteger
from tkinter.messagebox import showerror


def display_1():
    # .get is used to obtain the current value
    # of entry_1 widget (This is always a string)
    print(entry_1.get())


def display_2():
    num = entry_2.get()
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


def display_3():
    # Ask String Window (Title, Prompt)
    # Returned value is a string
    ans = askstring('Enter String', 'Please enter any set of characters')
    # If the user clicks cancel, None is returned
    # .strip is used to ensure the user doesn't
    # enter only spaces ' '
    if ans is not None and ans.strip():
        print(ans)
    elif ans is not None:
        showerror('Invalid String', 'You must enter something')


def display_4():
    # Ask Integer Window (Title, Prompt)
    # Returned value is an int
    ans = askinteger('Enter Integer', 'Please enter an integer')
    # If the user clicks cancel, None is returned
    if ans is not None:
        print(ans)


# Create the main window
root = tk.Tk()

# Create the widgets
entry_1 = tk.Entry(root)
btn_1 = tk.Button(root, text="Display Text", command=display_1)

entry_2 = tk.Entry(root)
btn_2 = tk.Button(root, text="Display Integer", command=display_2)

btn_3 = tk.Button(root, text="Enter String", command=display_3)
btn_4 = tk.Button(root, text="Enter Integer", command=display_4)

# Grid is used to add the widgets to root
# Alternatives are Pack and Place
# canvas = tk.Canvas()
# canvas.pack()
# entry_1.pack(anchor=tk.CENTER,expand=True)

entry_1.grid(row=0, column=0)

btn_1.grid(row=1, column=0)

entry_2.grid(row=0, column=1)
btn_2.grid(row=1, column=1)

btn_3.grid(row=2, column=0)
btn_4.grid(row=2, column=1)

root.mainloop()
