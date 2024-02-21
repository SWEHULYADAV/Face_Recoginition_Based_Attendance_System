import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import os
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

def take_attendance(option):
    # Function to take attendance using camera or sample images
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    today_date = date.today().strftime("%m_%d_%y")

    if not os.path.isdir('Attendance'):
        os.makedirs('Attendance')

    if not os.path.isdir('static'):
        os.makedirs('static')

    if not os.path.isdir('static/faces'):
        os.makedirs('static/faces')

    if f'Attendance-{today_date}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/Attendance-{today_date}.csv', 'w') as f:
            f.write('Name,Roll,Time')

    def total_registered_users():
        return len(os.listdir('static/faces'))

    def extract_faces(img):
        if img is None:
            return []
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
            return face_points

    def identify_face(facearray):
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)

    def extract_attendance():
        df = pd.read_csv(f'Attendance/Attendance-{today_date}.csv')
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
        return names, rolls, times, l

    def add_attendance(name):
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")

        df = pd.read_csv(f'Attendance/Attendance-{today_date}.csv')
        if int(userid) not in list(df['Roll']):
            with open(f'Attendance/Attendance-{today_date}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')

    if option == 'camera':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if len(extract_faces(frame)) > 0:
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    elif option == 'samples':
        # Take attendance using sample images
        root = tk.Tk()
        root.withdraw()
        root.filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                                   filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        img = cv2.imread(root.filename)
        if img is not None:
            if len(extract_faces(img)) > 0:
                (x, y, w, h) = extract_faces(img)[0]
                cv2.rectangle(img, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(img, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(img[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(img, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Attendance', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            messagebox.showerror("Error", "Failed to open image file.")

# Create the main application window
app = tk.Tk()
app.title("Attendance System")

# Set window size and position
window_width = 600
window_height = 400
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
x_coordinate = (screen_width/2) - (window_width/2)
y_coordinate = (screen_height/2) - (window_height/2)
app.geometry(f'{window_width}x{window_height}+{int(x_coordinate)}+{int(y_coordinate)}')

# Set background color
app.config(bg="#f0f0f0")

# Add widgets (labels, entry, button) to the window
title_label = tk.Label(app, text="Attendance System", font=("Helvetica", 24), bg="#f0f0f0")
label_camera = tk.Label(app, text="Take Attendance with Camera", font=("Helvetica", 16), bg="#f0f0f0")
button_camera = tk.Button(app, text="Start", font=("Helvetica", 14), command=lambda: take_attendance('camera'))
label_samples = tk.Label(app, text="Take Attendance with Sample Images", font=("Helvetica", 16), bg="#f0f0f0")
button_samples = tk.Button(app, text="Start", font=("Helvetica", 14), command=lambda: take_attendance('samples'))

# Arrange the widgets using pack layout
title_label.pack(pady=20)
label_camera.pack(pady=10)
button_camera.pack(pady=5)
label_samples.pack(pady=10)
button_samples.pack(pady=5)

# Start the application
app.mainloop()
