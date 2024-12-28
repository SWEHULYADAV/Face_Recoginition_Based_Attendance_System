import cv2
import os
from flask import Flask, request, render_template, Response
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)

# Constants
NIMGS = 20
DATETODAY = date.today().strftime("%m_%d_%y")
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure directories exist
for directory in ['Attendance', 'static', 'static/faces']:
    os.makedirs(directory, exist_ok=True)

if f'Attendance-{DATETODAY}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{DATETODAY}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Helper functions
def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return FACE_DETECTOR.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return []

def identify_face(facearray):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

def train_model():
    faces, labels = [], []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{DATETODAY}.csv')
        return df['Name'], df['Roll'], df['Time'], len(df)
    except Exception as e:
        print(f"Error reading attendance file: {e}")
        return [], [], [], 0

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    try:
        df = pd.read_csv(f'Attendance/Attendance-{DATETODAY}.csv')
        if int(userid) not in df['Roll'].values:
            with open(f'Attendance/Attendance-{DATETODAY}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
    except Exception as e:
        print(f"Error adding attendance: {e}")

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if identified_person:
                add_attendance(identified_person[0])
                cv2.putText(frame, identified_person[0], (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

################## ROUTING FUNCTIONS #########################

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), mess='No trained model found. Please add a new face to continue.')
    return render_template('home.html')

@app.route('/stop', methods=['GET'])
def stop():
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), mess='Attendance taking stopped.')

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{NIMGS}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == NIMGS * 5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:  # ESC key to stop
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

if __name__ == '__main__':
    app.run(debug=True)
