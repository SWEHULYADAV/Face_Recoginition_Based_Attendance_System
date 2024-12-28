import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from PIL import Image, ImageTk
import threading

# Constants
NIMGS = 10
DATETODAY = date.today().strftime("%m_%d_%y")
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure directories exist
for directory in ['Attendance', 'static', 'static/faces']:
    os.makedirs(directory, exist_ok=True)

if f'Attendance-{DATETODAY}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{DATETODAY}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Add new constants for styling
LIGHT_THEME = {
    'bg': '#f0f2f5',
    'fg': '#2d3436',
    'button_bg': '#0984e3',
    'button_fg': 'white',
    'frame_bg': 'white',
    'table_bg': '#dfe6e9',
    'table_fg': '#2d3436',
    'header_bg': '#74b9ff'
}

DARK_THEME = {
    'bg': '#2d3436',
    'fg': '#dfe6e9',
    'button_bg': '#00a8ff',
    'button_fg': 'white',
    'frame_bg': '#2d3436',
    'table_bg': '#353b48',
    'table_fg': '#dfe6e9',
    'header_bg': '#0984e3'
}

class ModernButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        style = ttk.Style()
        style.configure('Modern.TButton', 
                       padding=10, 
                       font=('Helvetica', 12, 'bold'),
                       borderwidth=0,
                       relief='flat')
        super().__init__(master, style='Modern.TButton', **kwargs)

class AttendanceSystem:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition Attendance System")
        self.window.state('zoomed')
        self.is_dark_mode = False
        
        # Configure theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Apply initial light theme
        self.apply_theme(LIGHT_THEME)
        
        # Add theme toggle button
        self.theme_btn = ModernButton(
            window,
            text="Toggle Dark Mode",
            command=self.toggle_theme
        )
        self.theme_btn.pack(pady=10)
        
        # Main container with padding
        self.main_container = ttk.Frame(window, padding="20")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Header with modern design
        header = ttk.Label(
            self.main_container,
            text="Face Recognition Attendance System",
            style='Heading.TLabel'
        )
        header.pack(pady=(0, 20))

        # Content frame with two columns
        content = ttk.Frame(self.main_container)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Attendance
        self.attendance_frame = ttk.LabelFrame(
            content,
            text="Attendance Management",
            padding="10"
        )
        self.attendance_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video frame with border - Update the layout
        self.video_frame = ttk.Frame(
            self.attendance_frame,
            relief='solid',
            borderwidth=1
        )
        self.video_frame.pack(fill=None, expand=False, pady=(0, 10))
        
        # Add a fixed size container for video
        video_container = ttk.Frame(self.video_frame, width=400, height=300)
        video_container.pack(padx=10, pady=10)
        video_container.pack_propagate(False)  # Prevent size changes
        
        self.video_label = ttk.Label(video_container)
        self.video_label.pack(expand=True, fill='both')

        # Add a label to show face detection status
        self.status_message = ttk.Label(self.video_frame, text="", font=('Helvetica', 12))
        self.status_message.pack(pady=10)

        # Control buttons - Move above video frame
        btn_frame = ttk.Frame(self.attendance_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ModernButton(
            btn_frame,
            text="Start Attendance",
            command=self.start_attendance
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ModernButton(
            btn_frame,
            text="Stop Attendance",
            command=self.stop_attendance
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Table with scrollbar - Update the layout and add a summary label
        table_frame = ttk.Frame(self.attendance_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            table_frame,
            columns=("Name", "ID", "Time"),
            show='headings',
            height=10
        )
        
        # Configure table columns
        self.tree.heading("Name", text="Name")
        self.tree.heading("ID", text="ID")
        self.tree.heading("Time", text="Time")
        self.tree.column("Name", width=150)
        self.tree.column("ID", width=100)
        self.tree.column("Time", width=150)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add a summary label below the table
        self.summary_label = ttk.Label(self.attendance_frame, text="", font=('Helvetica', 12))
        self.summary_label.pack(pady=10)

        # Right column - User Registration
        self.register_frame = ttk.LabelFrame(
            content,
            text="New User Registration",
            padding="10"
        )
        self.register_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Registration form
        form_frame = ttk.Frame(self.register_frame)
        form_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(form_frame, text="Full Name:").pack(anchor=tk.W, pady=(0, 5))
        self.name_entry = ttk.Entry(form_frame, width=30)
        self.name_entry.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(form_frame, text="ID Number:").pack(anchor=tk.W, pady=(0, 5))
        self.id_entry = ttk.Entry(form_frame, width=30)
        self.id_entry.pack(fill=tk.X, pady=(0, 20))

        self.register_btn = ModernButton(
            form_frame,
            text="Register New User",
            command=self.add_user
        )
        self.register_btn.pack(pady=10)

        self.status_label = ttk.Label(
            form_frame,
            text="Total Users: 0",
            font=('Helvetica', 10)
        )
        self.status_label.pack(pady=10)

        # Initialize
        self.is_taking_attendance = False
        self.cap = None
        self.update_status()
        self.create_tooltips()

    def apply_theme(self, theme):
        self.style.configure('TFrame', background=theme['bg'])
        self.style.configure('TLabel', 
                           background=theme['bg'],
                           foreground=theme['fg'],
                           font=('Helvetica', 11))
        self.style.configure('Heading.TLabel',
                           background=theme['bg'],
                           foreground=theme['fg'],
                           font=('Helvetica', 24, 'bold'))
        self.style.configure('Modern.TButton',
                           background=theme['button_bg'],
                           foreground=theme['button_fg'])
        self.style.configure('Treeview',
                           background=theme['table_bg'],
                           foreground=theme['table_fg'],
                           fieldbackground=theme['table_bg'])
        self.style.configure('Treeview.Heading',
                           background=theme['header_bg'],
                           foreground=theme['fg'],
                           font=('Helvetica', 10, 'bold'))
        
        # Update window background
        self.window.configure(bg=theme['bg'])
        
        # Update all frames
        for widget in self.window.winfo_children():
            if isinstance(widget, ttk.Frame):
                widget.configure(style='TFrame')

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme(DARK_THEME if self.is_dark_mode else LIGHT_THEME)

    # Add responsive design handling
    def handle_resize(self, event=None):
        width = self.window.winfo_width()
        if width < 800:  # Mobile view
            self.attendance_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.register_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:  # Desktop view
            self.attendance_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.register_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def create_tooltips(self):
        # Add tooltips to buttons and inputs
        self.add_tooltip(self.start_btn, "Start taking attendance")
        self.add_tooltip(self.stop_btn, "Stop taking attendance")
        self.add_tooltip(self.name_entry, "Enter the full name of the new user")
        self.add_tooltip(self.id_entry, "Enter a unique ID number")

    def add_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            
        widget.bind('<Enter>', show_tooltip)

    def show_loading(self, message="Processing..."):
        self.loading = ttk.Label(
            self.window,
            text=message,
            font=('Helvetica', 14),
            foreground=DARK_THEME['button_bg'] if self.is_dark_mode else LIGHT_THEME['button_bg']
        )
        self.loading.pack(pady=20)
        self.window.update()

    def hide_loading(self):
        if hasattr(self, 'loading'):
            self.loading.destroy()

    def update_status(self):
        count = totalreg()
        self.status_label.config(text=f"Total Users: {count}")

    def start_attendance(self):
        try:
            if not os.path.exists('static/face_recognition_model.pkl'):
                messagebox.showerror("Error", "Please register at least one user first")
                return

            self.is_taking_attendance = True
            self.cap = cv2.VideoCapture(0)
            self.update_video()
            self.start_btn.state(['disabled'])
            self.stop_btn.state(['!disabled'])

        except Exception as e:
            messagebox.showerror("Error", f"Could not start attendance: {str(e)}")
            self.stop_attendance()

    def stop_attendance(self):
        self.is_taking_attendance = False
        if self.cap is not None:
            self.cap.release()
        self.video_label.config(image='')
        self.status_message.config(text="")
        self.update_attendance_table()
        self.start_btn.state(['!disabled'])
        self.stop_btn.state(['disabled'])

    def update_video(self):
        if not self.is_taking_attendance:
            return

        ret, frame = self.cap.read()
        if ret:
            # Resize frame to fit the container
            frame = cv2.resize(frame, (400, 300))
            faces = extract_faces(frame)
            if len(faces) > 0:
                self.status_message.config(text="Face detected", foreground="green")
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                    face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                    identified_person = identify_face(face.reshape(1, -1))
                    if identified_person:
                        add_attendance(identified_person[0])
                        cv2.putText(frame, identified_person[0], (x+5, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                self.status_message.config(text="No face detected. Please look at the camera.", foreground="red")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_video)

    def add_user(self):
        try:
            newusername = self.name_entry.get()
            newuserid = self.id_entry.get()

            if not newusername or not newuserid:
                messagebox.showerror("Error", "Please enter both name and ID")
                return

            userimagefolder = f'static/faces/{newusername}_{newuserid}'
            os.makedirs(userimagefolder, exist_ok=True)

            cap = cv2.VideoCapture(0)
            i, j = 0, 0

            self.show_loading("Capturing images... Please look at the camera")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow('Adding new User', frame)
                faces = extract_faces(frame)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                    if j % 5 == 0:
                        name = f'{newusername}_{i}.jpg'
                        cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                        i += 1
                    j += 1
                        
                    if i == NIMGS:
                        break

                if i == NIMGS or cv2.waitKey(1) == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
            
            if train_model():
                self.update_status()
                messagebox.showinfo("Success", "User added successfully!")
            else:
                # Clean up if training failed
                if os.path.exists(userimagefolder):
                    import shutil
                    shutil.rmtree(userimagefolder)
        except Exception as e:
            messagebox.showerror("Error", f"Error adding user: {str(e)}")
        finally:
            self.hide_loading()

    def update_attendance_table(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add new items
        names, rolls, times, count = extract_attendance()
        for name, roll, time in zip(names, rolls, times):
            self.tree.insert('', 'end', values=(name, roll, time))

        # Update summary label
        self.summary_label.config(text=f"Total Attendance Today: {count}")

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.window.destroy()

if __name__ == '__main__':
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
        try:
            faces, labels = [], []
            for user in os.listdir('static/faces'):
                for imgname in os.listdir(f'static/faces/{user}'):
                    img = cv2.imread(f'static/faces/{user}/{imgname}')
                    if img is not None:
                        resized_face = cv2.resize(img, (50, 50))
                        faces.append(resized_face.ravel())
                        labels.append(user)
            if not faces:
                messagebox.showerror("Error", "No faces found for training")
                return False
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(np.array(faces), labels)
            joblib.dump(knn, 'static/face_recognition_model.pkl')
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
            return False

    def extract_attendance():
        try:
            df = pd.read_csv(f'Attendance/Attendance-{DATETODAY}.csv')
            return df['Name'].tolist(), df['Roll'].tolist(), df['Time'].tolist(), len(df)
        except Exception as e:
            print(f"Error reading attendance file: {e}")
            return [], [], [], 0

    def add_attendance(name):
        try:
            username, userid = name.split('_')
            current_time = datetime.now().strftime("%H:%M:%S")
            df = pd.read_csv(f'Attendance/Attendance-{DATETODAY}.csv')
            if int(userid) not in df['Roll'].values:
                with open(f'Attendance/Attendance-{DATETODAY}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time}')
        except Exception as e:
            print(f"Error adding attendance: {str(e)}")

    try:
        root = tk.Tk()
        
        # Set minimum window size
        root.minsize(600, 400)
        
        # Center window on screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - 1000) // 2
        y = (screen_height - 800) // 2
        root.geometry(f'1000x800+{x}+{y}')
        
        # Create app instance first
        app = AttendanceSystem(root)
        
        # Then bind resize event
        root.bind('<Configure>', lambda e: app.handle_resize())
        
        # Handle graceful shutdown
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application crashed: {str(e)}")
