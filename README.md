# Face_Recoginition_Based_Attendance_System
Face_Recoginition_Based_Attendance_System .exe/WebApp
# Face Recognition Based Attendance System

This project is a face recognition-based attendance system that uses machine learning to identify faces and mark attendance. The system is built using Python, OpenCV, scikit-learn, and Flask for the web interface.

## Features

- **Face Detection and Recognition**: Uses OpenCV for face detection and a K-Nearest Neighbors (KNN) classifier for face recognition.
- **Attendance Management**: Automatically marks attendance when a recognized face is detected.
- **User Registration**: Allows new users to register their faces.
- **Dark Mode**: Toggle between light and dark themes.
- **Responsive Design**: Adapts to different screen sizes.

## Prerequisites

- Python 3.x
- Flask
- OpenCV
- scikit-learn
- pandas
- joblib
- PIL (Pillow)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/attendance-system.git
    cd attendance-system
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure the following directories exist**:
    - `Attendance`
    - `static`
    - `static/faces`

4. **Download the Haar Cascade for face detection**:
    - Download `haarcascade_frontalface_default.xml` from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in the project directory.

## Usage

### Running the Flask App

1. **Start the Flask app**:
    ```bash
    python app.py
    ```

2. **Open your web browser and go to**:
    ```
    http://127.0.0.1:5000/
    ```

### Using the Attendance System

1. **Home Page**:
    - Displays the current attendance records.
    - Provides options to start and stop attendance taking.
    - Allows new user registration.

2. **Start Attendance**:
    - Click the "Take Attendance" button to start the webcam and begin face detection.
    - The system will automatically mark attendance for recognized faces.

3. **Stop Attendance**:
    - Click the "Stop Attendance" button to stop the webcam and save the attendance records.

4. **Register New User**:
    - Enter the full name and ID number of the new user.
    - Click "Add New User" to capture images and train the model.

### Attendance Records

- Attendance records are saved in the `Attendance` directory with the filename format `Attendance-MM_DD_YY.csv`.
- Each record contains the columns: `Name`, `Roll`, and `Time`.

## Project Structure

```
attendance-system/
│
├── Attendance/                  # Directory to store attendance records
│   └── Attendance-MM_DD_YY.csv  # Example attendance file
│
├── static/                      # Static files directory
│   ├── faces/                   # Directory to store user face images
│   └── face_recognition_model.pkl  # Trained face recognition model
│
├── templates/                   # HTML templates directory
│   └── home.html                # Main HTML template
│
├── app.py                       # Flask application
├── attendance_app.py            # Tkinter application for local use
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── requirements.txt             # Python package requirements
└── README.md                    # Project README file
```

## Code Overview

### app.py

- **Flask App**: Defines the Flask application and routes.
- **Helper Functions**: Functions for face detection, face recognition, model training, and attendance management.
- **Routes**:
  - `/`: Home page displaying attendance records.
  - `/video_feed`: Video feed for face detection.
  - `/start`: Start attendance taking.
  - `/stop`: Stop attendance taking.
  - `/add`: Add a new user.

### attendance_app.py

- **Tkinter App**: Defines the Tkinter application for local use.
- **Classes**:
  - `ModernButton`: Custom button class with modern styling.
  - `AttendanceSystem`: Main class for the attendance system.
- **Methods**:
  - `apply_theme`: Apply light or dark theme.
  - `toggle_theme`: Toggle between light and dark themes.
  - `handle_resize`: Handle window resize for responsive design.
  - `create_tooltips`: Create tooltips for UI elements.
  - `show_loading`: Show loading message.
  - `hide_loading`: Hide loading message.
  - `update_status`: Update the total user count.
  - `start_attendance`: Start attendance taking.
  - `stop_attendance`: Stop attendance taking.
  - `update_video`: Update the video feed for face detection.
  - `add_user`: Add a new user and train the model.
  - `update_attendance_table`: Update the attendance table with records.
  - `on_closing`: Handle graceful shutdown.

### home.html

- **HTML Template**: Defines the structure and layout of the web interface.
- **Sections**:
  - Navbar: Contains the title and dark mode toggle button.
  - Attendance Management: Displays attendance records and provides options to start/stop attendance.
  - User Registration: Form to register new users.
  - Footer: Displays the copyright information.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- OpenCV for face detection.
- scikit-learn for machine learning.
- Flask for the web framework.
- Bootstrap for the responsive design.
