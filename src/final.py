import sys
import dlib
import os
import cv2
import numpy
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QHBoxLayout, QScrollArea, QCheckBox, QGroupBox, QDialog, QMainWindow, QMenuBar, QAction
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import numpy as np
import demo
import math
from pymediainfo import MediaInfo


### 
# Analyzing Image or Frame 
###

# Define the model paths
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

# Load the face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
dlib_face_landmark = dlib.shape_predictor(shape_predictor_path)

# Returns the attributes of the given image. 
def analyze_image(image_path):
    image = cv2.imread(image_path)
    pil_image = Image.open(image_path)
    detected = face_detector_with_pose(image)
    metadata = extract_metadata(image_path)
    headposes = demo.process_image(image)

    return {
        "Size": image.shape[:2],  # (height, width)
        "Mode": pil_image.mode,  # OpenCV reads images in BGR mode
        "NumberOfFaces": detected[0],
        "EyesDistance": int(detected[1]),
        "LeftEyeisClosed?": detected[3],
        "RightEyeisClosed?": detected[2],
        "IsMouthClosed?": detected[4],
        "Metadata": metadata,
        "HeadPose": headposes
    }

# Returns the attributes of the given frame.
def analyze_frame(frame):
    detected = face_detector_with_pose(frame)
    headposes = demo.process_image(frame)
    return {
        "NumberOfFaces": detected[0],
        "EyesDistance": int(detected[1]),
        "LeftEyeisClosed?": detected[3],
        "RightEyeisClosed?": detected[2],
        "IsMouthClosed?": detected[4],
        "HeadPose": headposes
    }


def extract_metadata(image_path):
    metadata = {}
    try:
        pil_image = Image.open(image_path)
        exif_data = pil_image._getexif()
        if not exif_data:
            return metadata 

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            metadata[tag] = value

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
    return metadata

# Function to calculate headpose, eyes and mouth attributes.
def face_detector_with_pose(image):
    face_number = 0
    eyes_distance = 0
    ear_left = None
    ear_right = None
    mouth_ear = None
    left_eye_closed = None
    right_eye_closed = None
    mouth_closed = None
    faces_array = face_detector(image, 1)
    face_number = len(faces_array)
    if face_number > 0:
        face = faces_array[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_landmarks = dlib_face_landmark(gray, face)
        right_eye_tuples = []
        for i in range(42, 48):
            x = face_landmarks.part(i).x
            y = face_landmarks.part(i).y
            right_eye_tuples.append((x, y))
        left_eye_tuples = []
        for i in range(36, 42):
            x = face_landmarks.part(i).x
            y = face_landmarks.part(i).y
            left_eye_tuples.append((x, y))

        # Calculating EAR
        # for right_eye:
        rd1 = distance_calculator(right_eye_tuples[1], right_eye_tuples[5])
        rd2 = distance_calculator(right_eye_tuples[2], right_eye_tuples[4]) 
        rd3 = distance_calculator(right_eye_tuples[0], right_eye_tuples[3])

        ear_right = (rd1 + rd2) / (2 * rd3)

        # for left_eye
        ld1 = distance_calculator(left_eye_tuples[1], left_eye_tuples[5])
        ld2 = distance_calculator(left_eye_tuples[2], left_eye_tuples[4])
        ld3 = distance_calculator(left_eye_tuples[0], left_eye_tuples[3])

        ear_left = (ld1 + ld2) / (2 * ld3)

        # for mouth
        md1 = distance_calculator((face_landmarks.part(50).x, face_landmarks.part(50).y), (face_landmarks.part(58).x, face_landmarks.part(58).y))
        md2 = distance_calculator((face_landmarks.part(52).x, face_landmarks.part(52).y), (face_landmarks.part(56).x, face_landmarks.part(56).y))
        md3 = distance_calculator((face_landmarks.part(48).x, face_landmarks.part(48).y), (face_landmarks.part(54).x, face_landmarks.part(54).y))

        mouth_ear = (md1 + md2) / (2 * md3)

        # Calculating the mid point by considering two edge points
        l_leftmost_x = min(tup[0] for tup in left_eye_tuples)
        l_rightmost_x = max(tup[0] for tup in left_eye_tuples)

        l_up_y = min(tup[1] for tup in left_eye_tuples)
        l_bottom_y = max(tup[1] for tup in left_eye_tuples)

        r_leftmost_x = min(tup[0] for tup in right_eye_tuples)
        r_rightmost_x = max(tup[0] for tup in right_eye_tuples)

        r_up_y = min(tup[1] for tup in right_eye_tuples)
        r_bottom_y = max(tup[1] for tup in right_eye_tuples)

        l_mid_point_v2 = ((l_leftmost_x + l_rightmost_x) // 2, (l_up_y + l_bottom_y) // 2)
        r_mid_point_v2 = ((r_leftmost_x + r_rightmost_x) // 2, (r_up_y + r_bottom_y) // 2)

        eyes_distance = distance_calculator(l_mid_point_v2, r_mid_point_v2)

    if (ear_left is not None) and (ear_right is not None):
        if ear_left < 0.21:
            left_eye_closed = True
        else:
            left_eye_closed = False

        if ear_right < 0.21:
            right_eye_closed = True
        else:
            right_eye_closed = False

    if mouth_ear is not None:
        if mouth_ear < 0.4:
            mouth_closed = True
        else:
            mouth_closed = False

    return (face_number, eyes_distance, left_eye_closed, right_eye_closed, mouth_closed)

# Helper function to calculate the euqlidian distance between two points
def distance_calculator(tuple1, tuple2):
    return math.sqrt((tuple1[0] - tuple2[0])**2 + (tuple1[1] - tuple2[1])**2)

# The first method is removed since it included ffmpeg which is a more complex library requiring a local installation.
def extract_video_metadata2(file_path):
    media_info = MediaInfo.parse(file_path)
    format_info = next((track for track in media_info.tracks if track.track_type == "General"), {})
    video_stream = next((track for track in media_info.tracks if track.track_type == "Video"), {})
    
    return {
        "Format": format_info.format,
        "Duration": format_info.duration,
        "Size": format_info.file_size,
        "Bit_rate": format_info.overall_bit_rate,
        "Creation_time": format_info.recorded_date,
        "Location": None,  # MediaInfo does not provide location information directly
        "Device": video_stream.encoded_library,
        "Codec": video_stream.codec_id
    }


properties = {
        "Size": True,
        "Mode": True,
        "NumberOfFaces": True,
        "EyesDistance": True,
        "LeftEyeisClosed?": True,
        "RightEyeisClosed?": True,
        "IsMouthClosed?": True,
        "Metadata": True,
        "HeadPose": True,
        "Video_metadata": True
    }


### 
# Graphical User Interface Part 
###


class MediaAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.frame_count = 0
        self.video_capture = None
        self.files = []
        self.current_file_index = -1

    def initUI(self):
        self.setWindowTitle('Media Analyzer')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.main_layout = QVBoxLayout()

        # Buttons layout
        self.buttons_layout = QHBoxLayout()

        self.button_image = QPushButton('Upload Folder', self)
        self.button_image.clicked.connect(self.upload_folder)
        self.buttons_layout.addWidget(self.button_image)

        self.button_left = QPushButton('<', self)
        self.button_left.clicked.connect(self.show_previous_file)
        self.buttons_layout.addWidget(self.button_left)

        self.button_right = QPushButton('>', self)
        self.button_right.clicked.connect(self.show_next_file)
        self.buttons_layout.addWidget(self.button_right)

        self.main_layout.addLayout(self.buttons_layout)

        # Label for displaying images or videos
        self.image_container_layout = QVBoxLayout()
        self.image_container_layout.addStretch()

        self.label = QLabel(self)
        self.label.setFixedSize(400, 300)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.image_container_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.image_container_layout.addStretch()
        self.main_layout.addLayout(self.image_container_layout)

        # Label for displaying metadata
        self.metadata_scroll_area = QScrollArea()
        self.metadata_scroll_area.setWidgetResizable(True)
        self.metadata_label = QLabel(self)
        self.metadata_scroll_area.setWidget(self.metadata_label)
        
        self.other_scroll_area = QScrollArea()
        self.other_scroll_area.setWidgetResizable(True)
        self.other_label = QLabel(self)
        self.other_scroll_area.setWidget(self.other_label)

        self.properties_layout = QHBoxLayout()
        self.properties_layout.addWidget(self.other_scroll_area)
        self.properties_layout.addWidget(self.metadata_scroll_area)

        self.main_layout.addLayout(self.properties_layout)

        # Set the layout
        self.central_widget.setLayout(self.main_layout)

        # Create menu bar
        menubar = self.menuBar()
        properties_menu = menubar.addMenu('Properties')

        self.property_actions = {}
        for property_name in properties:
            action = QAction(property_name, self, checkable=True)
            action.setChecked(True)
            action.triggered.connect(self.update_metadata)
            properties_menu.addAction(action)
            self.property_actions[property_name] = action


    def update_metadata(self):
        if self.current_file_index >= 0:
            self.show_file(self.current_file_index)

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            self.files.sort()
            if self.files:
                self.current_file_index = 0
                self.show_file(self.current_file_index)

    def show_file(self, index):
        file_path = self.files[index+1]
        print(file_path)
        self.label.clear()
        self.metadata_label.clear()
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): # Accepted image formats
            self.display_image(file_path)
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')): # Accepted video formats
            self.display_video(file_path)

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap)
        analysis_result = analyze_image(image_path)
        self.show_data(analysis_result)

    def display_video(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path) 
        self.timer.start(int(1000 / self.video_capture.get(cv2.CAP_PROP_FPS))) # The timer which is connected to process_frame function starts.
        self.video_metadata=extract_video_metadata2(video_path)
       

    def process_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.video_capture.release()
            return
        else:
            self.frame_count+=1
            if(self.frame_count % int(self.video_capture.get(cv2.CAP_PROP_FPS)) == 0): # A frame is processed each second.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                analysis_result = analyze_frame(frame)
                analysis_result['Video_metadata'] = self.video_metadata
                self.show_data(analysis_result)
               
                # Put frame count on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                position = (10, 50)  # position (bottom-left corner of the text string)
                font_scale = 1.5
                font_color = (255, 0, 0)  # red color
                line_type = 2

                cv2.putText(frame, f'Frame: {self.frame_count}', position, font, font_scale, font_color, line_type)


                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(q_image))
                
               
    # Sets the text labels for metadata and other results.
    def show_data(self, data):
   
        selected_properties = {
            key: value for key, value in data.items() 
            if self.property_actions[key].isChecked()
        }

        other_items = {k: v for k, v in selected_properties.items() if k not in ["Video_metadata", "Metadata"]}
        metadata_items = {k: v for k, v in selected_properties.items() if k in ["Video_metadata", "Metadata"]}

        def flatten_dict(d, parent_key='', sep=': '):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{k}"
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key + sep))
                else:
                    items.append(f"{new_key}{sep}{v}")
            return items


        metadata_text = "\n".join(flatten_dict(metadata_items))
        self.metadata_label.setText(metadata_text)

        others_text = "\n".join(flatten_dict(other_items))
        self.other_label.setText(others_text)

    def show_previous_file(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.stop_video_capture()  
            self.show_file(self.current_file_index)

    def show_next_file(self):
        if self.current_file_index < len(self.files) - 2:
            self.current_file_index += 1
            self.stop_video_capture()  
            self.show_file(self.current_file_index)

    # Resets the variables for the next possible video.
    def stop_video_capture(self):
        if self.video_capture is not None:
            self.timer.stop()
            self.video_capture.release()
            self.frame_count = 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MediaAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())
