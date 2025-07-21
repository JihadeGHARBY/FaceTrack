"""
Main_Code.py :: Real-Time Face Detection & Personalized Audio Greeting
====================================================================
This standalone script opens your default webcam, detects faces in real time,
compares them against a small local database of reference images, and—if a
match is confident enough—greets the recognised person by name through your
speakers.

How it works (high-level):
  1.   Loads two pre-trained *dlib* models:
         • `shape_predictor_68_face_landmarks.dat` – facial landmarks.
         • `dlib_face_recognition_resnet_model_v1.dat` – 128-D face embeddings.
  2.   Scans every image found in `../IMAGES_FOLDER/` (relative to this file)
       and builds an **in-memory vector database** of known face encodings plus
       their labels (the file name without extension).
  3.   If an audio file `<name>.mp3` is missing in `../AUDIO_FOLDER/`, it
       auto-generates one using Google-TTS so that each person has their own
       spoken greeting.
  4.   Starts capturing frames from a video source (default 0 = first webcam),
       detects faces, computes their encodings, and finds the closest match in
       the database. When the best distance is < 0.6 it is considered a match.
  5.   Draws a bounding box & similarity score on the frame **and** plays the
       corresponding audio file. A short cooldown prevents the same greeting
       from looping continuously whilst the face remains in view.

Folder layout (expected):
  your_project/
  ├─ Detection/
  │  ├─ Main_Code.py               # <— you are here
  │  └─ models/
  │     ├─ shape_predictor_68_face_landmarks.dat
  │     └─ dlib_face_recognition_resnet_model_v1.dat
  ├─ IMAGES_FOLDER/                # Reference faces (e.g. alice.jpg)
  └─ AUDIO_FOLDER/                 # Optional pre-recorded greetings

Quick start
-----------
1.  Install dependencies (ideally in a virtualenv):
        pip install opencv-python dlib pygame numpy gTTS
    On Windows you may need the Visual C++ build tools for *dlib*.
2.  Put at least one face image into `../IMAGES_FOLDER/`, e.g. `bob.jpg`.
3.  Run the script:
        python Main_Code.py
4.  Press **q** in the video window to quit.

Adjustments
-----------
•   Change `VIDEO_SOURCE` to a file path or stream URL to process a video file
    instead of the webcam.
•   Tweak `COOLDOWN_SECONDS` and the 0.6 distance threshold to make the system
    more or less strict.

Author: <your-name-here>
"""

# Install required packages if you haven't:
# pip install opencv-python dlib pygame numpy pandas gTTS

import cv2
import dlib
import numpy as np
import time
import threading
import os
import pygame


# --- CONFIGURATION ---
VIDEO_SOURCE = 0  # 0 for webcam

# Determine project root (one directory up from this script file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to the images and audio folders (renamed by the user)
IMAGES_FOLDER = os.path.join(PROJECT_ROOT, 'IMAGES_FOLDER')  # Folder where images are stored
COOLDOWN_SECONDS = 5
AUDIO_FOLDER = os.path.join(PROJECT_ROOT, 'AUDIO_FOLDER')

# --- INITIALIZATION ---
print("Loading known people...")

if not os.path.exists(IMAGES_FOLDER):
    raise FileNotFoundError(f"Images folder not found at {IMAGES_FOLDER}")

if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Initialize dlib's face detector and face recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")

# --- LOAD PEOPLE FROM IMAGES ---
# We now build the database of known faces by scanning every image file inside
# `people_dataset`. The base filename (without extension) is treated as the
# person's name and is also used to look up (or generate) a corresponding
# `.mp3` file inside `nicknames_audio`.

known_encodings = []  # List[np.ndarray] of 128-D face descriptors
known_names = []      # Parallel list of the person/name for each encoding

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

for file_name in os.listdir(IMAGES_FOLDER):
    # Ignore non-image files
    if not file_name.lower().endswith(VALID_EXTS):
        continue

    name = os.path.splitext(file_name)[0]            # label
    image_path = os.path.join(IMAGES_FOLDER, file_name)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector(rgb_img, 1)
    if len(detections) == 0:
        print(f"Warning: No face found in image {image_path}. Skipping.")
        continue

    shape = shape_predictor(rgb_img, detections[0])
    descriptor = np.array(face_rec_model.compute_face_descriptor(rgb_img, shape))

    known_encodings.append(descriptor)
    known_names.append(name)

    # Ensure an audio file exists for this individual
    audio_path = os.path.join(AUDIO_FOLDER, f"{name}.mp3")
    if not os.path.exists(audio_path):
        print(f"Warning: Greeting audio for {name} not found at {audio_path}. This person will be recognised silently.")

print(f"Loaded {len(known_names)} known people.")

# Initialize pygame mixer for audio
pygame.mixer.init()

cooldown = False

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Function to play sound asynchronously
def play_audio_alert(nickname):
    """Play the pre-generated `<nickname>.mp3` greeting on a background thread."""
    try:
        audio_path = os.path.join(AUDIO_FOLDER, f"{nickname}.mp3")
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Failed to play audio for {nickname}: {e}")

# Cooldown management
def start_cooldown():
    """Prevent the same name from being announced repeatedly for a few seconds."""
    global cooldown
    cooldown = True
    time.sleep(COOLDOWN_SECONDS)
    cooldown = False

# Start video capture
print("Starting video stream...")  # Notify user that webcam capture is about to begin
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

try:
    # Infinite processing loop – captures a frame, detects faces, and greets any matches
    while True:
        # Grab a single frame of video; `ret` is False if the capture failed (e.g., camera unplugged)
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Convert BGR (OpenCV default) to RGB for dlib compatibility
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces in the current frame (upsample once for better accuracy)
        detections = face_detector(rgb_frame, 1)

        # Iterate over every face found in this frame
        for det in detections:
            shape = shape_predictor(rgb_frame, det)
            # Compute a 128-D embedding (face descriptor) for the detected face
            face_descriptor = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))

            # Calculate Euclidean distance from this face to every known face in the database
            distances = [np.linalg.norm(face_descriptor - known_encoding) for known_encoding in known_encodings]
            if len(distances) == 0:
                continue

            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]  # Lowest distance → best match

            # Consider it a match only if the distance is below the empirical threshold of 0.6
            if min_distance < 0.6:
                detected_name = known_names[min_distance_idx]

                # Draw rectangle and nickname
                # Draw a green bounding box around the recognised face
                cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 2)
                # Write the person's name plus a similarity score above the rectangle
                cv2.putText(frame, f"{detected_name} ({1 - min_distance:.2f})", (det.left(), det.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Play sound if not in cooldown
                if not cooldown:
                    threading.Thread(target=play_audio_alert, args=(detected_name,)).start()
                    threading.Thread(target=start_cooldown).start()

        # Display the resulting frame
        # Show the annotated frame in a window titled "Video"
        cv2.imshow('Video', frame)

        # Press 'q' to quit
        # Exit cleanly when the user presses the letter q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release everything
    video_capture.release()  # Release the webcam hardware
    cv2.destroyAllWindows()
    print("Video stream stopped.")