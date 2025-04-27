import cv2
import face_recognition
from PIL import Image
import numpy as np
import sys
import time

# --- Step 1: Load and convert your reference image ---
image_path = 'img.jpg'  # Change to your actual file name

try:
    pil_image = Image.open(image_path).convert('RGB')
    reference_image = np.array(pil_image)
    print(f"✅ Loaded and converted image: {image_path}")
except FileNotFoundError:
    print("❌ Image not found. Check the file name and path.")
    sys.exit()
except Exception as e:
    print(f"❌ Failed to load image: {e}")
    sys.exit()

# --- Step 2: Generate face encoding from image ---
try:
    known_encodings = face_recognition.face_encodings(reference_image)
    if not known_encodings:
        print("⚠️ No face detected in the reference image.")
        sys.exit()
    known_face_encoding = known_encodings[0]
    print("✅ Face encoding generated from reference image.")
except Exception as e:
    print(f"❌ Error during face encoding: {e}")
    sys.exit()

# --- Step 3: Initialize webcam ---
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("❌ Could not access the webcam.")
    sys.exit()

print("⌛ Starting webcam...")

face_matched = False
frame_count = 0
start_time = time.time()

# --- Main loop to process the webcam feed ---
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to read from camera.")
        break

    elapsed_time = time.time() - start_time
    frame_count += 1

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = []
    face_encodings = []

    # Only process every 3rd frame to reduce load
    if frame_count % 3 == 0:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        print(f"Detected {len(face_locations)} face(s).")  # Debugging line

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces([known_face_encoding], face_encoding)[0]

        # Scale back face locations to match original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if match:
            color = (0, 255, 0)  # Green for match
            print("✅ Face matched!")  # Debugging line
            face_matched = True
        else:
            color = (255, 0, 0)  # Blue for unknown
            print("❎ Face not matched!")

        # Draw box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Smooth display timing
    delay = 10 if face_locations else 50
    key = cv2.waitKey(delay) & 0xFF

    # Exit after 4 seconds if face matched, or if user quits
    if (face_matched and elapsed_time >= 3) or key == ord('q'):
        print("✅ Closing camera.")
        break

# --- Step 4: Cleanup ---
video_capture.release()
cv2.destroyAllWindows()
print("✅ Camera closed.")
