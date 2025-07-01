import math # Import the math module for mathematical operations (e.g., ceil for rounding confidence).
import time # Import the time module for time-related functions (e.g., FPS calculation).

import cv2 # Import the OpenCV library for image and video processing.
import cvzone # Import the cvzone library for computer vision helper functions (e.g., drawing rectangles).
from ultralytics import YOLO # Import YOLO from ultralytics for object detection using YOLO models.

confidence = 0.6 # Set the minimum confidence threshold for a detection to be considered valid.

cap = cv2.VideoCapture(0) # Initialize video capture from the default webcam (camera index 0).
cap.set(3, 640) # Set the width of the captured video frame to 640 pixels.
cap.set(4, 480) # Set the height of the captured video frame to 480 pixels.

# Load the custom YOLO model.
# The path should point to your trained YOLO model file (e.g., 'l_version_1_300.pt').
# Make sure this path is correct on your system.
model = YOLO("model/l_version_1_300.pt")

# Define the class names that your model is trained to detect.
# In this case, "fake" and "real" faces.
classNames = ["fake", "real"]

prev_frame_time = 0 # Variable to store the timestamp of the previous frame, used for FPS calculation.
new_frame_time = 0 # Variable to store the timestamp of the current frame, used for FPS calculation.

# --- Main Loop for Real-time Detection ---
while True:
    new_frame_time = time.time() # Get the current time for FPS calculation.
    success, img = cap.read() # Read a frame from the webcam.

    # If the image could not be read successfully, break the loop.
    if not success:
        print("Could not read image from camera. Exiting...")
        break

    # Flip the image horizontally (around the Y-axis) for a mirror effect, which is common for webcam applications.
    # 1: Horizontal flip
    # 0: Vertical flip
    # -1: Both horizontal and vertical flip
    img_flipped = cv2.flip(img, 1) # All subsequent operations will be performed on the flipped image.

    # Run object detection on the flipped image.
    # stream=True processes the results as an iterator for efficiency.
    # verbose=False suppresses detailed output from the YOLO model.
    results = model(img_flipped, stream=True, verbose=False)
    
    # Process the detection results.
    for r in results:
        boxes = r.boxes # Get the detected bounding boxes.
        for box in boxes:
            # Get bounding box coordinates (x1, y1, x2, y2) and convert to integers.
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1 # Calculate width and height of the bounding box.

            # Get confidence score and round it to two decimal places.
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Get the class ID and convert to integer.
            cls = int(box.cls[0])

            # Filter detections by confidence threshold.
            if conf > confidence:
                # Set color based on detected class (real or fake).
                if classNames[cls] == 'real':
                    color = (0, 255, 0) # Green for 'real'
                else:
                    color = (0, 0, 255) # Red for 'fake'

                # Draw the bounding box and class name with confidence on the flipped image.
                cvzone.cornerRect(img_flipped, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img_flipped, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), # Position the text.
                                   scale=2, thickness=4, colorR=color, colorB=color)

    # Calculate and display FPS.
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # Add FPS information directly onto the image.
    cv2.putText(img_flipped, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the processed (flipped) image.
    cv2.imshow("Image", img_flipped)

    # Wait for 1 millisecond for a key press. If 'q' is pressed, break the loop.
    key = cv2.waitKey(1)
    if key == ord('q'): # Check if the 'q' key's ASCII value is pressed.
        break

# Release camera resources and close all OpenCV windows when the loop exits.
cap.release()
cv2.destroyAllWindows()