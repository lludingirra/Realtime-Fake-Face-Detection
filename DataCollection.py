import cv2 # Import OpenCV for video capture and image processing.
import cvzone # Import cvzone for helper functions.
from cvzone.FaceDetectionModule import FaceDetector # Import FaceDetector from cvzone for face detection.
from time import time # Import time for timestamp generation.

classID = 0 # Define the class ID for collected data: 0 for "fake" and 1 for "real".
outputFolderPath = 'Dataset/DataCollect' # Path to the folder where collected images and labels will be saved.
confidence = 0.8 # Minimum confidence score for a detected face to be considered valid for data collection.
save = True # Boolean flag to enable/disable saving of collected data.
blurThreshold = 35 # Threshold for Laplacian variance to check image blurriness. Faces with blur below this are considered blurry.

debug = False # Debug flag: if True, it will draw rectangles and text on the original 'img' as well.
offsetPercentageW = 10 # Percentage to expand the width of the bounding box.
offsetPercentageH = 20 # Percentage to expand the height of the bounding box.
floatingPoint = 6 # Number of decimal places for normalized bounding box coordinates in the label file.

cap = cv2.VideoCapture(0) # Initialize video capture from the default webcam (index 0).
camWidth, camHeight = 640, 480 # Define camera resolution.
cap.set(3, camWidth) # Set camera width.
cap.set(4, camHeight) # Set camera height.

detector = FaceDetector() # Initialize FaceDetector object.

# --- Main Data Collection Loop ---
while True:
    success, img = cap.read() # Read a frame from the webcam.
    imgOut = img.copy() # Create a copy of the original image to draw on for display.
    
    if not success:
        print("Camera frame could not be read. Exiting...") # Print error if frame reading fails.
        break # Exit the loop.
    
    # Find faces in the image. draw=False prevents the detector from drawing on 'img' directly.
    img, bboxs = detector.findFaces(img, draw=False)
    
    listBlur = [] # List to store blur status for each detected face (True if clear, False if blurry).
    listInfo = [] # List to store normalized bounding box info for each face.
    
    if bboxs: # If faces are detected:
        for bbox in bboxs: # Iterate through each detected bounding box.
            x,y,w,h = bbox["bbox"] # Get bounding box coordinates and dimensions.
            score = bbox["score"][0] # Get the confidence score of the detection.
            
            if score > confidence: # Filter faces based on confidence threshold.
                # Adjust bounding box to add offset (expand the box).
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3) # Offset Y more upwards, often for forehead.
                h = int(h + offsetH * 3.5) # Offset H more downwards.

                ih, iw, _ = img.shape # Get image dimensions (height, width).
                
                # Ensure bounding box coordinates stay within image boundaries.
                x = max(0, x) 
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                # Crop the face region from the original image.
                imgFace = img[y:y+h, x:x+w]
                
                # Check if the cropped face image is valid (not empty).
                if imgFace.shape[0] > 0 and imgFace.shape[1] > 0:
                    cv2.imshow('face', imgFace) # Display the cropped face (for debug/preview).
                    
                    # Calculate Laplacian variance to estimate blurriness.
                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                    
                    if blurValue > blurThreshold : # If blur value is above threshold, face is considered clear.
                        listBlur.append(True) 
                    else : # Otherwise, it's considered blurry.
                        listBlur.append(False)
                else:
                    blurValue = 0 # If cropped face is invalid, set blur to 0.
                    listBlur.append(False) # And mark as blurry/not usable.
                    
                # Calculate normalized bounding box coordinates (xc, yc, wn, hn) for YOLO format.
                xc, yc = x + w / 2, y + h / 2 # Center coordinates.
                
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                # Ensure normalized coordinates are within [0, 1] range.
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                # Append the formatted label string to listInfo.
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # Draw bounding box and display score/blur on the 'imgOut' (display) image.
                cv2.rectangle(imgOut, (x,y,w,h), (255,0,0), 3) # Blue rectangle.
                cvzone.putTextRect(imgOut, f'Score: {int(score*100)}% Blur: {blurValue}',
                                   (x,y-20), scale=2, thickness=3)
                
                # Debug drawing on the original 'img' if debug flag is True.
                if debug :
                    cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score*100)}% Blur: {blurValue}',
                                       (x,y-20), scale=2, thickness=3)
            
        # If saving is enabled AND all detected faces are not blurry AND at least one face was detected.
        if save:
            if all(listBlur) and listBlur != []:
                # Generate a unique timestamp for the image and label file names.
                timeNow = str(time()).replace('.', '') # Get current time and remove decimal.
                
                # Save the image.
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                
                # Save the label information to a .txt file.
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", "a")
                    f.write(info)
                    f.close()
                
    cv2.imshow("Image", imgOut) # Display the main output image.
    
    # Wait for 1 millisecond for a key press. If 'q' is pressed, exit.
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
        
# Release camera resources and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()