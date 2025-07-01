# Fake Face Detection (Anti-Spoofing)

This comprehensive project implements a real-time fake face detection (anti-spoofing) system using custom-trained YOLOv8 models. It's designed to distinguish between real human faces and fake attempts (e.g., photos, videos) presented to a camera. The project includes modules for data collection, dataset splitting, model training, and real-time inference.

## Key Features

* **Real-time Inference:** Detects "real" or "fake" faces in real-time from a webcam feed.
* **Custom YOLOv8 Model:** Utilizes a custom-trained YOLOv8 model for high accuracy.
* **Data Collection Tool (`DataCollection.py`):**
    * Automated face detection and cropping.
    * Blur detection to ensure high-quality data.
    * Saves images and YOLO-formatted labels (.txt files) for training.
    * Configurable offsets for bounding box expansion.
* **Dataset Splitter (`splitData.py`):**
    * Automatically splits your collected dataset into training, validation, and testing sets (e.g., 70/20/10 ratio).
    * Generates the `data.yaml` configuration file required by YOLO for training.
* **Model Training Script (`FaceDetectorTest.py`):**
    * Facilitates training of the YOLOv8 model using your prepared dataset.
* **Live Detection (`main.py`):**
    * Processes webcam feed, flips it for user-friendly display.
    * Performs inference with the custom model.
    * Draws bounding boxes and labels ("REAL" or "FAKE") with confidence scores.
    * Displays real-time FPS.

## Prerequisites

* Python (3.8 or higher recommended).
* A webcam connected to your computer.
* **Crucially, a powerful enough GPU** is highly recommended for efficient YOLO model training. CPU training can be very slow.
* A pre-trained YOLOv8 model checkpoint (e.g., `yolov8n.pt` for initial training, or your custom `.pt` file).

## Installation

1.  **Clone or Download the Repository:**
    Get all project files to your local machine.

2.  **Install Required Libraries:**
    Open your terminal or command prompt and run:
    ```bash
    pip install opencv-python ultralytics cvzone numpy
    ```
    * **Note for Ultralytics:** Ensure you have the correct `ultralytics` version. Sometimes, specific `torch` or `cuda` installations are needed for optimal GPU performance. Refer to the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/quickstart/) for detailed installation instructions, especially for GPU setup.

## Step-by-Step Usage Guide

To get this project fully operational, you will follow these steps:

### Step 1: Data Collection (`DataCollection.py`)

This script helps you collect a dataset of "real" and "fake" faces. High-quality and diverse data is crucial for good model performance.

1.  **Prepare a folder:** Ensure the `Dataset/DataCollect` directory exists (it will be created if not).
2.  **Set `classID`:** In `DataCollection.py`, set `classID = 1` for "real" faces and `classID = 0` for "fake" faces.
    * Start with `classID = 1`: Collect images of yourself and others looking directly at the camera.
    * Change to `classID = 0`: Collect images of "fake" faces (e.g., hold up a photo of someone's face, play a video of a face on a screen).
3.  **Run `DataCollection.py`:**
    ```bash
    python DataCollection.py
    ```
4.  **Collect Data:**
    * A window will open showing your webcam feed.
    * Ensure your face (or the fake face) is clearly visible.
    * The script will automatically save images and generate corresponding `.txt` label files in `Dataset/DataCollect` when a clear face is detected (based on `blurThreshold`).
    * Collect a significant number of samples for both "real" and "fake" classes. More diverse data (different angles, lighting, people, fake methods) will lead to a more robust model.
    * Press `q` to stop collecting.
5.  **Move Data:** After collection, move all `.jpg` and `.txt` files from `Dataset/DataCollect` into `Dataset/All`.

### Step 2: Split Dataset (`splitData.py`)

This script organizes your collected data into `train`, `val`, and `test` sets, which are essential for training and evaluating machine learning models.

1.  **Ensure data is in `Dataset/All`:** Make sure all your collected `.jpg` and `.txt` files are in the `Dataset/All` folder.
2.  **Review `splitRatio`:** In `splitData.py`, you can adjust `splitRatio = {"train" : 0.7, "val" : 0.2, "test" : 0.1}` if you prefer different proportions.
3.  **Run `splitData.py`:**
    ```bash
    python splitData.py
    ```
4.  **Verify:** The script will create `Dataset/SplitData` with `train`, `val`, and `test` subfolders (each containing `images` and `labels`). It will also generate `data.yaml` inside `Dataset/SplitData`. Review `data.yaml` to ensure paths and class names are correct.

### Step 3: Train YOLO Model (`FaceDetectorTest.py`)

This step trains your custom YOLOv8 model using the prepared dataset.

1.  **Download a base YOLOv8 model:** If you don't have one, `FaceDetectorTest.py` uses `yolov8n.pt`. It will be downloaded automatically by `ultralytics` if not found.
2.  **Run `FaceDetectorTest.py`:**
    ```bash
    python FaceDetectorTest.py
    ```
3.  **Monitor Training:**
    * The training process will start. This can take a considerable amount of time depending on your dataset size, `epochs` (currently set to 3 for demonstration), and hardware (GPU is highly recommended).
    * Training metrics will be displayed in the console.
    * Trained weights will be saved in `runs/detect/trainX/weights/best.pt`.
4.  **Copy Trained Model:** Once training is complete, copy the `best.pt` file from the training output directory (e.g., `runs/detect/train/weights/best.pt` or `runs/detect/train2/weights/best.pt` etc.) to your `model` folder (e.g., `model/l_version_1_300.pt`). **You must update the path in `main.py` if your model name or location changes.**

### Step 4: Real-time Fake Face Detection (`main.py`)

After collecting data, splitting it, and training your model, you can run the main application for real-time detection.

1.  **Update Model Path:** In `main.py`, ensure the `model = YOLO("C:/Users/Burak/OneDrive/Masaüstü/Face/model/l_version_1_300.pt")` line points to the correct path of your *trained* `.pt` model file.
2.  **Run `main.py`:**
    ```bash
    python main.py
    ```
3.  **Observe:**
    * A window will open showing your webcam feed.
    * The system will attempt to classify faces as "REAL" (green bounding box) or "FAKE" (red bounding box) in real-time.
    * FPS will be displayed on the top-left.
4.  **Exit:** Press the `q` key on your keyboard to close the application window.

## Customization

* **Camera Index & Resolution:** In `main.py` and `DataCollection.py`, adjust `cap = cv2.VideoCapture(0)` for different cameras and `cap.set(3, 640)`, `cap.set(4, 480)` for desired resolution.
* **Detection Confidence:** In `main.py` and `DataCollection.py`, adjust the `confidence` variable (e.g., `confidence = 0.6`) to set the minimum score for a detection to be displayed/saved.
* **`DataCollection.py` Specifics:**
    * `blurThreshold`: Adjust if faces are saved too blurry or too few are saved.
    * `offsetPercentageW`, `offsetPercentageH`: Change these to expand or shrink the bounding box used for cropping faces during data collection.
* **`FaceDetectorTest.py` Specifics:**
    * `epochs`: Increase for potentially better model accuracy (at the cost of longer training time).
    * `data`: Ensure this points to your `data.yaml` file.
* **`splitData.py` Specifics:**
    * `splitRatio`: Modify the train/val/test proportions as needed.
    * `classes`: Ensure this list matches the classes you are collecting.

## Troubleshooting

* **"Could not read image from camera":** Ensure your webcam is connected, not being used by another application, and its drivers are installed correctly. Try changing `cv2.VideoCapture(0)` to `1` or higher.
* **"Model not found":** Double-check the path to your `.pt` model file in `main.py`.
* **No detections/Poor accuracy:**
    * **Data quality:** Ensure you collected a diverse and clean dataset with sufficient examples for both "real" and "fake" classes.
    * **Training:** Increase `epochs` during training.
    * **Model Size:** Consider starting with a larger YOLOv8 model (e.g., `yolov8s.pt` or `yolov8m.pt`) if `yolov8n.pt` is not sufficient, but this will require more computational resources.
    * **Confidence threshold:** Try lowering `confidence` in `main.py` to see more detections (though some might be false positives).
* **GPU Issues:** If training on GPU fails, ensure CUDA/cuDNN and PyTorch are installed correctly and compatible with your GPU and Ultralytics version.
