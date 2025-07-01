from ultralytics import YOLO # Import YOLO from ultralytics for training models.

# Initialize a YOLO model. 'yolov8n.pt' is a pre-trained Nano model.
model = YOLO('yolov8n.pt')

def main():
    """
    Main function to initiate the training process of the YOLO model.
    It specifies the data configuration file and the number of epochs.
    """
    # Start training the model.
    # 'data': Path to the YAML file that defines the dataset structure (image paths, class names).
    # 'epochs': Number of training epochs.
    model.train(data='Dataset/SplitData/dataOffline.yaml', epochs=3)


if __name__ == '__main__':
    # Ensure that the main() function is called only when the script is executed directly.
    main()