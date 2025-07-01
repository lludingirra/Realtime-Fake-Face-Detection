import os # Import the os module for operating system interactions (paths, directories).
import random # Import random for shuffling data.
import shutil # Import shutil for high-level file operations (e.g., copying, deleting directories).
from itertools import islice # Import islice for slicing iterators (used for splitting data).

# Define input and output folder paths for the dataset.
inputFolderPath = "Dataset/All" # Folder containing all original images and their corresponding labels (annotations).
outputFolderPath = "Dataset/SplitData" # Folder where the split dataset (train, val, test) will be saved.

# Define the ratio for splitting the dataset into training, validation, and test sets.
splitRatio = {"train" : 0.7, "val" : 0.2, "test" : 0.1}
classes = ["fake","real"] # Define the class names in your dataset.

try :
    # Attempt to remove the output folder if it already exists, to ensure a clean split.
    shutil.rmtree(outputFolderPath)
    print("Removed Directory:", outputFolderPath)
    
except OSError as e :
    # If the directory doesn't exist (or other OS error), create it.
    os.mkdir(outputFolderPath)
    print(f"Created Directory: {outputFolderPath} (or it already existed and was handled)") # More descriptive message
    
# Create necessary subdirectories for images and labels within train, val, and test sets.
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True) # exist_ok=True prevents error if dir already exists
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)

# Get all file names from the input folder.
listNames = os.listdir(inputFolderPath)
uniqueNames = [] # List to store unique base names (e.g., "image1" from "image1.jpg" and "image1.txt").

# Extract unique file names (without extensions) to group image and label files.
for name in listNames :
    uniqueNames.append(name.split(".")[0])

uniqueNames = list(set(uniqueNames)) # Convert to set then back to list to ensure uniqueness.

random.shuffle(uniqueNames) # Shuffle the unique names to ensure random distribution.

# Calculate the number of samples for each split based on the ratios.
lenData = len(uniqueNames) # Total number of unique samples.
lenTrain = int(lenData * splitRatio['train']) # Number of training samples.
lenVal = int(lenData * splitRatio['val']) # Number of validation samples.
lenTest = int(lenData * splitRatio['test']) # Number of test samples.

# Adjust lengths if there's a discrepancy due to integer conversion.
# This ensures all unique names are distributed without loss.
if lenData != lenTrain + lenTest + lenVal :
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining # Add any remaining items to the training set.

lenghtToSplit = [lenTrain, lenVal, lenTest] # List containing the final counts for each split.
Input = iter(uniqueNames) # Create an iterator from the shuffled unique names.
# Use islice to split the unique names into train, val, and test sets.
Output = [list(islice(Input, elem)) for elem in lenghtToSplit]
print(f'Total Images: {lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

sequence = ['train', 'val', 'test'] # Define the sequence of output folders.

# Copy files to their respective split directories.
for i, out in enumerate(Output) :
    for fileName in out :
        # Copy both the image (.jpg) and its corresponding label (.txt) file.
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process Completed...")

# Create the data.yaml file required by YOLO for training.
# This file specifies paths to train/val/test images, number of classes, and class names.
dataYaml = f'path: ../Data\n\
            train: ../train/images\n\
            val: ../val/images\n\
            test: ../test/images\n\
            \n\
            nc: {len(classes)}\n\
            names: {classes}'

# Write the data.yaml content to the file.
f = open(f"{outputFolderPath}/data.yaml", "a") # Open in append mode ('a') to create if not exists, append if exists.
f.write(dataYaml)
f.close()

print("Data.yaml file Created")