import os
import shutil
import random

def split_dataset(dataset_folder, output_folder, split_ratio=0.8):
    """
    Splits the dataset into training and test sets.

    Parameters:
    - dataset_folder: Path to the original dataset folder.
    - output_folder: Path where the split datasets will be saved.
    - split_ratio: Proportion of data to be used for training (0 < split_ratio < 1).
    """
    # Create output folders
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Get list of classes
    classes = os.listdir(dataset_folder)

    for cls in classes:
        cls_path = os.path.join(dataset_folder, cls)
        if os.path.isdir(cls_path):
            # Create class folders in train and test directories
            os.makedirs(os.path.join(train_folder, cls), exist_ok=True)
            os.makedirs(os.path.join(test_folder, cls), exist_ok=True)

            # List all files in the class folder
            files = os.listdir(cls_path)
            random.shuffle(files)  # Shuffle files to randomize the split

            # Determine split index
            split_index = int(len(files) * split_ratio)

            # Split files into training and test sets
            train_files = files[:split_index]
            test_files = files[split_index:]

            # Copy files to respective folders
            for file_name in train_files:
                shutil.copy(os.path.join(cls_path, file_name), os.path.join(train_folder, cls, file_name))
            for file_name in test_files:
                shutil.copy(os.path.join(cls_path, file_name), os.path.join(test_folder, cls, file_name))

# Example usage
split_dataset('data/MRI', 'data/MRI_split', split_ratio=0.8)
