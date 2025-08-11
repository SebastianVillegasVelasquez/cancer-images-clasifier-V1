import os
import pandas as pd
import shutil
import random
from collections import defaultdict

def create_balanced_dataset():
    try:
        # Define benign and malign categories
        benign_classes = ['bkl', 'nv', 'df', 'vasc']  # Note: 'vasc' instead of 'vas' based on actual folder name
        malign_classes = ['mel', 'bcc', 'akiec']

        # Read the CSV file
        csv_file = pd.read_csv('./HAM10000_metadata.csv')

        # Create the main data directory
        data_dir = '../data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Create benign and malign directories with train, validation, and test subdirectories
        for category in ['benign', 'malign']:
            category_dir = os.path.join(data_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

            for split in ['train', 'validation', 'test']:
                split_dir = os.path.join(category_dir, split)
                if not os.path.exists(split_dir):
                    os.makedirs(split_dir)
                    print(f"Created directory: {split_dir}")

        # Collect all image paths by diagnosis
        image_paths_by_dx = defaultdict(list)

        # Walk through the dataset directory to find all images
        for dx in os.listdir('../dataset'):
            dx_dir = os.path.join('../dataset', dx)
            if os.path.isdir(dx_dir):
                for file in os.listdir(dx_dir):
                    if file.endswith('.jpg'):
                        image_paths_by_dx[dx].append(os.path.join(dx_dir, file))

        # Determine the category (benign or malign) for each diagnosis
        dx_to_category = {}
        for dx in benign_classes:
            dx_to_category[dx] = 'benign'
        for dx in malign_classes:
            dx_to_category[dx] = 'malign'

        # Find the minimum number of images per category to ensure balance
        benign_images = []
        for dx in benign_classes:
            benign_images.extend(image_paths_by_dx.get(dx, []))

        malign_images = []
        for dx in malign_classes:
            malign_images.extend(image_paths_by_dx.get(dx, []))

        min_images = min(len(benign_images), len(malign_images))
        print(f"Total benign images: {len(benign_images)}")
        print(f"Total malign images: {len(malign_images)}")
        print(f"Using {min_images} images per category for balance")

        # Randomly sample to get balanced datasets
        random.seed(42)  # For reproducibility
        if len(benign_images) > min_images:
            benign_images = random.sample(benign_images, min_images)
        if len(malign_images) > min_images:
            malign_images = random.sample(malign_images, min_images)

        # Split into train (80%), validation (10%), and test (10%)
        for category, images in [('benign', benign_images), ('malign', malign_images)]:
            random.shuffle(images)

            train_size = int(0.8 * len(images))
            val_size = int(0.1 * len(images))

            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]

            print(f"{category} split: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")

            # Copy images to their respective directories
            for split_name, split_images in [('train', train_images), ('validation', val_images), ('test', test_images)]:
                target_dir = os.path.join(data_dir, category, split_name)

                for src_path in split_images:
                    filename = os.path.basename(src_path)
                    dst_path = os.path.join(target_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied {src_path} to {dst_path}")

        print("Dataset creation completed!")

    except Exception as e:
        print(f'Error: {str(e)}')
        print('Make sure HAM10000_metadata.csv is in the utils directory and dataset folder exists.')

if __name__ == "__main__":
    create_balanced_dataset()
