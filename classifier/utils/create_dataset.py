import os
import pandas as pd
import shutil

def create_data_set():
    try:
        csv_file = pd.read_csv('./HAM10000_metadata.csv')
        labels = csv_file.loc[:]['dx'].unique()
        print("Labels found:", labels)

        # Create dataset directory if it doesn't exist
        dataset_dir = '../dataset'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Create a subfolder for each label
        for label in labels:
            label_folder = os.path.join(dataset_dir, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
                print(f"Created folder for label: {label}")

        # Dictionary to keep track of image count per label
        image_count = {label: 0 for label in labels}

        # Walk through the HAM directory and its subfolders
        for root, dirs, files in os.walk('../HAM'):
            for file in files:
                if file.endswith('.jpg'):
                    # Extract image_id from filename (remove .jpg extension)
                    image_id = file.split('.')[0]

                    # Find the label for this image in the CSV
                    image_data = csv_file[csv_file['image_id'] == image_id]

                    if not image_data.empty:
                        label = image_data['dx'].values[0]

                        # Create new filename with label name and counter
                        image_count[label] += 1
                        new_filename = f"{label}_{image_count[label]}.jpg"

                        # Define source and destination paths
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(dataset_dir, label, new_filename)

                        # Copy the image to the appropriate label folder with the new name
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied {file} to {dst_path}")

        print("Dataset creation completed!")
        print("Images per label:")
        for label, count in image_count.items():
            print(f"{label}: {count} images")

    except Exception as e:
        print(f'Error: {str(e)}')
        print('Make sure HAM10000_metadata.csv is in the utils directory and HAM folder exists.')

create_data_set()
