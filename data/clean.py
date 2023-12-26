import os
import pandas as pd
from PIL import Image

# Function to check if an image is valid (3 channels)
def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        return img.mode == 'RGB'  # Check if the image has 3 channels
    except (IOError, OSError):
        return False

# Function to process the DataFrame and delete invalid images
def process_dataframe(df):
    invalid_rows = []
    for index, row in df.iterrows():
        file_path = f"./Images/{row['file']}"
        if not is_valid_image(file_path):
            invalid_rows.append(index)
            try:
                os.remove(file_path)  # Delete the image file
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")

    # Drop rows with invalid images
    df_cleaned = df.drop(index=invalid_rows)
    return df_cleaned

def clean_dataset(csv_path, images_path="./Images"):
  df = pd.read_csv(csv_path, names=["file", "class1", "class2", "class3"])
  df_cleaned = process_dataframe(df)
  return df_cleaned.reset_index(drop=True)

if __name__ == "__main__":
    train_dataframe = clean_dataset(csv_path = "./Yoga-82/yoga_train.txt").to_csv("train_dataframe.csv", index=False)
    test_dataframe = clean_dataset(csv_path = "./Yoga-82/yoga_test.txt").to_csv("test_dataframe.csv", index=False)