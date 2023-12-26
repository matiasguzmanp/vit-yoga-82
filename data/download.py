import os
from PIL import Image

def get_all_file_names(folder_path: str):
    file_names = []
    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            file_names.append(os.path.join(foldername, filename))
    return file_names

def download_and_store(file_path: str, output_folder: str):
    with open(file_path, 'r') as file:
      lines = file.readlines()
    for i, line in enumerate(lines):
      line = line.strip()
      try:
        name, url = line.split("\t")
        image_path = os.path.join(output_folder, name)
        if not os.path.isfile(image_path):
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            os.system("wget -O " + image_path + " " + url + " --timeout=5 --tries=3")
            print(f"Image downloaded successfully and saved at: {name}")
      except Exception as e:
        print(f"Image could not be downloaded: {e}")

if __name__ == "__main__":
    folder_path = "./Yoga-82/yoga_dataset_links"
    output_folder = "./Images/"

    files = get_all_file_names(folder_path)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for file in files:
        download_and_store(file, output_folder)