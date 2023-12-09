from torch.utils.data import Dataset
from pathlib import Path
import os
import pandas as pd
import numpy as np

def image_exists(image_name, images_path = "./Images"):
  path = os.path.join(images_path, image_name)
  return os.path.exists(path)

def load_image(image_name, images_path = "./Images"):
  path = os.path.join(images_path, image_name)
  return Image.open(path)

class Yoga82(Dataset):
  def __init__(self, train_or_test, images_dir = "./Images", labels_dir = "./Yoga-82"):
    self.images_dir = Path(images_dir)
    self.labels_dir = Path(labels_dir)
    self.train_or_test = train_or_test
    assert train_or_test == "train" or train_or_test == "test", "train or test only"

    # Read dataframe
    csv_path = os.path.join(self.labels_dir, f"yoga_{self.train_or_test}.txt")
    self.data = pd.read_csv(csv_path, names=["file", "class1", "class2", "class3"])
    self.data = self.data[self.data['file'].apply(image_exists, images_path=self.images_dir)].reset_index(drop=True)

  def __len__(self):
    return len(self.data)

  @staticmethod
  def preprocess(pil_img):
    pil_img = pil_img.resize((128, 128))
    img = np.asarray(pil_img)
    return img

  def __getitem__(self, index):
    pil_img = load_image(self.data.iloc[index,0], self.images_dir)
    img = self.preprocess(pil_img)
    label = self.data.iloc[index, 1]
    return {
        "image": torch.from_numpy(np.array(img.copy(), dtype=np.float32).transpose(2,0,1)),
        "label": label
    }