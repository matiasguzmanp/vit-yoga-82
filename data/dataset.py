from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch

import os

import pandas as pd
import numpy as np

def load_image(image_name, images_path = "./Images"):
  path = os.path.join(images_path, image_name)
  return Image.open(path)

class Yoga82(Dataset):
  def __init__(self, train_val_test, csv_path="./train_dataframe.csv", images_path="./Images",  n_classes=82, transform=None):
    self.images_path = images_path
    self.transform = transform
    self.data = pd.read_csv(csv_path).sample(frac=1)

    if train_val_test == "train":
      start_index = 0
      end_index = int(len(self.data)*0.8)
      self.data = self.data.iloc[start_index:end_index]
    elif train_val_test == "val":
      start_index = int(len(self.data)*0.8)
      end_index = len(self.data)
      self.data = self.data.iloc[start_index:end_index]
    else:
      self.data = self.data

    self.data = self.data.values.tolist()


    match n_classes:
      case 82:
        self.label_index = 3
      case 20:
        self.label_index = 2
      case 6:
        self.label_index = 1
      case _:
        print("Invalid number of classes for Yoga-82 dataset! Setting default 82 classes...")
        self.label_index = 3

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    pil = load_image(self.data[index][0], self.images_path)
    label = self.data[index][self.label_index]
    img = self.transform(pil)
    return {
        "images": img,
        "labels": label
    }