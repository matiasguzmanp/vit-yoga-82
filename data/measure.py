import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight

def mean_and_std_calculator(loader):
  var = 0.0
  pixel_count = 0
  mean = 0.0
  for data in loader:
      images = data["images"]
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
  mean = mean / len(loader.dataset)

  var = 0.0
  pixel_count = 0
  for data in loader:
      images = data["images"]
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      var += ((images - mean.unsqueeze(1))**2).sum([0,2])
      pixel_count += images.nelement() / images.size(1)
  std = torch.sqrt(var / pixel_count)
  return mean, std

def compute_weights(loader):
  y = []
  for data in loader:
    label = data["labels"].tolist()
    y.extend(label)
  class_weights=compute_class_weight(
      class_weight="balanced",
      classes=np.unique(y),
      y=np.array(y))
  class_weights=torch.tensor(class_weights,dtype=torch.float)
  return class_weights