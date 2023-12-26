
from utils.train import train, make
from utils.test import test 

from data.dataset import Yoga82
from data.measure import compute_weights, mean_and_std_calculator

from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import wandb
import torch

def model_pipeline(hyperparameters, train_mean, train_std, class_weights):
    # tell wandb to get started
    with wandb.init(project="vit-yoga82-20", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, val_loader, test_loader, criterion, optimizer = make(config, train_mean, train_std, class_weights, n_classes=config.n_classes)
      print(model)

      # and use them to train the model
      train_loss, val_loss = train(model, train_loader, val_loader, criterion, optimizer, config)

      # and test its final performance
      conf_mat, acc = test(model, test_loader, device=config.device)

    return model, train_loss, val_loss, conf_mat, acc

if __name__ == "__main__":

  # First we calculate the normalization parameters and the weight list
  transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
  ])  

  dataset = Yoga82(train_val_test="train", csv_path="./train_dataframe.csv", transform=transform, n_classes=6)

  loader = DataLoader(dataset,
                          batch_size=10,
                          num_workers=0,
                          shuffle=False,
                          drop_last=False)

  mean, std = mean_and_std_calculator(loader)
  weights = compute_weights(loader)

  config = dict(
      chw = (3,128,128),
      patch_size = 8,
      D = 768,
      n_classes = 6,
      heads = 12,
      layers = 12,
      epochs = 6,
      lr = 1e-5,
      batch_size = 32,
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      )
  
  model, train_loss, val_loss, conf_mat, acc = model_pipeline(config,
                                                   train_mean = mean,
                                                   train_std = std,
                                                   class_weights = weights)