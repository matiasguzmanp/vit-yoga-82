import torch
import torch.nn as nn

from utils.train import train, make
from utils.test import test 

from data.dataset import Yoga82
from data.measure import compute_weights, mean_and_std_calculator

from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import wandb

def fine_tune_model_pipeline(hyperparameters, train_mean, train_std, class_weights, weights_path, new_n_classes):
    # tell wandb to get started
    with wandb.init(project="vit-yoga82-6to82", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, val_loader, test_loader, criterion, optimizer = make(config, train_mean, train_std, class_weights, n_classes=new_n_classes)
      print(model)

      # load weights and change model
      state_dict = torch.load(weights_path)["model_state_dict"]
      model.load_state_dict(state_dict)

      for i, (module, param) in enumerate(zip(model.modules(), model.parameters())):
        param.requires_grad = False

      model.classifier = nn.Linear(config.D, new_n_classes).to(config.device)

      print(model)
      # and use them to train the model
      train_loss, val_loss = train(model, train_loader, val_loader, criterion, optimizer, config)

      # and test its final performance
      conf_mat, acc = test(model, test_loader)

    return model, train_loss, val_loss, conf_mat, acc


if __name__ == "__main__":
   
   transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = Yoga82(train_val_test="train", csv_path="./train_dataframe.csv", transform=transform, n_classes=82)

loader = DataLoader(dataset,
                        batch_size=10,
                        num_workers=0,
                        shuffle=False,
                        drop_last=False)

mean, std = mean_and_std_calculator(loader)
weights = compute_weights(loader)

weights_path = "/content/drive/MyDrive/yoga-82/vit_checkpoint_6_20.pth"


config = dict(
    chw = (3,128,128),
    patch_size = 8,
    D = 768,
    n_classes = 20,
    heads = 12,
    layers = 12,
    epochs = 5,
    lr = 1e-5,
    batch_size = 32,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

model, train_loss, val_loss, conf_mat, acc = fine_tune_model_pipeline(config,
                                                  train_mean = mean,
                                                  train_std = std,
                                                  class_weights = weights,
                                                  weights_path=weights_path,
                                                  new_n_classes=82)