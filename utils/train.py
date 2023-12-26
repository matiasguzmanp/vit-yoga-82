import torch
import torch.nn as nn

from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score

from data.dataset import Yoga82
from ViT.vit import ViT
from tqdm.auto import tqdm

import wandb

def get_data(train_val_test, transform, n_classes, train_mean, train_std):

  if transform:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize(train_mean, train_std),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(train_mean,train_std)
        ])
  else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize(train_mean, train_std)
        ])

  assert train_val_test == "train" or train_val_test == "val" or train_val_test == "test", "Only train, val or test is valid"

  if train_val_test == "train" or train_val_test == "val":
    csv_path = "./train_dataframe.csv"
  elif train_val_test == "test":
    csv_path = "./test_dataframe.csv"

  dataset = Yoga82(train_val_test=train_val_test, csv_path=csv_path, transform=transform, n_classes=n_classes)
  return dataset

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=2,
                                         drop_last=True)
    return loader


def make(config, train_mean, train_std, class_weights, n_classes = 82):
    # Make the data
    train = get_data(train_val_test="train", transform=True, train_mean=train_mean, train_std=train_std, n_classes=n_classes)
    val = get_data(train_val_test="val", transform=False, train_mean=train_mean, train_std=train_std, n_classes=n_classes)
    test = get_data(train_val_test="test", transform=False, train_mean=train_mean, train_std=train_std, n_classes=n_classes)

    train_loader = make_loader(train, batch_size=config.batch_size)
    val_loader = make_loader(val, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ViT(config).to(config.device)
    class_weights = class_weights.to(config.device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr)

    return model, train_loader, val_loader, test_loader, criterion, optimizer

def train(model, train_loader, val_loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(config.epochs)):
        train_epoch_loss = 0
        train_sample_counter = 0
        val_epoch_loss = 0
        val_sample_counter = 0

        for _, batch in enumerate(train_loader):
          images = batch["images"]
          labels = batch["labels"]
          loss = train_batch(images, labels, model, optimizer, criterion, device=config.device)
          train_epoch_loss += loss.item()
          train_sample_counter += 1
          example_ct +=  len(images)
          batch_ct += 1
          # Report metrics and save model every 25th batch
          if ((batch_ct + 1) % 25) == 0:
              train_log(loss, example_ct, epoch)

        for _, batch in enumerate(val_loader):
          images = batch["images"]
          labels = batch["labels"]
          loss = val_batch(images, labels, model, criterion, device=config.device)
          val_epoch_loss += loss.item()
          val_sample_counter += 1

        torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss,
              }, f'vit_checkpoint_{epoch}.pth')
        val_loss_list.append(train_epoch_loss/train_sample_counter)
        train_loss_list.append(val_epoch_loss/val_sample_counter)
    return train_loss_list, val_loss_list





def val_batch(images, labels, model, criterion, device):
    images, labels = images.to(device), labels.to(device)

    # Forward pass ➡
    with torch.no_grad():
      outputs = model(images)
      loss = criterion(outputs, labels)

    return loss


def train_batch(images, labels, model, optimizer, criterion, device):
    images, labels = images.to(device), labels.to(device)

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def val_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")