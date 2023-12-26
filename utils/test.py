import torch
from sklearn.metrics import confusion_matrix, accuracy_score


def test(model, test_loader,device):
  y_pred = []
  y_true = []
  model.eval()
  with torch.no_grad():
    for _, batch in enumerate(test_loader):
      images = batch["images"].to(device)
      labels = batch["labels"]
      output = model(images)
      _, batch_predictions = torch.max(output, dim=1)

      y_pred.extend(batch_predictions.cpu().tolist())
      y_true.extend(labels.tolist())

  # Calculate confusion matrix
  conf_matrix = confusion_matrix(y_true, y_pred, normalize='all')

  # Calculate accuracy
  accuracy = accuracy_score(y_true, y_pred)

  return conf_matrix, accuracy