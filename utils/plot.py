import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss(train_loss, val_loss, info):
  plt.figure()
  plt.plot(train_loss, label="Loss de entrenamiento")
  plt.plot(val_loss, label="Loss de validación")
  plt.legend()
  plt.grid("on")
  plt.title(f"Loss de entrenamiento y validación en función de la época: {info}")


def plot_conf_mat(conf_mat, acc, info):
  plt.figure(figsize=(10,8))
  sns.heatmap(conf_mat)
  plt.title(f"Matriz de confusión {info}.\nAccuracy={acc:.4f}")
  plt.xlabel('Predichas')
  plt.ylabel('Reales')
  plt.show()