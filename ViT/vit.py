import torch
import torch.nn as nn

def make_patches(input, P):
  N, C, W, H = input.shape
  print(N, C, W, H)
  output = torch.zeros((N, H*W//(P*P), P*P*C), dtype=torch.float32)
  n = 0
  for i in range(0, H - P + 1, P):
    for j in range(0, W - P + 1, P):
      patch = input[:,:,i:i+P, j:j+P].flatten()
      output[:,n,:] = patch
      n += 1
  return output

class ViT(nn.Module):
  def __init__(self, channels = 3, height = 128, width = 128, patch_size = 8):
    super(ViT, self).__init__()

    # Parameters
    self.C = channels
    self.H = height
    self.P = patch_size
    self.d = self.P*self.P*self.C
    self.D = 16

    self.mapper = nn.Linear(self.d, self.D)

  def forward(self, images):
    patches = make_patches(images, self.P)
    print(patches.shape)
    mapped = self.mapper(patches)
    print(mapped.shape)