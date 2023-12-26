import torch
import torch.nn as nn
import math

def make_patches(input, P):
  N, C, W, H = input.shape
  patches = input.unfold(2, P, P). \
    unfold(3, P, P). \
    permute(0, 2, 3, 1, 4, 5). \
    contiguous(). \
    view(N, H*W//(P*P), -1)
  return patches

class AttentionHead(nn.Module):
    def __init__(self, D, attention_head_size):
        super().__init__()
        self.D = D
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.Q = nn.Linear(D, attention_head_size)
        self.K = nn.Linear(D, attention_head_size)
        self.V = nn.Linear(D, attention_head_size)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        return attention_output

class MultiHeadAttention(nn.Module):
  def __init__(self, D, num_attention_heads=12):
    super(MultiHeadAttention, self).__init__()
    self.D = D
    self.num_attention_heads = num_attention_heads
    assert D % num_attention_heads == 0, "Hidden dimension must be divisible by number of attention heads"
    self.attention_head_size = self.D // self.num_attention_heads

    self.heads = nn.ModuleList([])
    for _ in range(self.num_attention_heads):
      head = AttentionHead(self.D, self.attention_head_size)
      self.heads.append(head)

  def forward(self, tensor):
    out_list = []
    for head in self.heads:
      out = head(tensor)
      out_list.append(out)

    attentions = []
    for out in out_list:
      attentions.append(out)

    attention = torch.cat(attentions, dim=-1)
    return attention

class TransformerBlock(nn.Module):
  def __init__(self, D, E, heads):
    super(TransformerBlock, self).__init__()
    self.heads = heads

    self.attention = MultiHeadAttention(D, heads)
    self.norm1 = nn.LayerNorm(D)
    self.norm2 = nn.LayerNorm(D)

    self.l1 = nn.Linear(D, E)
    self.gelu = nn.GELU()
    self.l2 = nn.Linear(E, D)

  def forward(self, tensor):
    # First normalization
    n1 = self.norm1(tensor)

    # Into Multi-Head Attention
    attention = self.attention(n1)

    # Residual
    tensor = n1 + tensor

    # Into 2 layer MLP with gelu activation
    n2 = self.norm2(tensor)
    mlp1 = self.l1(n2)
    activated = self.gelu(mlp1)
    mlp2 = self.l2(activated)

    # Last residual
    out = tensor + mlp2
    return out

class TransformerEncoder(nn.Module):
  def __init__(self, D, E, layers, heads):
    super(TransformerEncoder, self).__init__()
    self.blocks = nn.ModuleList([])
    self.heads = heads
    for _ in range(layers):
      self.blocks.append(TransformerBlock(D=D, E=E, heads=heads))
  def forward(self, tensor):
    for block in self.blocks:
      tensor = block(tensor)

    return tensor

class ViT(nn.Module):
  def __init__(self, config):
    super(ViT, self).__init__()

    # Parameters
    self.N = config.batch_size
    self.C = config.chw[0]
    self.H = config.chw[1]
    self.W = config.chw[2]
    self.P = config.patch_size
    self.d = self.P*self.P*self.C
    self.D = config.D
    self.heads = config.heads
    self.n_classes = config.n_classes
    self.layers = config.layers

    # Stuff
    self.mapper = nn.Linear(self.d, self.D)
    self.class_embedding = nn.Parameter(torch.rand(self.N, 1, self.D))
    self.positional_embedding = nn.Parameter(torch.rand((self.N, self.H*self.W//(self.P*self.P)+1, self.D )))

    self.encoder = TransformerEncoder(D=self.D, E = 4*self.D, layers=self.layers, heads=self.heads)
    self.classifier = nn.Linear(self.D, self.n_classes)


  def forward(self, images):
    # Separamos en parches
    patches = make_patches(images, self.P)

    # Mapeamos linealmente los parches hacia la dimension oculta D
    mapped = self.mapper(patches)

    # Concatenamos el embedding de clase
    embedded = torch.cat((self.class_embedding, mapped), dim = 1)

    # Le sumamos el embedding de posición
    embedded_with_position = embedded + self.positional_embedding
    encoded = self.encoder(embedded_with_position)
    out = self.classifier(encoded[:,0,:])
    return out
