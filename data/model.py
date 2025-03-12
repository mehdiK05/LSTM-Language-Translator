import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
  def __init__(input_dim, embed_dim, hidden_dim, num_layers, dropout):
    super().__init__()
    self.embd = nn.Embedding(input_dim,embed_dim)
    self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,bidirectional = True)
    self.dropout = nn.Dropout(dropout)

  #using dropout to prevent overfitting
  def forward(self, X):
    embeddings = self.dropout(self.embd(x))
    encoder_outs, hidden = self.rnn(embeddings)

    return self.dropout(encoder_outs), hidden

