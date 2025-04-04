import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

#The Encoder implementation :


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


#Attention : 

class Attention(nn.Module):
    def __init__(self, hidden_dim, method):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim) 
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim*2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, dec_out, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(dec_out, enc_outs)
        return F.softmax(attn_energies, dim=0)
    

        #dot methos:
    def dot(self, dec_out, enc_outs):
      # dec_out: [batch_size, hidden_dim]
      # enc_outs: [seq_len, batch_size, hidden_dim]

      
      energy = dec_out * enc_outs  # [seq_len, batch_size, hidden_dim]

      # Step 2: Sum over hidden_dim (dim=2)
      return torch.sum(energy, dim=2)  # [seq_len, batch_size]

      #general method: 

    def general(self, dec_out, enc_outs):
      # enc_outs: [seq_len, batch_size, hidden_dim]
      # Step 1: Apply Wₐ to encoder states
      energy = self.w(enc_outs)  # [seq_len, batch_size, hidden_dim]

      # Step 2: Same as dot product
      return torch.sum(dec_out * energy, dim=2)  # [seq_len, batch_size]
    
      #Concatenation method : 

    def concat(self, dec_out, enc_outs):
      # dec_out: [batch_size, hidden_dim]
      # enc_outs: [seq_len, batch_size, hidden_dim]

      # Step 1: Expand dec_out to match enc_outs' seq_len
      dec_out_expanded = dec_out.unsqueeze(0)  # [1, batch_size, hidden_dim]
      dec_out_repeated = dec_out_expanded.expand_as(enc_outs)  # [seq_len, batch_size, hidden_dim]

      # Step 2: Concatenate decoder and encoder states
      combined = torch.cat([dec_out_repeated, enc_outs], dim=2)  # [seq_len, batch_size, hidden_dim*2]

      # Step 3: Apply Wₐ and tanh
      transformed = torch.tanh(self.w(combined))  # [seq_len, batch_size, hidden_dim]

      # Step 4: Multiply by vₐ (learned vector)
      # v has shape [hidden_dim], transformed is [seq_len, batch_size, hidden_dim]
      # To compute the dot product, we sum over hidden_dim after multiplying
      scores = torch.sum(self.v * transformed, dim=2)  # [seq_len, batch_size]

      return scores 

      