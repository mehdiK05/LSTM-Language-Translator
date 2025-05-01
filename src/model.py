'''This code implements a sequence-to-sequence model with attention mechanism using PyTorch.
The model consists of an encoder and a decoder, both using LSTM layers. 
The encoder processes the input sequence and generates hidden states,while the decoder uses these hidden states to generate the output sequence.

The shapes of the tensors at each step are mentioned in the comments to help understand the flow of data through the model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math

#The Encoder implementation :


class Encoder(nn.Module):
  def __init__(self,vocabsize_src, embed_dim, hidden_dim, num_layers,bidirectional ,dropout):
    super().__init__()
    self.embd = nn.Embedding(vocabsize_src,embed_dim)
    self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,bidirectional = bidirectional)
    self.dropout = nn.Dropout(dropout)

  #using dropout to prevent overfitting
  def forward(self, x):
    embeddings = self.dropout(self.embd(x))
    encoder_outs, hidden = self.rnn(embeddings)

    return self.dropout(encoder_outs), hidden


#Attention : 

class Attention(nn.Module):
    def __init__(self, hidden_dim, method):
        super().__init__()
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
      #Apply Wₐ to encoder stat
      energy = self.w(enc_outs)  # [seq_len, batch_size, hidden_dim]

      
      return torch.sum(dec_out * energy, dim=2)  # [seq_len, batch_size]
    
      #Concatenation method : 

    def concat(self, dec_out, enc_outs):
      # dec_out: [batch_size, hidden_dim]
      # enc_outs: [seq_len, batch_size, hidden_dim]

      #Expand dec_out to match encodr outpts
      dec_out_new = dec_out.unsqueeze(0)  # [1, batch_size, hidden_dim]
      dec_out_expanded = dec_out_expanded.expand_as(enc_outs)  # [seq_len, batch_size, hidden_dim]

      
      combined = torch.cat([dec_out_expanded, enc_outs], dim=2)  # [seq_len, batch_size, hidden_dim*2]

      
      transformed = torch.tanh(self.w(combined))  # [seq_len, batch_size, hidden_dim]

      scores = torch.sum(self.v * transformed, dim=2)  # [seq_len, batch_size]

      return scores 

   
class Decoder(nn.Module):
    """
    
    This decoder takes the encoder outputs and previous hidden state to generate the next token.
    
    Args:
        vocabsize_tgt (int): Vocabulary size of the target language
        embed_dim (int): Dimension of the embedding layer
        hidden_dim (int): Hidden dimension of the LSTM
        n_layers (int): Number of LSTM layers
        use_birnn (bool): Whether the encoder used a bidirectional RNN
        dout (float): Dropout probability
        attn (str): Type of attention mechanism ('dot', 'general', or 'concat')
        tied (bool): Whether to tie the embedding and output projection weights
    """
    def __init__(self, vocabsize_tgt, embed_dim, hidden_dim, n_layers, use_birnn, 
                 dout, attn, tied):
        super().__init__()
        
        self.hidden_dim = hidden_dim*2 if use_birnn else hidden_dim
        
        self.embed = nn.Embedding(vocabsize_tgt, embed_dim)
        
        self.rnn = nn.LSTM(embed_dim, self.hidden_dim, n_layers)
        # Attention mechanism
        self.attn = Attention(self.hidden_dim, attn)
        
        # Context combination layer - combines decoder output with context vector
        # The size is doubled because we concatenate decoder output and context vector
        self.context_combiner = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
        # mappinghidden state to vocab
        self.out_projection = nn.Linear(self.hidden_dim, vocabsize_tgt)
        
        # Tiie share weights between input embedding and output projection
        # This is a common technique to reduce params
        if tied: 
            if embed_dim != self.hidden_dim:
                raise ValueError(
                    f"When using tied weights, embed_dim ({embed_dim}) "
                    f"must equal hidden_dim ({self.hidden_dim})")
            self.out_projection.weight = self.embed.weight
            
        # Dropout for regularization
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs, hidden, enc_outs):
        """
        Forward pass of the decoder.
        
        Args:
            inputs (Tensor): Input tokens of shape [batch_size]
            hidden (tuple): Previous hidden state (h, c) from the decoder
            enc_outs (Tensor): Encoder outputs of shape [seq_len, batch_size, hidden_dim]
            
        Returns:
            Tensor: Predicted token probabilities
            tuple: Updated hidden state
        """
        # Add sequence dimension to inputs [1, batch_size]
        inputs = inputs.unsqueeze(0)
        
        embedded = self.dropout(self.embed(inputs)) # [1, batch_size, embed_dim]
        
       
        # dec_out shape: [1, batch_size, hidden_dim]
        # hidden is a tuple of (h, c) each with shape [n_layers, batch_size, hidden_dim]
        dec_out, hidden = self.rnn(embedded, hidden)
        # Calculate attention weights 
        attn_weights = self.attn(dec_out, enc_outs).transpose(1, 0) #[batch_size, seq_len]
    
        enc_outs = enc_outs.transpose(1, 0) #[batch_size, seq_len, hidden_dim]
        
        # cotext vect
        
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outs) # [batch_size, 1, hidden_dim]
        
        # Combine decoder output with context vector
        # Shape: [1, batch_size, hidden_dim*2] -> [1, batch_size, hidden_dim]
        combined = torch.cat((dec_out, context.transpose(1, 0)), dim=2)
        combined = self.context_combiner(combined).tanh()
        
        # Project to vocabulary size and remove sequence dimension
        # Shape: [batch_size, vocabsize_src]
        predictions = self.out_projection(combined.squeeze(0))
        
        return predictions, hidden
    


class Seq2SeqModel(nn.Module):

    
    def __init__(self, args, fields, device):
        
        super().__init__()
         # Unpack the field objects for source and target languages
        self.source_field, self.target_field = fields
        
        self.source_vocab_size = len(self.source_field[1].itos)
        self.target_vocab_size = len(self.target_field[1].itos)
        
        # Store model configuration for later use
        self.device = device
        self.num_layers = args.n_layers
        self.hidden_dimension = args.hidden_dim
        self.bidirectional = args.bidirectional
        
       
        self.encoder = Encoder(
            vocabsize_tgt=self.source_vocab_size,        
            embed_dim=args.embed_dim,          
            hidden_dim=args.hidden_dim,        
            n_layers=args.n_layers,           
            use_birnn=args.bidirectional,      
            dout=args.dropout                  
        )
        

        self.decoder = Decoder(
            vocabsize_tgt=self.target_vocab_size,        
            embed_dim=args.embed_dim,          
            hidden_dim=args.hidden_dim,        
            n_layers=args.n_layers,            
            use_birnn=args.bidirectional,      
            dout=args.dropout,                 
            attn=args.attn,                    
            tied=args.tied                     
        )

    def forward(self, source_seq, target_seq=None, max_length=100, teacher_forcing_ratio=0.0):
        """
        
        Args:
            source_seq: Input sequence tensor of shape (src_len, batch_size)
            target_seq: Target sequence tensor of shape (tgt_len, batch_size),
                        or None during inference
           
        Returns:
            Tensor of shape (tgt_len, batch_size, tgt_vocab_size) containing
            probability distributions over the target vocabulary for each output position
        """
        
        source_length, batch_size = source_seq.size()
        
        # target sequence length:
        
        target_length = target_seq.size(0) if isinstance(target_seq, torch.Tensor) else max_length
        
        effective_tf_ratio = teacher_forcing_ratio if isinstance(target_seq, torch.Tensor) else 0.0
        
        # - hidden: tuple of (h_n, c_n) containing final hidden states
        encoder_outputs, hidden = self.encoder(source_seq)
        
        
        decoder_input = torch.ones_like(source_seq[0]) * 2  # <eos> token at id 2
        
        # Container for storing decoder outputs at each step
        outputs = []
        
        
        if self.bidirectional:
            def transform_hidden(hidden_state):
                
                # (layers*directions, batch, hidden) -> (layers, directions, batch, hidden)
                reshaped = hidden_state.view(self.num_layers, 2, batch_size, self.hidden_dimension)
                
                # For each layer, concatenate forward and backward directions
                # Result: (layers, batch, hidden*2)
                concatenated = torch.stack([torch.cat((h[0], h[1]), 1) for h in reshaped])
                return concatenated
            
            # Apply transformation to both hidden and cell states (h_n and c_n)
            hidden = tuple(transform_hidden(h) for h in hidden)
        
        #Decode step-by-step
        for step in range(target_length):
            # Pass current token, previous hidden state, and encoder outputs to decoder
            prediction, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            
            # Store current step prediction
            outputs.append(prediction)
            
            use_teacher_forcing = random.random() < effective_tf_ratio
            decoder_input = target_seq[step] if use_teacher_forcing else prediction.max(1)[1]
        
        # Stack all decoder outputs
        return torch.stack(outputs)
    
