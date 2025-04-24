import torch  
import torch.nn as nn  
import torch.optim as optim  
import pandas as pd  
from tqdm import tqdm  
import argparse  
import os 

from model import Seq2SeqModel  
import numpy as np 
import random  
from nltk.translate.bleu_score import corpus_bleu  

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.Data_Processing import prepare_data, get_data_loaders

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Tain the model on one epoch
def train_one_epoch(model, train_loader, optimizer, criterion, device, clip=1):
 
    model.train()   
    epoch_loss = 0  
    
    for src, tgt in tqdm(train_loader, desc="Training"):
        src = src.to(device) 
        tgt = tgt.to(device)  
        
        optimizer.zero_grad()  
        
        
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        
        # Reshape output and target for loss calculation
        
        output_dim = output.shape[-1]  
        #remove first token (<sos>) 
        output = output[1:].reshape(-1, output_dim)   
        tgt = tgt[1:].reshape(-1)  
        
        loss = criterion(output, tgt)  
        loss.backward()  
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()  
        epoch_loss += loss.item()  
    
    
    return epoch_loss / len(train_loader)

# Evaluating the model 
def evaluate(model, val_loader, criterion, device):
   
    model.eval() 
    epoch_loss = 0  
    
    with torch.no_grad():  
        for src, tgt in tqdm(val_loader, desc="Evaluating"):
            src = src.to(device)  
            tgt = tgt.to(device)  
            
            # Forward pass with only predicted tokens
            output = model(src, tgt, teacher_forcing_ratio=0)
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]  #
            output = output[1:].view(-1, output_dim)  
            tgt = tgt[1:].reshape(-1)  

            loss = criterion(output, tgt) 
            epoch_loss += loss.item()  
    
    
    return epoch_loss / len(val_loader)

#Translate one sentence
def translate_sentence(model, sentence, src_field, tgt_field, device, max_length=50):
  
    model.eval()  

    #add batch dimension:
    src_tensor = sentence.unsqueeze(1).to(device)
        
    # Run the encoder 
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
    # First input to the decoder is the <sos> token
    tgt_idx = [tgt_field[1].vocab.stoi["<sos>"]]
    
    # Generate translation token by token
    for _ in range(max_length):
        tgt_tensor = torch.LongTensor([tgt_idx[-1]]).to(device)  # Get the last predicted token
        
        #next token prediction
        with torch.no_grad():
            output, hidden = model.decoder(tgt_tensor, hidden, encoder_outputs)
            
        pred_token = output.argmax(1).item()
        
        if pred_token == tgt_field[1].vocab.stoi["<eos>"]:
            break
            
        tgt_idx.append(pred_token)  
        
    # Convert indices to words (remove <sos> token at the beginning)
    tgt_tokens = [tgt_field[1].vocab.itos[i] for i in tgt_idx[1:]]
    
    return tgt_tokens

# Function to calculate BLEU score on the test set
def calculate_bleu(model, test_loader, src_field, tgt_field, device):
   
    model.eval()  
    references = []  #store reference translations
    predicted = []  # store predicted translations
    
    with torch.no_grad():  
        for src, tgt in tqdm(test_loader, desc="Calculating BLEU"):
            src = src.to(device)  
            
            # Process each sentence in the batch
            for i in range(src.shape[1]):
                src_sentence = src[:, i]  # Get i-th sentence in batch
                tgt_sentence = tgt[:, i]  # Get i-th target in batch
                
                # Get reference and remove special tokens
                reference = [tgt_field[1].vocab.itos[token.item()] for token in tgt_sentence 
                            if token.item() != tgt_field[1].vocab.stoi["<sos>"] and 
                               token.item() != tgt_field[1].vocab.stoi["<eos>"] and
                               token.item() != tgt_field[1].vocab.stoi["<pad>"]]
                
                # Get predicted translation
                translation = translate_sentence(model, src_sentence, src_field, tgt_field, device)
                
                
                references.append([reference])  #
                predicted.append(translation)
    
    # Calculate and return BLEU score
    return corpus_bleu(references, predicted)

# Main function to run the training pipeline
def main():
    """Main function to parse arguments and run the training pipeline."""
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Train a Seq2Seq model for translation')
    
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--output_dir', type=str, default='./model_checkpoints', help='Directory to save model checkpoints')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional encoder')
    parser.add_argument('--attn', type=str, default='general', choices=['dot', 'general', 'concat'], help='Attention mechanism')
    parser.add_argument('--tied', action='store_true', help='Tie encoder and decoder embedding weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()  
    
    
    set_seed(args.seed)
    
    # Create output directory 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data from CSV file
    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} sentence pairs")
    
    train_dataset, val_dataset, test_dataset, fields = prepare_data(df)
    
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )
    
    # Extract fields
    darija_field = fields[0]
    english_field = fields[1]
    
    # vocabulary sizes
    print(f"Darija vocabulary size: {len(darija_field[1].vocab)}")
    print(f"English vocabulary size: {len(english_field[1].vocab)}")
    
    # Initialize model
    model = Seq2SeqModel(args, fields, device).to(device)
    
    # Initialize optimizer (Adam) and loss function (CrossEntropyLoss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=darija_field[1].vocab.stoi["<pad>"])
    
    # Training loop variables
    best_valid_loss = float('inf')  # Initialize with infinity to always save first model
    patience_counter = 0  # Counter for early stopping
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.clip)
        
        # Evaluate on validation set
        valid_loss = evaluate(model, val_loader, criterion, device)
        
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")
        
        # Save the best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss  # Update best validation loss
            
            # Save checkpoint with all necessary information
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fields': fields,
                'args': args
            }, os.path.join(args.output_dir, 'best_model.pt'))
            
            patience_counter = 0  # Reset patience counter
            print("Best model saved!")
        else:
            patience_counter += 1  # Increment patience counter if no improvement
            
        # Early stopping 
        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Calculate BLEU score on test set
    bleu_score = calculate_bleu(model, test_loader, darija_field, english_field, device)
    print(f"BLEU Score: {bleu_score:.4f}")
    
    # Save final model with test metrics
    torch.save({
        'model_state_dict': model.state_dict(),
        'fields': fields,
        'args': args,
        'test_loss': test_loss,
        'bleu_score': bleu_score
    }, os.path.join(args.output_dir, 'final_model.pt'))
    print("Final model saved!")


if __name__ == "__main__":
    main()