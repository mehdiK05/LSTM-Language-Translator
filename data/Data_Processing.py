import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

# Define vocabulary class
class Vocab:
    def __init__(self, freq_threshold=2):
        
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"} #index to string
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3} #string to index
        self.freq_threshold = freq_threshold

    #size of the vocabulary:    
    def __len__(self):
        return len(self.itos)
        
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start after special tokens
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
                
    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]

# Translation Dataset
class TranslationDataset(Dataset):
    def __init__(self, darija_text, english_text, darija_vocab, english_vocab):
        self.darija_text = darija_text
        self.english_text = english_text
        self.darija_vocab = darija_vocab
        self.english_vocab = english_vocab
        
    def __len__(self):
        return len(self.darija_text)
        
    def __getitem__(self, index):
        darija_sentence = self.darija_text[index]
        english_sentence = self.english_text[index]
        
        # Convert to numerical form
        numericalized_darija = [self.darija_vocab.stoi["<sos>"]]
        numericalized_darija += self.darija_vocab.numericalize(darija_sentence)
        numericalized_darija.append(self.darija_vocab.stoi["<eos>"])
        
        numericalized_english = [self.english_vocab.stoi["<sos>"]]
        numericalized_english += self.english_vocab.numericalize(english_sentence)
        numericalized_english.append(self.english_vocab.stoi["<eos>"])
        
        return torch.tensor(numericalized_darija), torch.tensor(numericalized_english)

# Padding collate function
def pad_collate(batch):
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        src_list.append(_src)
        tgt_list.append(_tgt)
        
    src = pad_sequence(src_list, padding_value=0)
    tgt = pad_sequence(tgt_list, padding_value=0)
    
    return src, tgt

# Function to prepare data
def prepare_data(df,test_size=0.1):
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=42)
    
    
    darija_vocab = Vocab(freq_threshold=2)
    darija_vocab.build_vocabulary(train_df["darija"].tolist())
    
    english_vocab = Vocab(freq_threshold=2)
    english_vocab.build_vocabulary(train_df["english"].tolist())
    
    train_dataset = TranslationDataset(
        train_df["darija"].tolist(),
        train_df["english"].tolist(),
        darija_vocab,
        english_vocab
    )
    
    val_dataset = TranslationDataset(
        val_df["darija"].tolist(),
        val_df["english"].tolist(),
        darija_vocab,
        english_vocab
    )
    
    test_dataset = TranslationDataset(
        test_df["darija"].tolist(),
        test_df["english"].tolist(),
        darija_vocab,
        english_vocab
    )
    
    return train_dataset, val_dataset, test_dataset, darija_vocab, english_vocab

# Function to create DataLoaders
def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate
    )
    
    return train_loader, val_loader, test_loader

