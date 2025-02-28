#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import BertForSequenceClassification, BertTokenizer

import torch
import pandas as pd
from benchmark_lms.pipeline import create_dataloaders, train_bert, evaluate_bert
from torch.optim import Adam

import torch.nn as nn
import matplotlib.pyplot as plt

import argparse


# In[ ]:


def main(json_file_path, batch_size = 32, n_epochs = 3, max_len = 128):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    df = pd.read_json(json_file_path)
    print('Loaded data from', json_file_path)
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    print('Loaded BERT model')
    bert_model = bert_model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Loaded BERT tokenizer')
    total_params = sum(p.numel() for p in bert_model.parameters())

    print(f"Total parameters in BERT: {total_params}")

    train_loader, val_loader, test_loader = create_dataloaders(df, tokenizer, batch_size=batch_size, max_len=max_len)
                                                            
    optimizer = Adam(bert_model.parameters(), lr=1e-3)


    train_losses = []
    train_accuracies = []
    val_accuracies = []

    print('Starting training...')
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        
        train_loss, train_acc = train_bert(bert_model, train_loader, optimizer, device)
        val_acc = evaluate_bert(bert_model, val_loader, device)
        
        # Append metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}')


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a BERT model for sentiment analysis.')
    parser.add_argument('json_file_path', type=str, help='Path to the JSON file containing the dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    parser.add_argument('--n_epochs', type=int, default=8, help='Number of epochs to train the model.')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length for BERT.')

    args = parser.parse_args()
    
    main(args.json_file_path, args.batch_size, args.n_epochs, args.max_len)

