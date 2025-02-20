#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
import pandas as pd



import torch
import torch.nn as nn
from torch.optim import Adam

import matplotlib.pyplot as plt


from transformers import BertTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pipeline import create_dataloaders, train_lstm, evaluate_lstm

from LSTM import LSTMClassifier

import argparse


# In[ ]:


def main(json_file_path, batch_size=32, max_len=256, n_epochs=10):
    
    # Load the preprocessed data
    df = pd.read_json(json_file_path)
    print('Loaded data from', json_file_path)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader, val_loader, test_loader = create_dataloaders(df, tokenizer, batch_size=32, max_len=256)

    embedding_dim = 100
    hidden_dim = 128
    vocab_size = tokenizer.vocab_size
    output_dim = 2
    n_layers = 3
    bidirectional = True
    dropout = 0.5



    # Instantiate model, loss function, optimizer
    lstm_model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size, output_dim, n_layers, bidirectional, dropout)
    lstm_model = lstm_model.to(device)
    optimizer = Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()



    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in tqdm(range(n_epochs), desc='Training Epochs'):
        train_loss, train_acc = train_lstm(lstm_model, train_loader, optimizer, criterion, device)
        val_acc = evaluate_lstm(lstm_model, val_loader, device )
        
        # Append metrics to lists
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print epoch information
        tqdm.write(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}')
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'saves/lstm_model_epoch_{epoch+1}.pth'
            torch.save(lstm_model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved to {checkpoint_path}')

    # Evaluate on test set
    test_acc = evaluate_lstm(lstm_model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.4f}')


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM model for spoiler analysis')
    parser.add_argument('json_file_path', type=str, help='Path to the preprocessed JSON file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum sequence length for tokenization')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train the model')

    args = parser.parse_args()

    main(args.json_file_path, args.batch_size, args.max_len, args.n_epochs)





