
from torch.utils.data import Dataset, DataLoader


import torch





from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm



class SpoilerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    

def create_dataloaders(df, tokenizer, batch_size=32, max_len=256):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['is_spoiler'])

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])


    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])


    train_dataset = SpoilerDataset(train_df['cleaned_review_text'].tolist(), train_df['label'].tolist(), tokenizer, max_len)
    val_dataset = SpoilerDataset(val_df['cleaned_review_text'].tolist(), val_df['label'].tolist(), tokenizer, max_len)
    test_dataset = SpoilerDataset(test_df['cleaned_review_text'].tolist(), test_df['label'].tolist(), tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def train_lstm(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0


    # Wrap the dataloader with tqdm for progress tracking
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        print('input_ids shape:', input_ids.shape)
        print('labels shape:', labels.shape)
       
        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

    return total_loss / len(dataloader), total_acc / len(dataloader)


# Evaluation function
def evaluate_lstm(model, dataloader, device):
    model.eval()
    total_acc = 0

    with torch.no_grad():
        # Wrap the dataloader with tqdm for progress tracking
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)

            total_acc += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

    return total_acc / len(dataloader)




def train_bert(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

        progress_bar.set_postfix(loss=total_loss / (len(dataloader)), acc=total_acc / (len(dataloader)))

    return total_loss / len(dataloader), total_acc / len(dataloader)

def evaluate_bert(model, dataloader, device):
    model.eval()
    total_acc = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

            total_acc += acc

            progress_bar.set_postfix(acc=total_acc / (len(dataloader)))

    return total_acc / len(dataloader)




def preprocess_t5_data(examples, tokenizer):
    # Format the inputs and labels for text-to-text task
    inputs = ["classify: " + text for text in examples['cleaned_review_text']]
    targets = [str(label) for label in examples['is_spoiler']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=2, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs