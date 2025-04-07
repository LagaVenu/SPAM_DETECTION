import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load and prepare dataset
def load_data(file_path='spam.csv'):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['v1', 'v2']]  # Use columns: label (v1) and text (v2)
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# 2. Create PyTorch Dataset
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
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

# 3. Training function
def train_model():
    # Load data
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

    # Create data loaders
    MAX_LEN = 128
    BATCH_SIZE = 16

    train_dataset = EmailDataset(
        texts=train_df.text.to_numpy(),
        labels=train_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_length=MAX_LEN
    )
    val_dataset = EmailDataset(
        texts=val_df.text.to_numpy(),
        labels=val_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_length=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Training setup
    EPOCHS = 3
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Training loop
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        print(f'Training loss: {avg_train_loss}')

    # Save model and tokenizer
    os.makedirs('saved_model', exist_ok=True)
    model.save_pretrained('saved_model')
    tokenizer.save_pretrained('saved_model')
    print('Model and tokenizer saved to saved_model/ directory')

if __name__ == '__main__':
    train_model()