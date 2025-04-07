import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load and Prepare Dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['v1', 'v2']]  # Ham/Spam label and text
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Sample dataset (replace with your own)
# Dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = load_data('spam.csv')  # Replace with your dataset path

# 2. Split Data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# 3. Initialize BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 4. Create PyTorch Dataset
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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Parameters
MAX_LEN = 128
BATCH_SIZE = 16

# Create data loaders
def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = EmailDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_length=max_len
    )
    return DataLoader(dataset, batch_size=batch_size)

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# 5. Initialize BERT Model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification (spam/ham)
).to(device)

# 6. Training Setup
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# 7. Training Function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# 8. Evaluation Function
def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# 9. Train the Model
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(
        model, train_data_loader, loss_fn, optimizer, device, scheduler
    )
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(
        model, val_data_loader, loss_fn, device
    )
    
    print(f'Validation loss {val_loss} accuracy {val_acc}')
    print()

# 10. Test the Model
test_acc, test_loss = eval_model(model, test_data_loader, loss_fn, device)
print(f'Test accuracy: {test_acc}')

# 11. Save the Model
torch.save(model.state_dict(), 'bert_spam_detector.pth')

# 12. Prediction Function
def predict_spam(text, model, tokenizer, max_len=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    _, prediction = torch.max(logits, dim=1)
    
    return 'spam' if prediction == 1 else 'ham'

# Example usage
sample_email = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize now!"
print(predict_spam(sample_email, model, tokenizer))  # Should output "spam"

sample_email2 = "Hi John, just checking in to see if you're available for a meeting tomorrow at 2pm."
print(predict_spam(sample_email2, model, tokenizer))  # Should output "ham"