import torch
from transformers import BertTokenizer, BertForSequenceClassification
from train_bert_spam import EmailDataset  # Reuse dataset class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpamDetector:
    def __init__(self, model_path='saved_model'):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.max_len = 128

    def predict(self, email_text):
        encoding = self.tokenizer.encode_plus(
            email_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction].item()

        return prediction.item() == 1, confidence

# Create a global instance of the detector
detector = SpamDetector()

def predict_spam(text):
    """
    Function that app.py expects to use for spam detection.
    Returns a tuple of (is_spam, confidence)
    """
    return detector.predict(text)

if __name__ == '__main__':
    # Test examples
    test_emails = [
        "Congratulations! You've won a free iPhone! Click here to claim now!",  # Spam
        "Hi John, can we schedule a meeting for tomorrow at 3 PM?",  # Ham
        "URGENT: Your bank account has been compromised. Verify your details immediately.",  # Spam
        "Thanks for your email. I'll get back to you by EOD."  # Ham
    ]

    for email in test_emails:
        print(f"Email: {email[:60]}...")
        is_spam, confidence = predict_spam(email)
        print(f"Prediction: {'spam' if is_spam else 'ham'} (confidence: {confidence:.2f})")
        print("-" * 50)