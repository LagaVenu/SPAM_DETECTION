# Spam Mail Detection using BERT

This project implements a robust spam mail detection system using BERT (Bidirectional Encoder Representations from Transformers) for natural language processing. The system can effectively classify emails as spam or non-spam with high accuracy.

## Features

- BERT-based email classification
- Training and prediction scripts
- Email preprocessing and feature extraction
- High accuracy spam detection

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/bhargavak04/Spam_Detection_BERT_Enron.git
cd Spam_Mail_Robust
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## BERT Model

The project uses the BERT model from Hugging Face. You can download the pre-trained BERT model from:
```
https://huggingface.co/bert-base-uncased
```

The model will be automatically downloaded when you run the training script for the first time.

## Usage

### Training

To train the model:
```bash
python train_bert_spam.py
```

### Prediction

To predict whether an email is spam or not:
```bash
python predict_spam.py
```

## Project Structure

- `train_bert_spam.py`: Script for training the BERT model
- `predict_spam.py`: Script for making predictions
- `mail.py`: Core functionality for email processing
- `requirements.txt`: List of required Python packages
- `saved_model/`: Directory for storing trained models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the BERT implementation
- Transformers library for making BERT accessible 