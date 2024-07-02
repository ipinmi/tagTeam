import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import functools
import time
import os
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel
from torchtext.data import Field, Example, Dataset, BucketIterator
from sklearn.metrics import f1_score

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
N_EPOCHS = 10
DROPOUT = 0.25
MODEL_SAVE_PATH = 'bert-pos-tagger-model.pt'
TRAIN_DATA_PATH = 'C:/tagTeam/de_gsd-ud-train.txt'
VALID_DATA_PATH = 'C:/tagTeam/de_gsd-ud-dev.txt'
TEST_DATA_PATH = 'C:/tagTeam/de_gsd-ud-test.txt'

# Load BERT tokenizer for German language
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

# Tokenizer special tokens
init_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

max_input_length = tokenizer.model_max_length

# Preprocessing functions
def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens

def cut_to_max_length(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens

# Partial functions for preprocessing
text_preprocessor = functools.partial(cut_and_convert_to_id,
                                      tokenizer=tokenizer,
                                      max_input_length=max_input_length)

tag_preprocessor = functools.partial(cut_to_max_length,
                                     max_input_length=max_input_length)

# Define fields for text and tags
TEXT = Field(use_vocab=False,
             lower=False,
             preprocessing=text_preprocessor,
             init_token=init_token_idx,
             pad_token=pad_token_idx,
             unk_token=unk_token_idx)

UD_TAGS = Field(unk_token=None,
                init_token='<pad>',
                preprocessing=tag_preprocessor)

fields = [("text", TEXT), ("udtags", UD_TAGS)]

# Custom dataset class to handle loading of data
class CustomDataset(Dataset):
    def __init__(self, path, fields, **kwargs):
        examples = self.load_data(path, fields)
        super().__init__(examples, fields, **kwargs)

    def load_data(self, path, fields):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                if line.strip() == '':
                    if words:
                        examples.append(Example.fromlist([words, tags], fields))
                        words, tags = [], []
                else:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        word, tag = parts
                        words.append(word)
                        tags.append(tag)
            if words:
                examples.append(Example.fromlist([words, tags], fields))
        return examples

# Load the data
train_data = CustomDataset(TRAIN_DATA_PATH, fields)
valid_data = CustomDataset(VALID_DATA_PATH, fields)
test_data = CustomDataset(TEST_DATA_PATH, fields)

# Build vocabulary for tags
UD_TAGS.build_vocab(train_data)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define sort key
def sort_key(ex):
    return len(ex.text)

# Create iterators for the datasets
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=sort_key)

# Define the model
class BERTPoSTagger(nn.Module):
    def __init__(self, bert, output_dim, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, attention_mask):
        text = text.permute(1, 0)
        attention_mask = attention_mask.permute(1, 0)
        embedded = self.dropout(self.bert(text, attention_mask=attention_mask)[0])
        embedded = embedded.permute(1, 0, 2)
        predictions = self.fc(self.dropout(embedded))
        return predictions

# Load pre-trained BERT model for German language
bert = BertModel.from_pretrained('bert-base-german-cased')

# Define model, optimizer, and loss function
OUTPUT_DIM = len(UD_TAGS.vocab)
model = BERTPoSTagger(bert, OUTPUT_DIM, DROPOUT)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)

# Move model and loss function to the device
model = model.to(device)
criterion = criterion.to(device)

# Helper function to calculate F1 score
def calculate_f1(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim=1, keepdim=True)
    non_pad_elements = (y != tag_pad_idx).nonzero(as_tuple=True)
    y_pred = max_preds[non_pad_elements].squeeze(1).cpu().numpy()
    y_true = y[non_pad_elements].cpu().numpy()
    return f1_score(y_true, y_pred, average='weighted')

# Training function
def train(model, iterator, optimizer, criterion, tag_pad_idx, print_every=10):
    epoch_loss = 0
    epoch_f1 = 0
    model.train()
    for i, batch in enumerate(iterator):
        text = batch.text
        tags = batch.udtags
        attention_mask = (text != pad_token_idx).long()
        optimizer.zero_grad()
        predictions = model(text, attention_mask)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        loss = criterion(predictions, tags)
        f1 = calculate_f1(predictions, tags, tag_pad_idx)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_f1 += f1
        
        if (i + 1) % print_every == 0:
            print(f'Step: {i+1:04} | Step Loss: {loss.item():.3f} | Step F1: {f1:.2f}')
    
    return epoch_loss / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_f1 = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.udtags
            attention_mask = (text != pad_token_idx).long()
            predictions = model(text, attention_mask)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            f1 = calculate_f1(predictions, tags, tag_pad_idx)
            epoch_loss += loss.item()
            epoch_f1 += f1
    return epoch_loss / len(iterator), epoch_f1 / len(iterator)

# Helper function to calculate elapsed time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')
train_losses, valid_losses = [], []
train_f1s, valid_f1s = [], []

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_f1 = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX, print_every=10)
    valid_loss, valid_f1 = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_f1s.append(train_f1)
    valid_f1s.append(valid_f1)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train F1: {train_f1:.2f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1:.2f}')

# Plotting loss and F1 score
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_f1s, label='Train F1 Score')
plt.plot(valid_f1s, label='Valid F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.show()
