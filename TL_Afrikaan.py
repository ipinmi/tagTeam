# @author: Hao-En Hsu

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import functools
import time
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
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
N_EPOCHS = 10
DROPOUT = 0.15

TRAIN_DATA_PATH = '/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/UDdata/UD_Afrikaans/af_afribooms-ud-train.txt'    
VALID_DATA_PATH = '/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/UDdata/UD_Afrikaans/af_afribooms-ud-dev.txt'  
TEST_DATA_PATH = '/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/UDdata/UD_Afrikaans/af_afribooms-ud-test.txt'  

# Language selection
language = 'german'  # Choose from 'mandarin', 'english', 'german'

if language == 'mandarin':
    model_path = '/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/pretrained models/bert-pos-tagger-model-mandarin.pt'
elif language == 'german':
    model_path = '/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/pretrained models/bert-pos-tagger-model-german.pt'
elif language == 'english':
    model_path = '/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/pretrained models/bert-pos-tagger-model-english.pt'
else:
    raise ValueError("Language not supported.")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenizer special tokens
init_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

max_input_length = tokenizer.model_max_length

def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    """
    Tokenizes a sequence of tokens and converts them into token IDs.
    
    Args:
    - tokens (list): List of tokens to be tokenized and converted.
    - tokenizer (BertTokenizer): Tokenizer object.
    - max_input_length (int): Maximum length of the input tokens.

    Returns:
    - tokens (list): List of token IDs.
    """
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens

def cut_to_max_length(tokens, max_input_length):
    """
    Truncates a sequence of tokens to a maximum length.
    
    Args:
    - tokens (list): List of tokens to be truncated.
    - max_input_length (int): Maximum length of the input tokens.

    Returns:
    - tokens (list): Truncated list of tokens.
    """
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

class LoadDataset(Dataset):
    def __init__(self, path, fields, **kwargs):
        """
        A custom Dataset class to load data from a file.

        Args:
        - path (str): Path to the data file.
        - fields (list): List of tuples specifying the fields in the data.
        - **kwargs: Additional arguments to pass to Dataset.
        """
        examples = self.load_data(path, fields)
        super().__init__(examples, fields, **kwargs)

    def load_data(self, path, fields):
        """
        Loads data from a file into Examples.

        Args:
        - path (str): Path to the data file.
        - fields (list): List of tuples specifying the fields in the data.

        Returns:
        - examples (list): List of Example objects.
        """
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
train_data = LoadDataset(TRAIN_DATA_PATH, fields)
valid_data = LoadDataset(VALID_DATA_PATH, fields)
test_data = LoadDataset(TEST_DATA_PATH, fields)

# Build vocabulary for tags
UD_TAGS.build_vocab(train_data)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define sort key
def sort_key(ex):
    """
    Function to return the length of text for sorting purposes.

    Args:
    - ex: Example object containing text field.

    Returns:
    - int: Length of the text.
    """
    return len(ex.text)

# Create iterators for the datasets
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=sort_key)

# Determine the output dimension based on the number of unique tags
OUTPUT_DIM = len(UD_TAGS.vocab)

class BERTPoSTagger(nn.Module):
    def __init__(self, bert, output_dim, dropout):
        """
        BERT-based Part-of-Speech Tagger model.

        Args:
        - bert (BertModel): Pre-trained BERT model.
        - output_dim (int): Dimension of the output (number of tags).
        - dropout (float): Dropout rate.
        """
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, attention_mask):
        """
        Forward pass of the model.

        Args:
        - text (torch.Tensor): Input token IDs.
        - attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
        - predictions (torch.Tensor): Predicted logits.
        """
        text = text.permute(1, 0)
        attention_mask = attention_mask.permute(1, 0)
        embedded = self.dropout(self.bert(text, attention_mask=attention_mask)[0])
        embedded = embedded.permute(1, 0, 2)
        predictions = self.fc(self.dropout(embedded))
        return predictions

    def resize_output_layer(self, new_output_dim):
        """
        Resizes the output layer of the model.

        Args:
        - new_output_dim (int): New dimension of the output layer.
        """
        self.fc = nn.Linear(self.fc.in_features, new_output_dim)

def train_model(model, iterator, optimizer, criterion, tag_pad_idx, print_every=10):
    """
    Function to train the BERT PoS Tagger model.

    Args:
    - model (BERTPoSTagger): The BERT PoS Tagger model instance.
    - iterator (BucketIterator): Data iterator for training data.
    - optimizer (torch.optim): Optimizer for training.
    - criterion (torch.nn): Loss criterion.
    - tag_pad_idx (int): Index of the padding token in the tag vocabulary.
    - print_every (int): Interval for printing training progress.

    Returns:
    - epoch_loss (float): Average loss per epoch.
    - epoch_f1 (float): Average F1 score per epoch.
    """
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
            print(f'Step: {i+1:04} | Step Loss: {loss.item():.3f} | Step F1: {f1:.4f}')

    return epoch_loss / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx):
    """
    Function to evaluate the BERT PoS Tagger model.

    Args:
    - model (BERTPoSTagger): The BERT PoS Tagger model instance.
    - iterator (BucketIterator): Data iterator for evaluation data.
    - criterion (torch.nn): Loss criterion.
    - tag_pad_idx (int): Index of the padding token in the tag vocabulary.

    Returns:
    - epoch_loss (float): Average loss per epoch.
    - epoch_f1 (float): Average F1 score per epoch.
    """
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

def calculate_f1(preds, y, tag_pad_idx):
    """
    Calculates the F1 score for evaluation.

    Args:
    - preds (torch.Tensor): Predicted logits.
    - y (torch.Tensor): True tags.
    - tag_pad_idx (int): Index of the padding token in the tag vocabulary.

    Returns:
    - micro F1 score.
    """
    max_preds = preds.argmax(dim=1, keepdim=True)
    non_pad_elements = (y != tag_pad_idx).nonzero(as_tuple=True)
    y_pred = max_preds[non_pad_elements].squeeze(1).cpu().numpy()
    y_true = y[non_pad_elements].cpu().numpy()
    return f1_score(y_true, y_pred, average='micro')

def epoch_time(start_time, end_time):
    """
    Calculates the elapsed time.

    Args:
    - start_time (float): Start time in seconds.
    - end_time (float): End time in seconds.

    Returns:
    - elapsed_mins (int): Elapsed minutes.
    - elapsed_secs (int): Elapsed seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def load_pretrained_model(model_path, output_dim, device):
    """
    Loads a pre-trained BERT PoS Tagger model.

    Args:
    - model_path (str): Path to the pre-trained model file.
    - output_dim (int): Dimension of the output (number of tags).
    - device (torch.device): Device to load the model onto.

    Returns:
    - model (BERTPoSTagger): Loaded BERT PoS Tagger model.
    """
    state_dict = torch.load(model_path)

    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    model = BERTPoSTagger(bert_model, output_dim, DROPOUT)

    model_dict = model.state_dict()

    # Adjust the size of the embedding layer
    bert_embeddings = state_dict['bert.embeddings.word_embeddings.weight']
    current_embeddings = model.bert.embeddings.word_embeddings.weight

    if bert_embeddings.shape[0] != current_embeddings.shape[0]:
        # Copy the overlapping weights
        new_embeddings = torch.cat(
            (bert_embeddings, current_embeddings[bert_embeddings.shape[0]:, :]), 
            dim=0
        )
        state_dict['bert.embeddings.word_embeddings.weight'] = new_embeddings

    # Update only the overlapping weights
    pretrained_dict = {k: v for k, v in state_dict.items() if k.startswith('bert.')}
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict, strict=False)

    model.resize_output_layer(output_dim)
    model.to(device)

    return model

def write_results_to_file(epoch, train_loss, valid_loss, valid_f1, test_f1, language, file_path='results_TL.txt'):
    """
    Writes training and evaluation results to a file.

    Args:
    - epoch (int): Current epoch number.
    - train_loss (float): Training loss.
    - valid_loss (float): Validation loss.
    - valid_f1 (float): Validation F1 score.
    - test_f1 (float): Test F1 score.
    - language (str): Language identifier.
    - file_path (str): File path to write results (default: 'results_TL.txt').
    """
    with open(file_path, 'a') as f:
        f.write(f'Language: {language}\n')
        f.write(f'Epoch: {epoch+1}\n')
        f.write(f'Train Loss: {train_loss:.3f}\n')
        f.write(f'Validation Loss: {valid_loss:.3f}\n')
        f.write(f'Validation F1: {valid_f1:.4f}\n')
        f.write(f'Test F1: {test_f1:.4f}\n\n')

def main(language):
    """
    Main function to train and evaluate the BERT PoS Tagger model.

    Args:
    - language (str): Language identifier.
    """
    model_path = f'C:/tagTeam/bert-pos-tagger-model-{language}.pt'
    model = load_pretrained_model(model_path, OUTPUT_DIM, device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=UD_TAGS.vocab.stoi[UD_TAGS.pad_token])

    train_losses = []
    valid_losses = []
    valid_f1s = []
    test_f1s = []

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, _ = train_model(model, train_iterator, optimizer, criterion, UD_TAGS.vocab.stoi[UD_TAGS.pad_token])
        valid_loss, valid_f1 = evaluate(model, valid_iterator, criterion, UD_TAGS.vocab.stoi[UD_TAGS.pad_token])
        test_loss, test_f1 = evaluate(model, test_iterator, criterion, UD_TAGS.vocab.stoi[UD_TAGS.pad_token])

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_f1s.append(valid_f1)
        test_f1s.append(test_f1)

        write_results_to_file(epoch, train_loss, valid_loss, valid_f1, test_f1, language)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{language}-model.pt')

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1:.4f}')
        print(f'\t Test F1: {test_f1:.4f}')

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss', color='tab:blue')
    plt.plot(valid_losses, label='Validation Loss', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Plot validation and test F1 scores
    plt.figure(figsize=(8, 6))
    plt.plot(valid_f1s, label='Validation F1', color='tab:green')
    plt.plot(test_f1s, label='Test F1', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation and Test F1 Scores')
    plt.show()

if __name__ == "__main__":
    main(language)
