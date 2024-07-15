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
N_EPOCHS = 3
DROPOUT = 0.15
MODEL_SAVE_PATH_TEMPLATE = 'bert-pos-tagger-model-{}.pt'

# Define paths to datasets
DATA_PATHS = {
    'mandarin': {
        'train': 'C:/tagTeam/zh_gsd-ud-train.txt',
        'valid': 'C:/tagTeam/zh_gsd-ud-dev.txt',
        'test': 'C:/tagTeam/zh_gsd-ud-test.txt'
    },
    'english': {
        'train': 'C:/tagTeam/en_gum-ud-train.txt',
        'valid': 'C:/tagTeam/en_gum-ud-dev.txt',
        'test': 'C:/tagTeam/en_gum-ud-test.txt'
    },
    'german': {
        'train': 'C:/tagTeam/de_gsd-ud-train.txt',
        'valid': 'C:/tagTeam/de_gsd-ud-dev.txt',
        'test': 'C:/tagTeam/de_gsd-ud-test.txt'
    },
    'afrikaan': {
        'train': 'C:/tagTeam/af_afribooms-ud-train.txt',
        'valid': 'C:/tagTeam/af_afribooms-ud-dev.txt',
        'test': 'C:/tagTeam/af_afribooms-ud-test.txt'
    }
}

# Define BERT models and tokenizers for each language
BERT_MODELS = {
    'mandarin': 'bert-base-chinese',
    'english': 'bert-base-uncased',
    'german': 'bert-base-german-cased',
    'afrikaan': 'bert-base-multilingual-cased'
}

# Preprocessing functions
def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    """
    Cut the tokens to the maximum input length and convert them to IDs using the tokenizer.
    
    Parameters:
    tokens (list): List of tokens.
    tokenizer (BertTokenizer): BERT tokenizer.
    max_input_length (int): Maximum input length.
    
    Returns:
    list: List of token IDs.
    """
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens

def cut_to_max_length(tokens, max_input_length):
    """
    Cut the tokens to the maximum input length.
    
    Parameters:
    tokens (list): List of tokens.
    max_input_length (int): Maximum input length.
    
    Returns:
    list: List of cut tokens.
    """
    tokens = tokens[:max_input_length-1]
    return tokens

class LoadDataset(Dataset):
    def __init__(self, path, fields, **kwargs):
        """
        Custom dataset class to handle loading of data.
        
        Parameters:
        path (str): Path to the dataset file.
        fields (list): List of fields.
        """
        examples = self.load_data(path, fields)
        super().__init__(examples, fields, **kwargs)

    def load_data(self, path, fields):
        """
        Load data from the dataset file.
        
        Parameters:
        path (str): Path to the dataset file.
        fields (list): List of fields.
        
        Returns:
        list: List of examples.
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

class BERTPoSTagger(nn.Module):
    def __init__(self, bert, output_dim, dropout):
        """
        Define the BERT-based POS tagger model.
        
        Parameters:
        bert (BertModel): Pre-trained BERT model.
        output_dim (int): Output dimension.
        dropout (float): Dropout rate.
        """
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, attention_mask):
        """
        Forward pass of the model.
        
        Parameters:
        text (torch.Tensor): Input text tensor.
        attention_mask (torch.Tensor): Attention mask tensor.
        
        Returns:
        torch.Tensor: Output predictions.
        """
        text = text.permute(1, 0)
        attention_mask = attention_mask.permute(1, 0)
        embedded = self.dropout(self.bert(text, attention_mask=attention_mask)[0])
        embedded = embedded.permute(1, 0, 2)
        predictions = self.fc(self.dropout(embedded))
        return predictions

def calculate_f1(preds, y, tag_pad_idx):
    """
    Calculate the F1 score.
    
    Parameters:
    preds (torch.Tensor): Predictions.
    y (torch.Tensor): Ground truth labels.
    tag_pad_idx (int): Padding index for tags.
    
    Returns:
    float: F1 score.
    """
    max_preds = preds.argmax(dim=1, keepdim=True)
    non_pad_elements = (y != tag_pad_idx).nonzero(as_tuple=True)
    y_pred = max_preds[non_pad_elements].squeeze(1).cpu().numpy()
    y_true = y[non_pad_elements].cpu().numpy()
    return f1_score(y_true, y_pred, average='weighted')

def train(model, iterator, optimizer, criterion, tag_pad_idx, pad_token_idx, print_every=10):
    """
    Train the model for one epoch.
    
    Parameters:
    model (nn.Module): The model to train.
    iterator (BucketIterator): Data iterator.
    optimizer (torch.optim.Optimizer): Optimizer.
    criterion (nn.Module): Loss function.
    tag_pad_idx (int): Padding index for tags.
    pad_token_idx (int): Padding index for tokens.
    print_every (int): Print progress every specified number of batches.
    
    Returns:
    float: Average loss for the epoch.
    float: Average F1 score for the epoch.
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
            print(f'Step: {i+1:04} | Step Loss: {loss.item():.3f} | Step F1: {f1:.2f}')
    
    return epoch_loss / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx, pad_token_idx):
    """
    Evaluate the model.
    
    Parameters:
    model (nn.Module): The model to evaluate.
    iterator (BucketIterator): Data iterator.
    criterion (nn.Module): Loss function.
    tag_pad_idx (int): Padding index for tags.
    pad_token_idx (int): Padding index for tokens.
    
    Returns:
    float: Average loss.
    float: Average F1 score.
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

def epoch_time(start_time, end_time):
    """
    Calculate elapsed time in minutes and seconds.
    
    Parameters:
    start_time (float): Start time.
    end_time (float): End time.
    
    Returns:
    int: Elapsed minutes.
    int: Elapsed seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def write_final_f1_to_file(file_path, language, final_f1, test_f1):
    """
    Write final F1 score to a text file.
    
    Parameters:
    file_path (str): File path to write the scores.
    language (str): Language of the model.
    final_f1 (float): Final validation F1 score.
    test_f1 (float): Test F1 score.
    """
    with open(file_path, 'a') as f:
        f.write(f'Language: {language}\n')
        f.write(f'Final Validation F1 Score: {final_f1:.4f}\n')
        f.write(f'Test F1 Score: {test_f1:.4f}\n\n')

def main(language):
    """
    Train and evaluate the model for a given language.
    
    Parameters:
    language (str): Language to train and evaluate the model.
    """
    assert language in BERT_MODELS, f"Unsupported language: {language}"

    tokenizer = BertTokenizer.from_pretrained(BERT_MODELS[language])

    init_token = tokenizer.cls_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token

    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

    max_input_length = tokenizer.model_max_length

    text_preprocessor = functools.partial(cut_and_convert_to_id,
                                          tokenizer=tokenizer,
                                          max_input_length=max_input_length)

    tag_preprocessor = functools.partial(cut_to_max_length,
                                         max_input_length=max_input_length)

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

    data_paths = DATA_PATHS[language]
    train_data = LoadDataset(data_paths['train'], fields)
    valid_data = LoadDataset(data_paths['valid'], fields)
    test_data = LoadDataset(data_paths['test'], fields)

    UD_TAGS.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def sort_key(ex):
        return len(ex.text)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device,
        sort_key=sort_key)

    bert = BertModel.from_pretrained(BERT_MODELS[language])

    OUTPUT_DIM = len(UD_TAGS.vocab)
    model = BERTPoSTagger(bert, OUTPUT_DIM, DROPOUT)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    train_losses, valid_losses, test_losses = [], [], []
    train_f1s, valid_f1s, test_f1s = [], [], []

    f1_results_file = 'final_f1_scores.txt'

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_f1 = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX, pad_token_idx)
        valid_loss, valid_f1 = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX, pad_token_idx)
        test_loss, test_f1 = evaluate(model, test_iterator, criterion, TAG_PAD_IDX, pad_token_idx)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)
        train_f1s.append(train_f1)
        valid_f1s.append(valid_f1)
        test_f1s.append(test_f1)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH_TEMPLATE.format(language))

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train F1: {train_f1:.2f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid F1: {valid_f1:.2f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test F1: {test_f1:.2f}')
    
    final_f1 = valid_f1s[-1]
    write_final_f1_to_file(f1_results_file, language, final_f1, test_f1)

    epochs = range(1, N_EPOCHS + 1)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Losses')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_f1s, label='Train F1')
    plt.plot(epochs, valid_f1s, label='Validation F1')
    plt.plot(epochs, test_f1s, label='Test F1')
    plt.title('F1 Scores')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.plot(epochs, train_f1s, label='Train F1')
    plt.plot(epochs, valid_f1s, label='Validation F1')
    plt.plot(epochs, test_f1s, label='Test F1')
    plt.title('Losses and F1 Scores')
    plt.legend()

    plt.show()

# Select the language for training
language = 'afrikaan'  # Choose from 'mandarin', 'english', 'german', 'afrikaan'
main(language)
