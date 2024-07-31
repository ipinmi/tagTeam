# @author: Hao-En Hsu

import nltk
from nltk.tag import hmm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

def read_dataset(file_path):
    sentences = []
    sentence = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                word, tag = line.strip().split()
                sentence.append((word, tag))
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
        if sentence:
            sentences.append(sentence)
    
    return sentences

# Read the datasets
train_sentences = read_dataset('/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/BaselineData/train.col')
test_sentences = read_dataset('/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/BaselineData/test.col')
dev_sentences = read_dataset('/mount/arbeitsdaten65/studenten4/team-lab-cl/data2024/tagteam/BaselineData/dev.col')

# Combine training and development sentences for training
combined_train_sentences = train_sentences + dev_sentences

# Train HMM POS tagger
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(combined_train_sentences)

# Generate predictions for the test set
test_words = [word for sentence in test_sentences for word, tag in sentence]
test_tags = [tag for sentence in test_sentences for word, tag in sentence]
predicted_tags = [tag for sentence in test_sentences for word, tag in hmm_tagger.tag([word for word, tag in sentence])]

# Encode the tags to use with sklearn
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(test_tags)
predicted_labels_encoded = label_encoder.transform(predicted_tags)

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(true_labels_encoded, predicted_labels_encoded, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
