#### @Author: Chibundum Adebayo for BiLSTM-CRF Model Implementation

# TagTeam

Computational Linguistics Team Laboratory Project on Part-Of-Speech Tagging.

## Project Setup for BiLSTM-CRF Model

```python
python3 -m venv lstm_crf
source lstm_crf/bin/activate
pip install -r requirements.txt

# With Conda Environment
conda create --name lstm_crf --file requirements.txt
conda activate lstm_crf
```

## Steps to collect the Universal Dependencies data and run the BiLSTM-CRF model

### 1. Download and create the data files for the four languages (English, German, Chinese and Afrikaans), run the following command:

```python
# Language codes for the four languages:
# English (en),
# German (de),
# Chinese (zh),
# Afrikaans (af)
python ud_conllu_convert.py --lang_code={language} --ud_data_dir=ud_pos_data
```

### 2. Download the embeddings for the four languages (English, German, Chinese and Afrikaans), run the following command:

```python
python fasttext_embed.py --lang_code={language}  --embedding_dir=embeddings
```

### 3. Set experimental hyperparameters for the BiLSTM-CRF model in the `model_params.py` file

To train different ensemble models, modify the following hyperparameters with either `True` or `False`:

```python
self.use_pretrained = False # Set to True to use pretrained word embeddings
self.use_char = False # Set to True to use character-level embeddings
self.with_crf = False # Set to True to predict using CRF layer, if False, use Bilstm Softmax layer
```

### 4. Train and save the BiLSTM-CRF model for the four languages (English, German, Chinese and Afrikaans) run the following command:

```python
python train.py --lang_code={language} --data_dir=ud_pos_data --embedding_dir=embeddings --model_path=model
```

## Evaluation Results for BiLSTM-CRF Model

**Evaluation results for the Train,Test and dev data:**

**Micro Averaging F1-Score:**

|           | Train | Dev | Test |
| --------- | ----- | --- | ---- |
| German    |       |     |      |
| English   |       |     |      |
| Chinese   |       |     |      |
| Afrikaans |       |     |      |

**Negative Likelihood Loss:**
| | Train | Dev | Test |
| --------- | ----- | --- | ---- |
| German | | | |
| English | | | |
| Chinese | | | |
| Afrikaans | | | |

## Project Milestones for TagTeam

- [x] Project Setup
- [x] Evaluation implementation for POS tagging
- [x] Baseline POS tagger algorithm **(Hidden Markov Model)**
- [x] Comparison with Spacy POS tagger and dummy POS tagger
- [x] Advanced approach with **BiLSTM-CRF** model
- [x] Advanced approach with **Transformer** model and Transfer Learning
- [x] Evaluation and comparison of the advanced models on four languages (English, German, Chinese and Afrikaans)
