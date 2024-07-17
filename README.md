## BERT POS tagger

- [x] German
- [x] English
- [x] Mandarin
- [x] Afrikaan

## Evaluation Results for BERT taggers

**Evaluation results (F1 score) for the dev and test data:**

- German: **0.9388** ; **0.9200**
- English: **0.9176** ; **0.9175**
- Mandarin: **0.7900** ; **0.7957**
- Afrikaan: **0.8825** ; **0.8999**

## Transfer Learning

Using pretrained POS taggers of three languages: German, English, and Mandarin, and finetuning on Afrikaan datasets.
- [x] Afrikaan


## Evaluation Results (F1 score) for the dev data pretrained on different languages:

- German: dev: **0.8407** ; test: **0.8525**
- English: dev: **0.8396** ; test: **0.8585**
- Mandarin: dev: **0.8520** ; test: **0.8600**

## Project Setup

> **IMPORTANT: Please adjust the file paths in `BERTtagger.py (Lines 32-34, 37-39, 42-44, 47-49)` and `TL_Afrikaan.py (Lines 28-30)`, the language choice for BERT taggers in the `BERTtagger.py (Line 419)` and `TL_Afrikaan.py (Line 33)`, and pretrained model paths in `TL_Afrikaan.py (Lines 36, 38, 40)` to the appropriate paths on your local machine before running the program. Also, tune the hyperparameters in `BERTtagger.py (Lines 23-26)` and in `TL_Afrikaan.py (Lines 23-26)`**

```python
python3 -m venv transformers
source transformers/bin/activate

pip install -r requirements.txt

# How to run the program and generates the evaluation results and predictions
For TL_Afrikaan.py, choose the language in line 33 and For BERTtagger.py, choose the language in line 419, and run the file.
```

> - The evaluation results for BERT taggers are stored in the **results_BERTtagger.txt** file.
> - The evaluation results for Transfer learning are stored in the **results_TL.txt** file.

1. Overview:
   This project extension implements three BERT POS taggers in three languages: German, English, and Mandarin. Then the three BERT POS tagger models are saved as pretrained models later used in the transfer learning fine tuned on Afrikaan datasets.

2. Module Structure:
   - BERTtagger.py: Building, evaluating, and saving three different pretrained models
   - TL_Afrikaan.py: Loading three pretrained models and finetuning on Afrikaan datasets. 
