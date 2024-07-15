## BERT POS tagger

- [x] German
- [x] English
- [x] Mandarin

## Evaluation Results

**Evaluation results (F1 score) for the dev and test data:**

- German (3 epochs): **0.9388** 
- English (10 epochs): **0.9176**
- Mandarin (10 epochs): **0.7900** ; **0.7957**

## Transfer Learning

Using pretrained POS taggers of three languages: German, English, and Mandarin, and finetuning on Afrikaan datasets.
- [x] Afrikaan


## Evaluation Results (F1 score) for the dev data pretrained on different languages:

- German (3 epochs): dev: **0.84** ; test: **0.85**
- English (6 epochs): dev: **0.84** ; test: **0.86**
- Mandarin (6 epochs): dev: **0.85** ; test: **0.86**

## Project Setup

> **IMPORTANT: Please adjust the file paths in `BERTtagger.py (Lines 30-32, 35-37, 40-42, 45-47)` and `TL_Afrikaan.py (Lines 26-28)`, the language choice for BERT taggers in the `BERTtagger.py (Line 417)` and `TL_Afrikaan.py (Line 31)`, and pretrained model paths in `TL_Afrikaan.py (Lines 34, 36, 38)` to the appropriate paths on your local machine before running the program. Also, tune the hyperparameters in `BERTtagger.py (Lines 21-24)` and in `TL_Afrikaan.py (Lines 21-24)`**

```python
python3 -m venv team_lab
source team_lab/bin/activate

* requirements.txt

# How to run the program and generates the evaluation results and predictions
For TL_Afrikaan.py, simply run the file. For BERTtagger.py, choose the language in line 417 and run the file.
```

> - The evaluation results for BERT taggers are stored in the **results_BERTtagger.txt** file.
> - The evaluation results for Transfer learning are stored in the **results_TL.txt** file.

1. Overview:
   This project extension implements three BERT POS taggers in three languages: German, English, and Mandarin. Then the three BERT POS tagger models are saved as pretrained models later used in the transfer learning fine tuned on Afrikaan datasets.

2. Module Structure:
   - BERTtagger.py: Building, evaluating, and saving three different pretrained models
   - TL_Afrikaan.py: Loading three pretrained models and finetuning on Afrikaan datasets. 
