## BERT POS tagger

- [x] German
- [x] English
- [x] Mandarin

## Evaluation Results

**Evaluation results (F1 score) for the dev data:**

- German (3 epochs): **0.9388**
- English (10 epochs): **0.9176**
- Mandarin (10 epochs): **0.7900**

## Transfer Learning

- [x] Afrikaan


## Evaluation Results

**Evaluation results (F1 score) for the dev data pretrained on different languages:**

- German (10 epochs): **0.9388**
- English (10 epochs): **0.9176**
- Mandarin (10 epochs): **0.7900**

> **IMPORTANT: Please adjust the file paths in `BERTtagger.py (Lines 31-33, 36-38, 41-43)`, the language choice for BERT taggers in the `BERTtagger.py (Line 301)`, pretrained model paths in `TL_Afrikaan.py (Lines 209-211)` and finetune file paths in `TL_Afrikaan.py (Lines 26-28)`  to the appropriate paths on your local machine before running the program. Also, tune the hyperparameters in `BERTtagger.py (Lines 22-25)` and in `TL_Afrikaan.py (Lines 21-24)`**

```python
python3 -m venv team_lab
source team_lab/bin/activate

* requirements.txt

# How to run the program and generates the evaluation results and predictions
Simply run the two py files.
```

> - The evaluation results are stored in the **evaluation_results.txt** file.

1. Overview:
   This project extension implements three BERT POS taggers in three languages: German, English, and Mandarin. Then the three BERT POS tagger models are saved as pretrained models later used in the transfer learning fine tuned on Afrikaan datasets.

2. Module Structure:
   - BERTtagger.py: Building, evaluating, and saving three different pretrained models
   - TL_Afrikaan.py: Loading three pretrained models and finetuning on Afrikaan datasets. 
