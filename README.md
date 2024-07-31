## BERT POS tagger

- [x] German
- [x] English
- [x] Mandarin
- [x] Afrikaan

**Evaluation results (F1 score) for the dev and test data:**

- German: dev = **0.9388** ; test = **0.9200**
- English: dev = **0.9176** ; test = **0.9175**
- Mandarin: dev = **0.7900** ; test = **0.7957**
- Afrikaan: dev = **0.8825** ; test = **0.8999**

## Transfer Learning

Using pretrained POS taggers of three languages: German, English, and Mandarin, and finetuning on Afrikaan datasets.
- [x] Afrikaan


**Evaluation Results (F1 score) for the dev and test data pretrained on different languages:**

- German: dev = **0.8407** ; test = **0.8525**
- English: dev = **0.8396** ; test = **0.8585**
- Mandarin: dev = **0.8520** ; test = **0.8600**

As the results seem really similar, we also tried to run the pretrained models and finetune on different amount of data (500, 1000, 3000, 5000 tokens). Results are shown in the following figure:
![image](https://github.com/user-attachments/assets/e8dd3c06-d712-4a70-b3cb-3627c96f8e7d)


## Project Setup

> **IMPORTANT: Please adjust the file paths in `BERTtagger.py (Lines 32-34, 37-39, 42-44, 47-49)` and `TL_Afrikaan.py (Lines 28-30)`, the language choice for BERT taggers in the `BERTtagger.py (Line 425)` and `TL_Afrikaan.py (Line 33)`, and pretrained model paths in `TL_Afrikaan.py (Lines 36, 38, 40)` to the appropriate paths on your local machine before running the program. Also, tune the hyperparameters in `BERTtagger.py (Lines 23-26)` and in `TL_Afrikaan.py (Lines 23-26)`**

```python
python3 -m venv transformers
source transformers/bin/activate

pip install -r requirements.txt

# How to run the program and generates the evaluation results and predictions
For TL_Afrikaan.py, choose the language in line 33 and For BERTtagger.py, choose the language in line 425, and run the file.
```

> - The evaluation results for BERT taggers are stored in the **results_BERTtagger.txt** file.
> - The evaluation results for Transfer learning are stored in the **results_TL.txt** file.

1. Overview:
   This project extension implements three BERT POS taggers in three languages: German, English, and Mandarin. Then the three BERT POS tagger models are saved as pretrained models later used in the transfer learning fine tuned on Afrikaan datasets.

2. Module Structure:
   - BERTtagger.py: Building, evaluating, and saving three different pretrained models
   - TL_Afrikaan.py: Loading three pretrained models and finetuning on Afrikaan datasets. 
