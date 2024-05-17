# tagTeam

Computational Linguistics Team Laboratory Project on Part-Of-Speech Tagging.

## Team Members

- [Chibundum Adebayo]
- [Hao-En Hsu]

## Project Milestones

- [x] Project Setup
- [x] Evaluation implementation for POS tagging
- [x] Baseline POS tagger algorithm **(Hidden Markov Model)**

## Evaluation Results

**Evaluation results for the Test and dev data:**

- Smoothing Parameter Value: **0.0001**
- Dev data F1 Score: **0.93**
- Test data F1 Score: **0.94**

## Project Setup

> :warning: **IMPORTANT: Please adjust the file paths in the `main.py (Line 18)` and `matrices.py (Line 25)` files to the appropriate paths on your local machine before running the program.**

1. Libraries Used:
   - Python (3.12.3)
   - Numpy (1.26.4)
2. Overview:
   This project implements a Hidden Markov Model (HMM) based Part-of-Speech (POS) tagger, which assigns a POS tag to each word in a given input sentence. The model utilizes probabilistic approaches to determine the most likely sequence of POS tags based on observed words and transition probabilities between tags, which is decoded by Viterbi algorithm, a dynamic approach frequently used in POS tagging and NER recognition.
3. Module Structure:
   - data.py: Reading data files in the CoNLL format and extracting tokens and tags for evaluation purposes
   - evaluation.py: Evaluating the performance of the POS tagger using metrics including precision, recall, and **micro averaging** F1-score
   - tag_transition.py: computing the transition probabilities between POS tags
   - emission.py: Constructing the emission matrix, which contains the probabilities of emitting each word from each POS tag
   - matrices.py: Building transition and emission matrices from the training data and saving them to files
   - hmm.py: Implementing the Viterbi algorithm for POS tagging using HMM.
   - pred_eval.py: Predicting POS tags for sentences and evaluating the predicted tags against the gold standard tags
   - main.py: Executing the HMM POS tagger and evaluateing its performance
4. Run the following command to execute the program and get the evaluation results as well as the predictions for the appropriate data:
   `python3 main.py`

<img src="assets/team_lab_project_structure.png" alt="image" height="500" width="600" >
