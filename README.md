# tagTeam

Computational Linguistics Team Laboratory Project on Part-Of-Speech Tagging.

## Team Members

- [Chibundum Adebayo]
- [Hao-En Hsu]

## Project Milestones

- [x] Project Setup
- [x] Evaluation implementation for POS tagging
- [x] Baseline POS tagger algorithm

## Project Setup

1. Libraries needed:
   - Python 3.6
   - Numpy
2. Module Structure:
   - data.py: Reading data files in the CoNLL format and extracting tokens and tags for evaluation purposes
   - evaluation.py: Evaluating the performance of the POS tagger using metrics including precision, recall, and F1-score
   - tag_transition.py: Tag transition module
   - emission.py: Constructing the emission matrix, which contains the probabilities of emitting each word from each POS tag
   - matrices.py: Building transition and emission matrices from the training data and saving them to files
   - hmm.py: Implementing the Viterbi algorithm for POS tagging using HMM.
   - pred_eval.py: Predicting POS tags for sentences and evaluating the predicted tags against the gold standard tags
   - main.py: Executing the HMM POS tagger and evaluateing its performance
3. Run the following command to execute the program and get the evaluation results as well as the predictions for the appropriate data:
   `python3 main.py`

## Evaluation Results

**Evaluation results for the dev data:**

- Dev data F1 Score: **[0.93]**
- Test data F1 Score: **[0.94]**
