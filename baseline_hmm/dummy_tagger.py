# @Author: Chibundum Adebayo
import numpy as np
from data import read_data_file, extract_tokens_tags, tag_distribution
from evaluation import Evaluation

# Read the appropriate experiment file to get the gold standard tags
filename = "dataset/test.col"
gold_standard_data = read_data_file(filename)
sentences, _, gold_standard_tags = extract_tokens_tags(gold_standard_data)


# assign random tags to the tokens in each sentence
sentence_predicted_tags = []
for sentence in sentences:
    sentence_predicted_tags.append(np.random.choice(gold_standard_tags, len(sentence)))

all_predicted_tags = [tag for tags in sentence_predicted_tags for tag in tags]

prediction_distribution = tag_distribution(all_predicted_tags)
gold_standard_distribution = tag_distribution(gold_standard_tags)

# Evaluate the predicted tags against the gold standard tags
evaluator = Evaluation(all_predicted_tags, gold_standard_tags)
# Calculate precision, recall, and F1 score for all predicted tags
precision, recall, f1_score = evaluator.precision_recall_fScore()


# Write the evaluation results to a file
with open("random_tagging_evaluation_results.txt", "a") as file:
    if "test" in filename:
        file.write("Evaluation results for the test data:\n")
    else:
        file.write("Evaluation results for the dev data:\n")
    file.write(f"Gold Standard Tags Distribution: {gold_standard_distribution}\n")
    file.write(f"Predicted Tags Distribution: {prediction_distribution}\n")
    file.write(f"Precision {np.round(precision, 2)}\n")
    file.write(f"Recall: {np.round(recall,2)}\n")
    file.write(f"F1 Score: {np.round(f1_score,2)}\n")
    file.write("\n")
