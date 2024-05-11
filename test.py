from evaluation import Evaluation
from data import read_data_file, extract_tags
from tag_transition import (
    tagCount_dictionaries,
    calculate_transition_probabilities,
    create_tag_transition_matrix,
)

data = read_data_file("/Users/ipinmi/Desktop/pos/dataset/dev.col")

"""
#First Matrix test for evaluation
y_true = ['cat', 'dog', 'pig', 'cat', 'dog', 'pig']
y_pred = ['cat', 'pig', 'dog', 'cat', 'cat', 'dog']
evaluating = Evaluation()
matrix = evaluating.multiclass_confusion_matrix(y_pred, y_true)
print("matrix", matrix)

precision, recall, f_score = evaluating.precision_recall_fScore(y_pred, y_true, beta = 1, averagingType = "micro")
print("precision", precision, "recall", recall, "f_score", f_score)"""


""" 
sentence_tags, y_pred = extract_tags(data)
_, y_true = extract_tags(data)
print(data)
print("y_pred", y_pred)
print("y_true", y_true) 
print("sentence_tags", sentence_tags)


test_eval = Evaluation(y_pred, y_true)
conf_matrix = test_eval.multiclass_confusion_matrix()
# print("conf_matrix", conf_matrix)

precision, recall, f_score = test_eval.precision_recall_fScore(
    beta=1, averagingType="micro"
)
print("precision:", precision)
print("recall: ", recall)
print("f_score: ", f_score)
 """

tag_transition_counts, tag_counts = tagCount_dictionaries(data)
tag_transitions_prob, all_tags = calculate_transition_probabilities(
    tag_transition_counts, tag_counts
)

tag_index = list(tag_transitions_prob.keys()).index("<s>")
print("tag_index", tag_index)

transition_matrix = create_tag_transition_matrix(tag_transition_counts, tag_counts, 0)
print("transition_matrix", transition_matrix[tag_index, :])

# print("tag_prob", tag_transitions_prob)
# print("all_tags", all_tags)
