from evaluation import Evaluation


def read_data_file(file_name):
    """
    Reading the CONNLU format file in the following format:
    [(word1, tag1), (word2, tag2), ...] for each sentence.
    """
    with open(file_name, "r") as file:
        lines = file.readlines()

    data = []
    sentence = []
    for line in lines:
        if line == "\n":
            data.append(sentence)
            sentence = []
        else:
            word, tag = line.split()
            sentence.append((word, tag))

    return data


def extract_tags(data):
    """
    Extracting the tags from the data for evaluation in the following format:
    [tag1, tag2, ...] for each sentence.
    """
    sentence_tags = [[tag for _, tag in sentence] for sentence in data]

    # Flatten the list of lists
    tags = [tag for sentence in sentence_tags for tag in sentence]

    return tags


data = read_data_file("dummy.col")
y_pred = extract_tags(data)
y_true = extract_tags(data)

"""print(data)
print("y_pred", y_pred)
print("y_true", y_true)"""

test_eval = Evaluation(y_pred, y_true)
conf_matrix = test_eval.multiclass_confusion_matrix()
# print("conf_matrix", conf_matrix)

precision, recall, f_score = test_eval.precision_recall_fScore(
    beta=1, averagingType="micro"
)
print("precision:", precision)
print("recall: ", recall)
print("f_score: ", f_score)
