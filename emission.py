from data import read_data_file, extract_tags

data = read_data_file("/Users/ipinmi/Desktop/pos/dataset/train.col")

# Initialize dictionaries to store word-tag counts and tag counts
word_tag_counts = {}
tag_counts = {}

# Count occurrences of each word-tag pair and each tag
for sentence in data:
    for word, tag in sentence:
        if tag not in word_tag_counts:
            word_tag_counts[tag] = {}
        if word not in word_tag_counts[tag]:
            word_tag_counts[tag][word] = 0
        word_tag_counts[tag][word] += 1

        if tag not in tag_counts:
            tag_counts[tag] = 0
        tag_counts[tag] += 1

# Calculate probabilities for each word-tag pair
word_tag_prob = {}
for tag, word_counts in word_tag_counts.items():
    total_tag_count = tag_counts[tag]
    word_tag_prob[tag] = {}
    for word, count in word_counts.items():
        word_tag_prob[tag][word] = count / total_tag_count


def sort(probabilities):
    sorted_prob = dict(sorted(probabilities.items()))
    return sorted_prob


sorted_prob = sort(word_tag_prob)

# Print the sorted dictionary
for tag, probabilities in sorted_prob.items():
    print(f"{tag}: {probabilities}")
