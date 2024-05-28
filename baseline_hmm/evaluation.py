import numpy as np


class Evaluation:

    def __init__(self, y_pred, y_true):
        self.predicted = y_pred
        self.expected = y_true

        if len(self.predicted) != len(self.expected):
            raise ValueError("The result lists must be of the same length!")

        uniqueTag_Labels = sorted(set(self.expected + self.predicted))

        self._indices = {tag: i for (i, tag) in enumerate(uniqueTag_Labels)}

        # A list of all tags from both predicted and expected.
        self._values = uniqueTag_Labels
        #: A dictionary mapping values in tags to their indices.
        self._indices = self._indices

    def bin_count(self, dict_mapping: dict, source: list, length: int) -> list:
        """
        Creates a list equal to the number of tags and records the appropriate counts for the tags.

        Parameters:
            dict_mapping (Dict) : A dictionary mapping the tags to their respective indices.
            source (list) : A list containing the tags to be counted.
            length (int) : The number of tags to be counted.

        Returns:
            target (list) : A list containing the counts for each tag.
        """
        target = [0] * length
        for _, label_index in dict_mapping.items():
            target[label_index] = source.count(label_index)

        return target

    def multiclass_confusion_matrix(self):
        """
        Calculates the confusion matrix for multiclass classification.

        Returns:
            np.array : A numpy array representing the confusion matrix based on the predicted and expected values.
        """
        # Replace the tags with the appropriate indices, defaulting to the original list if no mapping found.
        y_pred_mapped = [*map(self._indices.get, self.predicted, self.predicted)]
        y_true_mapped = [*map(self._indices.get, self.expected, self.expected)]

        # Find all tags with true positives occurences
        tp_bins = [
            y_pred_mapped[i]
            for i in range(len(y_pred_mapped))
            if y_pred_mapped[i] == y_true_mapped[i]
        ]

        if len(tp_bins):
            # Update the number of TPs for each tag label
            tp_sum = self.bin_count(self._indices, tp_bins, len(self._values))

        else:
            # for the case where no tp value is found
            total_pred = total_expected = tp_sum = [0] * len(self._values)

        # Calculate the number of times each tag was predicted (total_pred = true positive + false positive)
        # and the number of times each tag is expected (total_expected = true positive + false negative)
        if len(y_true_mapped):
            total_pred = self.bin_count(self._indices, y_pred_mapped, len(self._values))
            total_expected = self.bin_count(
                self._indices, y_true_mapped, len(self._values)
            )

        tp = np.array(tp_sum)
        fp = np.array(total_pred) - np.array(tp_sum)
        fn = np.array(total_expected) - np.array(tp_sum)
        tn = np.array(y_pred_mapped).shape[0] - tp - fp - fn  # recheck

        return np.array([tp, fn, fp, tn]).T.reshape(-1, 2, 2)

    def precision_recall_fScore(self, beta=1, averagingType="micro"):
        """

        Parameters:
            predicted (list) : A list containing all the predicted tag sequences
            expected (list) : A list containing all the true (Gold standard) tags
            beta (int) : The beta value for the F score calculation
            averagingType (str) : The type of averaging to be used for the F score calculation

        Returns:
            precision (float) : The precision score for the given tag sequence
            recall (float) : The recall score for the given tag sequence
            f_score (float) : The F score for the given tag sequence

        """

        confusion_matrix = self.multiclass_confusion_matrix()

        true_pos = confusion_matrix[:, 0, 0]
        pred_array = true_pos + confusion_matrix[:, 1, 0]
        true_array = true_pos + confusion_matrix[:, 0, 1]

        if averagingType == "micro":
            tp_sum = np.array([true_pos.sum()])
            pred_sum = np.array([pred_array.sum()])
            true_sum = np.array([true_array.sum()])

            try:
                precision = tp_sum / pred_sum
                recall = tp_sum / true_sum

                if beta == float("inf"):
                    f_score = recall
                elif beta == 0:
                    f_score = precision
                else:
                    f_score = (
                        (1 + beta**2)
                        * (precision * recall)
                        / (beta**2 * precision + recall)
                    )
            except ZeroDivisionError:
                precision = recall = f_score = 0

        return precision, recall, f_score
