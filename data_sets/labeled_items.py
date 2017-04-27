from collections import OrderedDict
from operator import itemgetter

from acquisition.items import Items


class LabeledItems:
    """
    Common properties for data used as input for neural network. These include items and labels for these 
    items. 
    """

    def __init__(self, items, valid_labels):
        """
        :param items: List of Item objects or an Items object 
        :param valid_labels: Set of labels the network will be trained against
        """
        _check_constructor_arguments(items, valid_labels)
        self.items = items
        self.valid_labels = tuple(valid_labels)
        self.num_classes = len(valid_labels)
        self.labels_to_numbers = {label: i for i, label in enumerate(self.valid_labels)}
        self.numbers_to_labels = {v: k for k, v in self.labels_to_numbers.items()}

    def labels(self, predictions):
        """
        Converts predictions encoded as one-hot into human readable labels
        :param predictions: List of probabilities for each possible label
        :return: Dict with human-readable label as key and its probability as value
        """
        return {
            self.numbers_to_labels[index]: probability
            for index, probability in enumerate(predictions) if probability > 0
        }

    def labels_sorted_by_probability(self, predictions):
        """
        Converts predictions encoded as one-hot into human readable labels and sorts them by probability
        :param predictions: List of probabilities for each possible label
        :return: Dict with human-readable label as key and its probability as value, sorted by probability
        """
        pairs = sorted(
            [(label, probability) for label, probability in self.labels(predictions).items()],
            key=itemgetter(1), reverse=True
        )
        return OrderedDict(pairs)

    def _dense_to_one_hot(self, labels):
        raise NotImplementedError()


def _check_constructor_arguments(items, valid_labels):
    assert isinstance(items, Items) or isinstance(items, list)
