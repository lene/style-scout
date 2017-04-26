from collections import OrderedDict
from operator import itemgetter

from items import Items


class LabeledItems:

    def __init__(self, items, valid_labels):
        _check_constructor_arguments(items, valid_labels)
        self.items = items
        self.valid_labels = tuple(valid_labels)
        self.num_classes = len(valid_labels)
        self.labels_to_numbers = {label: i for i, label in enumerate(self.valid_labels)}
        self.numbers_to_labels = {v: k for k, v in self.labels_to_numbers.items()}

    def labels(self, predictions):
        return {
            self.numbers_to_labels[index]: probability
            for index, probability in enumerate(predictions) if probability > 0
        }

    def labels_sorted_by_probability(self, predictions):
        pairs = sorted(
            [(label, probability) for label, probability in self.labels(predictions).items()],
            key=itemgetter(1), reverse=True
        )
        return OrderedDict(pairs)

    def _dense_to_one_hot(self, labels):
        raise NotImplementedError()


def _check_constructor_arguments(items, valid_labels):
    assert isinstance(items, Items) or isinstance(items, list)
