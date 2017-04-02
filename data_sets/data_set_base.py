import numpy

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class DataSetBase:

    def __init__(self, input, labels):
        _check_constructor_arguments_valid(input, labels)
        self._num_examples = input.shape[0]
        self._input = input
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def input(self):
        return self._input

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size <= self._num_examples
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            start = self._start_new_epoch(batch_size, start)
        end = self._index_in_epoch
        return self._input[start:end], self._labels[start:end]

    def _start_new_epoch(self, batch_size, start):
        # Finished epoch
        self._epochs_completed += 1
        # Shuffle the data
        self._shuffle_data()
        # Start next epoch
        self._index_in_epoch = batch_size
        return 0

    def _shuffle_data(self):
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._input = self._input[perm]
        self._labels = self._labels[perm]


def _check_constructor_arguments_valid(input, labels):
    assert isinstance(input, numpy.ndarray), \
        'input not of type numpy.ndarray, but ' + type(input).__name__
    assert isinstance(labels, numpy.ndarray), \
        'labels not of type numpy.ndarray, but ' + type(input).__name__
    # assert len(labels.shape) == 1, 'labels must have one dimension: number of labels '+str(labels.shape)
    assert input.shape[0] == labels.shape[0], \
        'number of input records: {} != number of labels: {}'.format(input.shape[0], labels.shape[0])
