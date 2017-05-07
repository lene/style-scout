from os.path import join
from random import shuffle

import numpy
from PIL import Image

from acquisition.items import Items
from data_sets import add_border
from data_sets.contains_images import ContainsImages
from data_sets.labeled_items import LabeledItems
from utils.with_verbose import WithVerbose


class BatchGenerator:

    def __init__(self, chunks, batch_size):
        self.chunks = chunks
        self.batch_size = batch_size
        self.batches = None
        self.generate_batches()

    def generate_batches(self):
        shuffle(self.chunks)
        self.batches = [
            self.chunks[i:i + self.batch_size] for i in range(0, len(self.chunks), self.batch_size)
        ]

    def __len__(self):
        return len(self.batches)


class EbayDataGenerator(LabeledItems, WithVerbose, ContainsImages):
    """
    Returns the image data and labels for a data set in batches (of configurable size) instead of keeping them
    all in memory at once.
    """

    CACHE_FILE_PREFIX = 'style_scout'

    def __init__(self, items, valid_labels, size, test_share=0.2, batch_size=32, verbose=False):
        """
        Construct the generator from images and labels belonging to items passed in
        :param items: Items object corresponding to the data set
        :param valid_labels: Labels corresponding to the labels of the data set
        :param size: tuple(width, height): Size the images are scaled to
        :param batch_size: The size of the batches returned by the generator function
        :param cache_dir: Where to store the precomputed batch data
        :param verbose: If set, print status/progress information
        """
        _check_constructor_arguments_valid(items, size, self.DEPTH)
        LabeledItems.__init__(self, items, valid_labels)
        ContainsImages.__init__(self, *size)
        WithVerbose.__init__(self, verbose)

        self.batch_size = batch_size
        self.num_items = len(items)
        self._setup_batches(test_share)

    def _setup_batches(self, test_share):
        self.items.download_images()
        chunks = [(item.tags, picture_file) for item in self.items for picture_file in item.picture_files]
        self.train = BatchGenerator(chunks[:int(len(chunks) * (1 - test_share))], self.batch_size)
        self.test = BatchGenerator(chunks[int(len(chunks) * (1 - test_share)):], self.batch_size)

    def train_length(self):
        return len(self.train)

    def test_length(self):
        return len(self.test)

    def train_generator(self):
        """
        Generator function returning all images and their labels used as training set
        """
        while True:
            for i in range(self.train_length()):
                yield (
                    self.images_for_batch(self.train.batches, i),
                    self.labels_for_batch(self.train.batches, i)
                )
            self.train.generate_batches()

    def test_generator(self):
        """
        Generator function returning all images and their labels in the test set
        """
        while True:
            for i in range(self.test_length()):
                yield self.images_for_batch(self.test.batches, i), self.labels_for_batch(self.test.batches, i)

    def images_for_batch(self, batches, batch_index):
        """
        :param batch_index: index of the batch (0 <= batch_index <= len(self)
        :return: image data for batch number batch_index
        """
        return numpy.asarray([
            self.downscale(Image.open(join(data_point[1])).convert('RGB'), method=add_border)
            for data_point in batches[batch_index]
        ])

    def labels_for_batch(self, batches, batch_index):
        """
        :param batch_index: index of the batch (0 <= batch_index <= len(self)
        :return: labels for batch number batch_index
        """
        return numpy.asarray(
            [self._dense_to_one_hot(data_point[0]) for data_point in batches[batch_index]]
        )

    def _dense_to_one_hot(self, label):
        labels_one_hot = numpy.zeros(self.num_classes)
        for tag in label:
            labels_one_hot[self.labels_to_numbers[tag]] = 1
        return labels_one_hot


def _check_constructor_arguments_valid(items, size, depth):
    assert isinstance(items, Items), 'items argument needs to be an Items object'
    assert isinstance(size, tuple), 'size argument needs to be a tuple of the form (width, height)'
    assert len(size) == 2, 'size argument needs to be a tuple of the form (width, height)'
    assert isinstance(size[0], int), 'size argument needs to be a tuple of the form (width, height)'
    assert isinstance(size[1], int), 'size argument needs to be a tuple of the form (width, height)'
