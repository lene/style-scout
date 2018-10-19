import random
from os.path import join
from typing import Tuple, Set, Generator, List, Dict, Sized, Optional

import numpy
from PIL import Image

from acquisition.items import Items
from data_sets.contains_images import add_border, ContainsImages
from data_sets.labeled_items import LabeledItems
from utils.with_verbose import WithVerbose

Batch = List[Tuple[Set[str], str]]
Batches = List[Batch]


class BatchGenerator(Sized):

    def __init__(self, chunks: Batch, batch_size: int) -> None:
        self.chunks = chunks
        self.batch_size = batch_size
        self.batches = self.generate_batches()

    def generate_batches(self) -> Batches:
        random.shuffle(self.chunks)
        return [
            self.chunks[i:i + self.batch_size] for i in range(0, len(self.chunks), self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self.batches)


class EbayDataGenerator(LabeledItems, WithVerbose, ContainsImages):
    """
    Returns the image data and labels for a data set in batches (of configurable size) instead of
    keeping them all in memory at once.
    """

    CACHE_FILE_PREFIX = 'style_scout'

    def __init__(
            self, items: Items, valid_labels: Dict[str, int], size: Tuple[int, int],
            test_share: float=0.2, batch_size: int=32, random_seed: int=None, verbose: bool=False
    ) -> None:
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
        self._setup_batches(test_share, random_seed)

    def _setup_batches(self, test_share: float, random_seed: Optional[int]) -> None:
        self.items.download_images()
        chunks = [
            (item.tags, picture_file) for item in self.items for picture_file in item.picture_files
        ]
        random.seed(random_seed)
        self.train = BatchGenerator(chunks[:int(len(chunks) * (1 - test_share))], self.batch_size)
        self.test = BatchGenerator(chunks[int(len(chunks) * (1 - test_share)):], self.batch_size)

    def train_length(self) -> int:
        return len(self.train)

    def test_length(self) -> int:
        return len(self.test)

    def train_generator(self) -> Generator:
        """
        Generator function returning all images and their labels used as training set
        """
        if not self.train_length():
            raise ValueError("Length of training set is 0")
        while True:
            for i in range(self.train_length()):
                yield (
                    self.images_for_batch(self.train.batches, i),
                    self.labels_for_batch(self.train.batches, i)
                )
            self.train.generate_batches()

    def test_generator(self) -> Generator:
        """
        Generator function returning all images and their labels in the test set
        """
        while True:
            for i in range(self.test_length()):
                yield (
                    self.images_for_batch(self.test.batches, i),
                    self.labels_for_batch(self.test.batches, i)
                )

    def images_for_batch(self, batches: Batches, batch_index: int) -> numpy.ndarray:
        """
        :param batch_index: index of the batch (0 <= batch_index <= len(self)
        :return: image data for batch number batch_index
        """
        return numpy.asarray([
            self.downscale(Image.open(join(data_point[1])).convert('RGB'), method=add_border)
            for data_point in batches[batch_index]
        ])

    def labels_for_batch(self, batches: Batches, batch_index: int) -> numpy.ndarray:
        """
        :param batch_index: index of the batch (0 <= batch_index <= len(self)
        :return: labels for batch number batch_index
        """
        return numpy.asarray(
            [
                self._dense_to_one_hot(data_point[0])
                for data_point in batches[batch_index]
            ]
        )

    def _dense_to_one_hot(self, label: Set[str]) -> numpy.ndarray:
        labels_one_hot = numpy.zeros(self.num_classes)
        for tag in label:
            labels_one_hot[self.labels_to_numbers[tag]] = 1
        return labels_one_hot


def _check_constructor_arguments_valid(items: Items, size: Tuple[int, int], depth: int) -> None:
    assert isinstance(items, Items), 'items argument needs to be an Items object'
    assert isinstance(size, tuple), 'size argument needs to be a tuple of the form (width, height)'
    assert len(size) == 2, 'size argument needs to be a tuple of the form (width, height)'
    assert isinstance(size[0], int), 'size argument needs to be a tuple of the form (width, height)'
    assert isinstance(size[1], int), 'size argument needs to be a tuple of the form (width, height)'
