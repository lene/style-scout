from collections import OrderedDict
from operator import itemgetter
from functools import wraps
from os.path import join, isfile

from PIL import Image
import numpy

from items import Items
from data_sets import add_border


def batch_cache(method):
    @wraps(method)
    def _impl(self, batch_index):
        if isfile(self.cache_file(batch_index)):
            npz = numpy.load(self.cache_file(batch_index))
            return npz['images']
        else:
            images = method(self, batch_index)
            numpy.savez_compressed(self.cache_file(batch_index), images=images)
            return images
    return _impl


class EbayDataGenerator:

    DEPTH = 3

    def __init__(self, items, valid_labels, size, batch_size=32, cache_dir='/tmp', verbose=False):
        """
        Construct the data set from images belonging to items passed in
        TODO: finish this docstring
        :param test_share: fraction of the data used as test data
        :param validation_share:
        """
        _check_constructor_arguments_valid(items, size, self.DEPTH)

        self.items = items
        self.size = size
        self.num_features = size[0]*size[1]*self.DEPTH
        self.valid_labels = tuple(valid_labels)
        self.num_classes = len(valid_labels)
        self.labels_to_numbers = {label: i for i, label in enumerate(self.valid_labels)}
        self.numbers_to_labels = {v: k for k, v in self.labels_to_numbers.items()}
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.verbose = verbose
        for item in self.items:
            item.download_images(verbose=False)
        chunks = [(item.tags, picture_file) for item in self.items for picture_file in item.picture_files]
        self.batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]

    def __len__(self):
        return len(self.batches)

    def train_generator(self):
        while True:
            for i, batch in enumerate(self.batches):
                images = self.images_for_batch(i)
                labels = [self._dense_to_one_hot(data_point[0]) for data_point in batch]
                yield images.reshape((len(images), self.size[0], self.size[1], self.DEPTH)), \
                    numpy.asarray(labels).reshape(len(labels), self.num_classes)

    @batch_cache
    def images_for_batch(self, batch_index):
        images = numpy.asarray([
            numpy.asarray(
                self.downscale(Image.open(join(data_point[1])).convert('RGB'), method=add_border)
            ) for data_point in self.batches[batch_index]
        ])
        return images

    def cache_file(self, batch_index):
        return join(
            self.cache_dir,
            'style_scout_{:05d}_{:03d}_{:04d}.npz'.format(len(self.items), self.size[0], batch_index)
        )

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

    def downscale(self, image, method=add_border):
        w, h = image.size
        image = method(image, w, h)
        return image.resize(self.size, Image.BICUBIC)

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
