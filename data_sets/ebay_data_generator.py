from functools import wraps
from os.path import join, isfile

from PIL import Image
import numpy

from items import Items
from data_sets import add_border
from data_sets.labeled_items import LabeledItems


def batch_cache(images_generator):
    """
    Decorator checking whether the requested batch image data are already stored in a cache file. If so, 
    returns the requested data from the cache, otherwise generates them and writes them to the cache file.
    """
    @wraps(images_generator)
    def _impl(self, batch_index):
        if isfile(self.cache_file(batch_index)):
            npz = numpy.load(self.cache_file(batch_index))
            return npz['images']
        else:
            images = images_generator(self, batch_index)
            numpy.savez_compressed(self.cache_file(batch_index), images=images)
            return images
    return _impl


class EbayDataGenerator(LabeledItems):
    """
    Returns the image data and labels for a data set in batches (of configurable size) instead of keeping them
    all in memory at once.
    """

    DEPTH = 3
    CACHE_FILE_PREFIX = 'style_scout'

    def __init__(self, items, valid_labels, size, batch_size=32, cache_dir='/tmp', verbose=False):
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

        self.size = size
        self.num_features = size[0]*size[1]*self.DEPTH
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
        """
        Generator function returning all images and their labels used as training set
        """
        while True:
            for i in range(len(self.batches)):
                yield self.images_for_batch(i), self.labels_for_batch(i)

    @batch_cache
    def images_for_batch(self, batch_index):
        """
        :param batch_index: index of the batch (0 <= batch_index <= len(self)
        :return: image data for batch number batch_index
        """
        return numpy.asarray([
            self.downscale(Image.open(join(data_point[1])).convert('RGB'), method=add_border)
            for data_point in self.batches[batch_index]
        ])

    def labels_for_batch(self, batch_index):
        """
        :param batch_index: index of the batch (0 <= batch_index <= len(self)
        :return: labels for batch number batch_index
        """
        return numpy.asarray(
            [self._dense_to_one_hot(data_point[0]) for data_point in self.batches[batch_index]]
        )

    def cache_file(self, batch_index):
        """
        :param batch_index: index of the batch (0 <= batch_index <= len(self)
        :return: name of the file storing precomputed image data for batch number batch_index
        """
        return join(
            self.cache_dir,
            '{}_{:05d}_{:03d}_{:04d}.npz'.format(
                self.CACHE_FILE_PREFIX, len(self.items), self.size[0], batch_index
            )
        )

    def downscale(self, image, method=add_border):
        w, h = image.size
        image = method(image, w, h)
        return numpy.asarray(image.resize(self.size, Image.BICUBIC))

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
