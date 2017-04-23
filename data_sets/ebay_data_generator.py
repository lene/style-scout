from collections import OrderedDict
from operator import itemgetter

# from keras.preprocessing.image import ImageDataGenerator

from data_sets import add_border
from PIL import Image
import numpy
from os.path import isfile, join

from items import Items


class EbayDataGenerator:

    DEPTH = 3

    def __init__(self, items, valid_labels, size, verbose=False):
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
        self.verbose = verbose

    def generate_arrays(self):
        for i, item in enumerate(self.items):
            if self.verbose:
                print('Extracting images: {}/{}'.format(i+1, len(self.items)), end='\r')
            item.download_images(verbose=self.verbose)
            for image_file in item.picture_files:
                try:
                    image = Image.open(join(image_file)).convert('RGB')
                except OSError:
                    continue
                yield numpy.asarray(self.downscale(image, method=add_border)), self._dense_to_one_hot(item.tags)
            if self.verbose:
                print()
            # yield ({'input_1': x1, 'input_2': x2}, {'output': y})

    def __length_hint__(self):
        return sum(len(item.picture_files) for item in self.items)

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

    def _extract_images(self):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        import os.path
        images, labels = [], []
        for i, item in enumerate(self.items):
            if self.verbose:
                print('Extracting images: {}/{}'.format(i+1, len(self.items)), end='\r')
            item.download_images(verbose=False)
            for image_file in item.picture_files:
                try:
                    image = Image.open(os.path.join(image_file)).convert('RGB')
                except OSError:
                    continue
                images.append(numpy.asarray(self.downscale(image, method=add_border)))
                labels.append(tuple(item.tags))
        if self.verbose:
            print()

        return numpy.asarray(images), numpy.asarray(labels)

    def downscale(self, image, method=add_border):
        w, h = image.size
        image = method(image, w, h)
        return image.resize(self.size, Image.BICUBIC)

    def _dense_to_one_hot(self, label):
        labels_one_hot = numpy.zeros(self.num_classes)
        for tag in label:
            labels_one_hot[self.labels_to_numbers[tag]] = 1
        return labels_one_hot

    @classmethod
    def _npz_file_name(cls, data_file):
        if len(data_file) < 5 or data_file[-4:] != '.npz':
            data_file += '.npz'
        return data_file

    @classmethod
    def _create_from_file(cls, data_file, image_size, items, valid_labels, verbose=False):
        if verbose:
            print('Loading ' + data_file)
        npz = numpy.load(data_file)
        return cls(
            items, valid_labels, (image_size, image_size), 0, extract=False,
            train_images=npz['train_images'], train_labels=npz['train_labels'],
            test_images=npz['test_images'], test_labels=npz['test_labels'],
            validation_images=npz['validation_images'], validation_labels=npz['validation_labels']
        )

    @classmethod
    def _save_to_file(cls, data, data_file, verbose=False):
        if verbose:
            print('Storing ' + data_file)
        numpy.savez_compressed(
            data_file,
            train_images=data.train.input, train_labels=data.train.labels,
            test_images=data.test.input, test_labels=data.test.labels,
            validation_images=data.validation.input, validation_labels=data.validation.labels
        )


def _check_constructor_arguments_valid(items, size, depth):
    assert isinstance(items, Items)
    assert isinstance(size, tuple)
    assert len(size) == 2
    assert isinstance(size[0], int)
    assert isinstance(size[1], int)
