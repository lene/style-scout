from collections import OrderedDict
from operator import itemgetter

from data_sets import add_border
from PIL import Image
import numpy
from os.path import join

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
        for item in self.items:
            item.download_images(verbose=False)

    def __len__(self):
        return sum(len(item.picture_files) for item in self.items)

    def train_generator(self):
        while True:
            for i, item in enumerate(self.items):
                images = []
                labels = []
                for image_file in item.picture_files:
                    try:
                        image = Image.open(join(image_file)).convert('RGB')
                        image = self.downscale(image, method=add_border)
                        images.append(numpy.asarray(image))
                        labels.append(self._dense_to_one_hot(item.tags))
                    except OSError:
                        continue
                images = numpy.asarray(images)
                yield images.reshape((len(images), self.size[0], self.size[1], self.DEPTH)), \
                    numpy.asarray(labels).reshape(len(images), self.num_classes)

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
    assert isinstance(items, Items), 'items argument needs to be an Items object'
    assert isinstance(size, tuple), 'size argument needs to be a tuple of the form (width, height)'
    assert len(size) == 2, 'size argument needs to be a tuple of the form (width, height)'
    assert isinstance(size[0], int), 'size argument needs to be a tuple of the form (width, height)'
    assert isinstance(size[1], int), 'size argument needs to be a tuple of the form (width, height)'
