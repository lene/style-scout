from os import walk
from os.path import isfile
from subprocess import call
from pickle import dump, load
from gzip import open as gzopen

from PIL import Image
import numpy

from data_sets.data_sets import DataSets
from data_sets.images_labels_data_set import ImagesLabelsDataSet
from data_sets.contains_images import ContainsImages
from utils.with_verbose import WithVerbose

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

"""DataSets for RGB images read from files using the directory name as label."""

IMAGENET_SIZE = 299


class ImageFileDataSets(DataSets, ContainsImages, WithVerbose):
    """Data sets (training, validation and test data) containing RGB image files."""

    DEFAULT_VALIDATION_SHARE = 0.2

    @classmethod
    def get_data(cls, data_file=None, image_directory=None, image_size=IMAGENET_SIZE) -> DataSets:
        if data_file is not None and isfile(data_file):
            print('Loading ' + data_file)
            with gzopen(data_file, 'rb') as file:
                return load(file)
        else:
            data = ImageFileDataSets(image_directory, image_size, image_size, 0, True)
            try:
                with gzopen(data_file, 'wb') as file:
                    dump(data, file)
            except OverflowError:  # annoying python bug when using gzopen with data > 4GB
                uncompressed_file = '.'.join(data_file.split('.')[:-1])
                with open(uncompressed_file, 'wb') as file:
                    dump(data, file)
                call(('gzip', uncompressed_file))
            return data

    def __init__(self, base_dir, x_size, y_size, validation_share=None, one_hot=False, verbose=False):
        """Construct the data set from images stored in subdirs under base_dir
        :param base_dir: Where to store the MNIST data files.
        :param x_size:
        :param y_size:
        :param validation_share:
        :param one_hot:
        """
        ContainsImages.__init__(self, x_size, y_size)
        WithVerbose.__init__(self, verbose)
        self.one_hot = one_hot
        self.base_dir = base_dir

        all_images, all_labels = self._extract_images(base_dir)

        self.num_labels = len(set(all_labels))
        if one_hot:
            all_labels, self.labels_to_numbers = _dense_to_one_hot(all_labels)
            self.numbers_to_labels = {v: k for k, v in self.labels_to_numbers.items()}

        train_images, train_labels, test_images, test_labels = self.split_images(all_images, all_labels, 0.8)

        self.validation_size = int(
            len(all_images) * (
                self.DEFAULT_VALIDATION_SHARE if validation_share is None else validation_share
            )
        )
        validation_images = train_images[:self.validation_size]
        validation_labels = train_labels[:self.validation_size]
        train_images = train_images[self.validation_size:]
        train_labels = train_labels[self.validation_size:]

        super().__init__(
            ImagesLabelsDataSet(train_images, train_labels, self.DEPTH),
            ImagesLabelsDataSet(validation_images, validation_labels, self.DEPTH),
            ImagesLabelsDataSet(test_images, test_labels, self.DEPTH)
        )
        self._print_status('Image data loaded')

    @property
    def num_classes(self) -> int:
        return self.train.num_classes

    def get_label(self, number):
        try:
            return self.numbers_to_labels[number]
        except KeyError:
            raise KeyError('{} not in {}'.format(number, self.numbers_to_labels))

    ############################################################################

    def _extract_images(self, base_dir):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        import os.path
        self._print_status('Extracting', base_dir)
        images, labels = [], []
        all_dirs = list(walk(base_dir))
        i = 0
        for root, dirs, files in all_dirs:
            label = root.split('/')[-1]
            self._print_status(label, "%.2f%%" % (i / len(all_dirs) * 100))
            i += 1
            for j, file in enumerate(files):
                self._print_status(j, '/', len(files), end='\r')
                try:
                    image = Image.open(os.path.join(root, file)).convert('RGB')
                except OSError:
                    continue

                images.append(numpy.asarray(self.downscale(image)))
                labels.append(label)

        return numpy.asarray(images), numpy.asarray(labels)

    def split_images(self, images, labels, train_to_test_ratio):
        from random import shuffle
        assert len(images) == len(labels)
        assert 0 <= train_to_test_ratio <= 1
        # avoid shuffling images and labels in place
        images_ = images.copy()
        labels_ = labels.copy()
        test_size = int(len(images) * (1 - train_to_test_ratio))
        combined = list(zip(images, labels))
        shuffle(combined)
        images_[:], labels_[:] = zip(*combined)
        return images_[test_size:], labels_[test_size:], images_[:test_size], labels_[:test_size]

    def prediction_info(self, prediction, place):
        index, value = nth_index_and_value(prediction, place)
        label = self.get_label(index)
        return index, label, value


def _dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    num_classes = len(set(labels_dense))
    num_labels = labels_dense.shape[0]
    labels_to_numbers = {label: i for i, label in enumerate(list(set(labels_dense)))}
    labels_as_numbers = numpy.asarray([labels_to_numbers[label] for label in labels_dense])

    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_as_numbers.ravel()] = 1
    return labels_one_hot, labels_to_numbers


def nth_index_and_value(l, n):
    v = sorted(l)[-n]
    i = list(l).index(v)
    return i, v
