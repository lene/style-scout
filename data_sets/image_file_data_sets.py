from os import walk
from os.path import isfile
from subprocess import call
from pickle import dump, load
from gzip import open as gzopen

from PIL import Image
import numpy

from .data_sets import DataSets
from .images_labels_data_set import ImagesLabelsDataSet

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

"""DataSets for RGB images read from files using the directory name as label."""

IMAGENET_SIZE = 299


def crop_bottom(image, w, h):
    if w > h:
        image = image.crop(((w - h) / 2, 0, w - (w - h) / 2, h))
    elif h > w:
        image = image.crop((0, 0, w, w))
    return image


def add_border(image, w, h):
    if w == h:
        return image
    new_image = Image.new("RGB", (max(w, h), max(w, h)))
    new_image.paste(image, (0, 0))
    return new_image


class ImageFileDataSets(DataSets):
    """Data sets (training, validation and test data) containing RGB image files."""

    DEFAULT_VALIDATION_SHARE = 0.2
    depth = 3

    @classmethod
    def get_data(cls, data_file=None, image_directory=None, image_size=IMAGENET_SIZE):
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

    def __init__(self, base_dir, x_size, y_size, validation_share=None, one_hot=False):
        """Construct the data set from images stored in subdirs under base_dir
        :param base_dir: Where to store the MNIST data files.
        :param x_size:
        :param y_size:
        :param validation_share:
        :param one_hot:
        """
        self.one_hot = one_hot
        self.base_dir = base_dir
        self.size = (x_size, y_size)
        self.num_features = x_size*y_size*self.depth

        all_images, all_labels = self._extract_images(base_dir)

        self.num_labels = len(set(all_labels))
        if one_hot:
            all_labels, self.labels_to_numbers = _dense_to_one_hot(all_labels)
            self.numbers_to_labels = {v: k for k, v in self.labels_to_numbers.items()}

        train_images, train_labels, test_images, test_labels = self.split_images(all_images, all_labels, 0.8)

        self.validation_size = int(len(all_images)*(self.DEFAULT_VALIDATION_SHARE if validation_share is None else validation_share))
        validation_images = train_images[:self.validation_size]
        validation_labels = train_labels[:self.validation_size]
        train_images = train_images[self.validation_size:]
        train_labels = train_labels[self.validation_size:]

        super().__init__(
            ImagesLabelsDataSet(train_images, train_labels, self.depth),
            ImagesLabelsDataSet(validation_images, validation_labels, self.depth),
            ImagesLabelsDataSet(test_images, test_labels, self.depth)
        )

    def get_label(self, number):
        try:
            return self.numbers_to_labels[number]
        except KeyError:
            raise KeyError('{} not in {}'.format(number, self.numbers_to_labels))

    ############################################################################

    def _extract_image_from_url(self, url, labels):
        pass

    def _extract_images(self, base_dir):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        import os.path
        print('Extracting', base_dir)
        images, labels = [], []
        all_dirs = list(walk(base_dir))
        i = 0
        for root, dirs, files in all_dirs:
            label = root.split('/')[-1]
            print(label, "%.2f%%" % (i/len(all_dirs)*100))
            i += 1
            for j, file in enumerate(files):
                print(j, '/', len(files), end='\r')
                try:
                    image = Image.open(os.path.join(root, file)).convert('RGB')
                except OSError:
                    continue

                images.append(numpy.asarray(self.downscale(image)))
                labels.append(label)

        return numpy.asarray(images), numpy.asarray(labels)

    def downscale(self, image, method=crop_bottom):
        w, h = image.size
        image = method(image, w, h)
        return image.resize(self.size, Image.BICUBIC)

    @staticmethod
    def show_image(rgb_values, label=''):
        import matplotlib.pyplot as plt
        plt.imshow(rgb_values, cmap='gray')
        plt.title(label)
        plt.show()

    def split_images(self, images, labels, train_to_test_ratio):
        from random import shuffle
        test_size = int(len(images)*(1-train_to_test_ratio))
        combined = list(zip(images, labels))
        shuffle(combined)
        images[:], labels[:] = zip(*combined)
        return images[test_size:], labels[test_size:], images[:test_size], labels[:test_size]

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

