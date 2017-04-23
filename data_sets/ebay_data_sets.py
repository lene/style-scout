from collections import OrderedDict
from operator import itemgetter

from data_sets import ImageFileDataSets, add_border
from data_sets.data_sets import DataSets
from data_sets.images_labels_data_set import ImagesLabelsDataSet
from PIL import Image
import numpy
from os.path import isfile


class EbayDataSets(ImageFileDataSets):

    @classmethod
    def get_data(cls, data_file, items, valid_labels, image_size, test_share=0.2, verbose=False):
        """
        Read an EbayDataSet from the given file name, if present; else create a new one and save it
        to that file name. In the end, both the data set and the savefile exist.
        :param data_file: file name to read from/save to
        :param items: Items object corresponding to the data set
        :param valid_labels: Labels corresponding to the labels of the data set
        :param image_size: Size the images are scaled to
        :param test_share: fraction of the data used as test data
        :param verbose: If set, print status/progress information
        :return: EbayDataSet read from the file or created from the passed parameters
        """
        data_file = cls._npz_file_name(data_file)
        if data_file is not None and isfile(data_file):
            data = cls._create_from_file(data_file, image_size, items, valid_labels)
        else:
            data = cls(
                items, valid_labels, (image_size, image_size), 0, extract=True, test_share=test_share,
                verbose=verbose
            )
            cls._save_to_file(data, data_file)
        return data

    def __init__(
            self, items, valid_labels, size, validation_share=None, test_share=0.2, extract=True,
            train_images=None, train_labels=None, test_images=None, test_labels=None,
            validation_images=None, validation_labels=None,
            verbose=False
    ):
        """
        Construct the data set from images belonging to items passed in
        TODO: finish this docstring
        :param test_share: fraction of the data used as test data
        :param validation_share:
        """
        _check_constructor_arguments_valid(
            extract, size, self.DEPTH,
            train_images, train_labels, test_images, test_labels, validation_images, validation_labels
        )

        self.items = items
        self.size = size
        self.num_features = size[0]*size[1]*self.DEPTH
        self.valid_labels = tuple(valid_labels)
        self.num_classes = len(valid_labels)
        self.labels_to_numbers = {label: i for i, label in enumerate(self.valid_labels)}
        self.numbers_to_labels = {v: k for k, v in self.labels_to_numbers.items()}
        self.verbose = verbose

        if extract:
            all_images, all_labels = self._extract_images()
            required_ram = all_images.size*(4+1)+all_labels.nbytes
            if self.verbose:
                print('RAM needed for images and labels: {0:.2f}GB'.format(required_ram/1024/1024/1024))

            all_labels = self._dense_to_one_hot(all_labels)
            if self.verbose:
                print('all labels as one-hot:\n', all_labels)

            train_images, train_labels, test_images, test_labels = self.split_images(
                all_images, all_labels, 1-test_share
            )

            self.validation_size = int(len(all_images) * (self.DEFAULT_VALIDATION_SHARE if validation_share is None else validation_share))
            validation_images = train_images[:self.validation_size]
            validation_labels = train_labels[:self.validation_size]
            train_images = train_images[self.validation_size:]
            train_labels = train_labels[self.validation_size:]
        else:
            self.validation_size = len(validation_images)

        DataSets.__init__(
            self,
            ImagesLabelsDataSet(train_images, train_labels, self.DEPTH),
            ImagesLabelsDataSet(validation_images, validation_labels, self.DEPTH),
            ImagesLabelsDataSet(test_images, test_labels, self.DEPTH)
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

    def _dense_to_one_hot(self, labels):
        labels_one_hot = numpy.zeros((len(labels), self.num_classes))
        for i, tags in enumerate(labels):
            for tag in tags:
                labels_one_hot[i][self.labels_to_numbers[tag]] = 1
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


def _check_constructor_arguments_valid(
        extract, size, depth, train_images, train_labels, test_images, test_labels,
        validation_images, validation_labels
):
    assert isinstance(size, tuple)
    assert len(size) == 2
    assert isinstance(size[0], int)
    assert isinstance(size[1], int)

    if extract:
        assert train_images is None
        assert train_labels is None
        assert test_images is None
        assert test_labels is None
        assert validation_images is None
        assert validation_labels is None
    else:
        # assert train_images.shape[1] == size[0]*size[1]*self.DEPTH, str(train_images.shape)
        assert len(train_images.shape) == 4
        assert train_images.shape[1] == size[0], str(train_images.shape)
        assert train_images.shape[2] == size[1], str(train_images.shape)
        assert train_images.shape[3] == depth, str(train_images.shape)
        # assert test_images.shape[1] == size[0]*size[1]*self.DEPTH, str(test_images.shape)
        assert len(test_images.shape) == 4
        assert test_images.shape[1] == size[0], str(test_images.shape)
        assert test_images.shape[2] == size[1], str(test_images.shape)
        assert test_images.shape[3] == depth, str(test_images.shape)
        # assert validation_images.shape[1] == size[0]*size[1]*self.DEPTH, str(validation_images.shape)
        assert len(validation_images.shape) == 4
        assert validation_images.shape[1] == size[0], str(validation_images.shape)
        assert validation_images.shape[2] == size[1], str(validation_images.shape)
        assert validation_images.shape[3] == depth, str(validation_images.shape)
        assert len(train_labels.shape) == 2
        # print(train_labels.shape)
        assert len(test_labels.shape) == 2
        # print(test_labels.shape)
        assert len(validation_labels.shape) == 2
        # print(validation_labels.shape)
