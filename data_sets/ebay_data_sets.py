from os.path import isfile
from typing import List, Tuple, Dict, Set

import numpy
from PIL import Image

from acquisition.items import Items
from data_sets.image_file_data_sets import ImageFileDataSets
from data_sets.contains_images import ContainsImages, add_border
from data_sets.data_sets import DataSets
from data_sets.images_labels_data_set import ImagesLabelsDataSet
from data_sets.labeled_items import LabeledItems
from utils.with_verbose import WithVerbose


class EbayDataSets(ImageFileDataSets, LabeledItems, WithVerbose):

    @classmethod
    def get_data(
            cls, data_file: str, items: Items, valid_labels: Dict[str, int], image_size: int,
            test_share: float=0.2, verbose: bool=False
    ) -> ImageFileDataSets:
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
        data_file = EbayDataSets._npz_file_name(data_file)
        if data_file is not None and isfile(data_file):
            data = EbayDataSets._create_from_file(data_file, image_size, items, valid_labels)
        else:
            data = EbayDataSets.extract_and_init(
                items, valid_labels, (image_size, image_size), 0, test_share=test_share,
                verbose=verbose
            )
            cls._save_to_file(data, data_file)
        return data

    @classmethod
    def extract_and_init(
            cls, items: Items, valid_labels: Dict[str, int], size: Tuple[int, int],
            validation_share: float=None, test_share: float=0.2, verbose: bool=False
    ) -> 'EbayDataSets':
        all_images, all_labels = cls._extract_images(items, size, verbose)
        required_ram = all_images.size * (4 + 1) + all_labels.nbytes
        WithVerbose.print_status(
            verbose,
            'RAM needed for images and labels: {0:.2f}GB'.format(required_ram / 1024 / 1024 / 1024)
        )

        all_labels = cls._dense_to_one_hot(all_labels)

        train_images, train_labels, test_images, test_labels = self.split_images(
            all_images, all_labels, 1 - test_share
        )

        validation_size = int(
            len(all_images) * (cls.DEFAULT_VALIDATION_SHARE if validation_share is None else validation_share)
        )
        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]
        return cls(
            items, valid_labels, size,
            train_images, train_labels, test_images, test_labels, validation_images, validation_labels,
            verbose
        )

    def __init__(
            self, items: Items, valid_labels: Dict[str, int], size: Tuple[int, int],
            train_images: numpy.ndarray, train_labels: numpy.ndarray,
            test_images: numpy.ndarray, test_labels: numpy.ndarray,
            validation_images: numpy.ndarray, validation_labels: numpy.ndarray,
            verbose: bool
    ) -> None:
        """
        Construct the data set from images belonging to items passed in
        :param items: Items object corresponding to the data set
        :param valid_labels: Labels corresponding to the labels of the data set
        :param size: tuple(width, height): Size the images are scaled to
        :param train_images: Image data to be used as features for the training set
        :param train_labels: Labels to be used as labels for the training set
        :param test_images: Image data to be used as features for the test set
        :param test_labels: Labels to be used as labels for the test set
        :param validation_images: Image data to be used as features for the validation set
        :param validation_labels: Labels to be used as labels for the validation set
        :param verbose: If set, print status/progress information
        """
        _check_constructor_arguments_valid(
            size, self.DEPTH,
            train_images, train_labels, test_images, test_labels, validation_images, validation_labels
        )
        LabeledItems.__init__(self, items, valid_labels)
        ContainsImages.__init__(self, *size)
        WithVerbose.__init__(self, verbose)

        self.validation_size = len(validation_images)

        DataSets.__init__(
            self,
            ImagesLabelsDataSet(train_images, train_labels, self.DEPTH),
            ImagesLabelsDataSet(validation_images, validation_labels, self.DEPTH),
            ImagesLabelsDataSet(test_images, test_labels, self.DEPTH)
        )

    @classmethod
    def _extract_images(
            cls, items: Items, size: Tuple[int, int], verbose: bool, *args: str
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        import os.path
        images, labels = [], []
        for i, item in enumerate(items):
            WithVerbose.print_status(
                verbose, 'Extracting images: {}/{}'.format(i + 1, len(items)), end='\r'
            )
            item.download_images()
            for image_file in item.picture_files:
                try:
                    image = Image.open(os.path.join(image_file)).convert('RGB')
                except OSError:
                    continue
                images.append(numpy.asarray(ContainsImages.scale_image(image, size, method=add_border)))
                labels.append(tuple(item.tags))

        WithVerbose.print_status(verbose)

        return numpy.asarray(images), numpy.asarray(labels)

    def _dense_to_one_hot(self, labels: Set[str]) -> numpy.ndarray:
        labels_one_hot = numpy.zeros((len(labels), self.num_classes))
        for i, tags in enumerate(labels):
            for tag in tags:
                labels_one_hot[i][self.labels_to_numbers[tag]] = 1
        return labels_one_hot

    @classmethod
    def _npz_file_name(cls, data_file: str) -> str:
        if len(data_file) < 5 or data_file[-4:] != '.npz':
            data_file += '.npz'
        return data_file

    @classmethod
    def _create_from_file(
            cls, data_file: str, image_size: int, items: Items, valid_labels: Dict[str, int],
            verbose: bool=False
    ) -> 'EbayDataSets':
        if verbose:
            print('Loading ' + data_file)
        npz = numpy.load(data_file)
        return cls(
            items, valid_labels, (image_size, image_size),
            train_images=npz['train_images'], train_labels=npz['train_labels'],
            test_images=npz['test_images'], test_labels=npz['test_labels'],
            validation_images=npz['validation_images'], validation_labels=npz['validation_labels']
        )

    @classmethod
    def _save_to_file(cls, data: 'EbayDataSets', data_file: str, verbose: bool=False) -> None:
        if verbose:
            print('Storing ' + data_file)
        numpy.savez_compressed(
            data_file,
            train_images=data.train.input, train_labels=data.train.labels,
            test_images=data.test.input, test_labels=data.test.labels,
            validation_images=data.validation.input, validation_labels=data.validation.labels
        )


def _check_constructor_arguments_valid(
        size: Tuple[int, int], depth: int,
        train_images: numpy.ndarray, train_labels: numpy.ndarray,
        test_images: numpy.ndarray, test_labels: numpy.ndarray,
        validation_images: numpy.ndarray, validation_labels: numpy.ndarray
) -> None:
    assert isinstance(size, tuple)
    assert len(size) == 2
    assert isinstance(size[0], int)
    assert isinstance(size[1], int)

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
    assert len(test_labels.shape) == 2
    assert len(validation_labels.shape) == 2
