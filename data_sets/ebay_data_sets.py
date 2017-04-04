from data_sets import ImageFileDataSets, add_border
from data_sets.data_sets import DataSets
from data_sets.images_labels_data_set import ImagesLabelsDataSet
from PIL import Image
import numpy
from os.path import isfile
from subprocess import call
from dill import dump, load, HIGHEST_PROTOCOL
from gzip import open as gzopen


class EbayDataSets(ImageFileDataSets):

    @classmethod
    def get_data(cls, data_file, items, valid_labels, image_size):
        if data_file is not None and isfile(data_file):
            print('Loading ' + data_file)
            my_open = gzopen if '.gz' == data_file[-3:] else open
            with my_open(data_file, 'rb') as file:
                return load(file)
        else:
            data = EbayDataSets(items, valid_labels, image_size, image_size, 0)

            my_data_file = data_file[:-3] if '.gz' == data_file[-3:] else data_file
            with open(my_data_file, 'wb') as file:
                dump(data, file, protocol=HIGHEST_PROTOCOL)
            if '.gz' == data_file[-3:]:
                call(('gzip', my_data_file))
            return data

    def __init__(self, items, valid_labels, x_size, y_size, validation_share=None):
        """Construct the data set from images stored in subdirs under base_dir
        :param base_dir: Where to store the MNIST data files.
        :param x_size:
        :param y_size:
        :param validation_share:
        """
        self.one_hot = True
        self.items = items
        self.size = (x_size, y_size)
        self.num_features = x_size*y_size*self.depth
        self.valid_labels = tuple(valid_labels)
        self.num_classes = len(valid_labels)
        self.labels_to_numbers = {label: i for i, label in enumerate(self.valid_labels)}

        all_images, all_labels = self._extract_images()

        all_labels = self._dense_to_one_hot(all_labels)
        self.numbers_to_labels = {v: k for k, v in self.labels_to_numbers.items()}

        train_images, train_labels, test_images, test_labels = self.split_images(all_images, all_labels, 0.8)

        self.validation_size = int(len(all_images)*(self.DEFAULT_VALIDATION_SHARE if validation_share is None else validation_share))
        validation_images = train_images[:self.validation_size]
        validation_labels = train_labels[:self.validation_size]
        train_images = train_images[self.validation_size:]
        train_labels = train_labels[self.validation_size:]

        DataSets.__init__(
            self,
            ImagesLabelsDataSet(train_images, train_labels, self.depth),
            ImagesLabelsDataSet(validation_images, validation_labels, self.depth),
            ImagesLabelsDataSet(test_images, test_labels, self.depth)
        )

    def _extract_images(self):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        import os.path
        images, labels = [], []
        for i, item in enumerate(self.items):
            print('Extracting images: {}/{}'.format(i, len(self.items)), end='\r')
            item.download_images(verbose=False)
            try:
                for image_file in item.picture_files:
                    try:
                        image = Image.open(os.path.join(image_file)).convert('RGB')
                    except OSError:
                        continue
                    images.append(numpy.asarray(self.downscale(image, method=add_border)))
                    labels.append(tuple(item.tags))
            except AttributeError:
                continue
        print()

        return numpy.asarray(images), numpy.asarray(labels)

    def _dense_to_one_hot(self, labels):
        labels_one_hot = numpy.zeros((len(labels), self.num_classes))
        for i, tags in enumerate(labels):
            for tag in tags:
                labels_one_hot[i][self.labels_to_numbers[tag]] = 1
        return labels_one_hot
        index_offset = numpy.arange(len(labels)) * self.num_classes
        labels_one_hot.flat[index_offset + self.labels_to_numbers.ravel()] = 1

