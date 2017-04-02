import numpy

from data_sets.data_set_base import DataSetBase

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class ImagesLabelsDataSet(DataSetBase):

    def __init__(self, images, labels, depth=1):
        """Construct a DataSet.

        Args:
          images: 4D numpy.ndarray of shape (num images, image height, image width, image depth)
          labels: 1D numpy.ndarray of shape (num images)
        """

        _check_constructor_arguments_valid(images, labels)

        super().__init__(images, labels)

        # Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
        # TODO: assumes depth == 1
        images = images.reshape(images.shape[0], depth * images.shape[1] * images.shape[2])
        images = normalize(images)

        self._input = images

    def __len__(self):
        return self._input.shape(0)

def normalize(ndarray):
    """Transform a ndarray that contains uint8 values to floats between 0. and 1.

    :param ndarray:
    :return:
    """
    assert isinstance(ndarray, numpy.ndarray)
    assert ndarray.dtype == numpy.uint8

    return numpy.multiply(ndarray.astype(numpy.float32), 1.0/255.0)


def invert(ndarray):
    assert isinstance(ndarray, numpy.ndarray)
    assert ndarray.dtype == numpy.float32

    return numpy.subtract(1.0, ndarray)


def _check_constructor_arguments_valid(images, labels):
    assert len(images.shape) == 4, \
        'images must have 4 dimensions: number of images, image height, image width, color depth.' \
        'Actual shape: {}'.format(images.shape)
    # assert images.shape[3] == 1, 'image depth must be 1'

