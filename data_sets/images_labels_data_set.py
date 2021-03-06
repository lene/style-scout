from typing import Tuple, Sized

import numpy

from data_sets.data_set_base import DataSetBase

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class ImagesLabelsDataSet(DataSetBase, Sized):

    def __init__(
            self, images: numpy.ndarray, labels: numpy.ndarray, depth: int=1, reshape: bool=False
    ) -> None:
        """Construct a DataSet.

        Args:
          images: 4D numpy.ndarray of shape (num images, image height, image width, image depth)
          labels: 1D numpy.ndarray of shape (num images)
        """

        super().__init__(images, labels)

        # Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
        if reshape:
            images = images.reshape(images.shape[0], depth * images.shape[1] * images.shape[2])
        images = normalize(images)
        self._input = images

    def __len__(self) -> int:
        return self._input.shape[0]


def normalize(ndarray: numpy.ndarray) -> numpy.ndarray:
    """Transform a ndarray that contains uint8 values to floats between 0. and 1.

    :param ndarray:
    :return:
    """
    assert isinstance(ndarray, numpy.ndarray)
    if ndarray.dtype == numpy.uint8:
        return numpy.multiply(ndarray.astype(numpy.float32), 1.0 / 255.0)
    return ndarray
