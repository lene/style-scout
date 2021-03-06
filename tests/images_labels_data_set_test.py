from typing import Iterable, Tuple

from data_sets.images_labels_data_set import ImagesLabelsDataSet, normalize

import numpy

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
# pylint: disable=missing-docstring

NUM_TRAINING_SAMPLES = 20
IMAGE_WIDTH = 12
IMAGE_HEIGHT = 10
BATCH_SIZE = 5


class ImagesLabelsDataSetTest(unittest.TestCase):

    def test_init_without_fake_data_runs(self) -> None:
        _create_empty_data_set()

    def test_init_length(self) -> None:
        images = create_empty_image_data()
        labels = create_empty_label_data()
        data = ImagesLabelsDataSet(images, labels)
        self.assertEquals(NUM_TRAINING_SAMPLES, len(data))

    def test_init_with_different_label_size_fails(self) -> None:
        images = create_empty_image_data()
        labels = create_empty_label_data_of_size(NUM_TRAINING_SAMPLES + 1)
        with self.assertRaises(AssertionError):
            ImagesLabelsDataSet(images, labels)

    def test_next_batch_returns_correct_data_format(self) -> None:
        data_set = _create_empty_data_set()
        images, labels = data_set.next_batch(BATCH_SIZE)
        self.assertIsInstance(images, numpy.ndarray)
        self.assertEqual(4, len(images.shape))
        self.assertEqual(BATCH_SIZE, images.shape[0])
        self.assertEqual(IMAGE_WIDTH, images.shape[1])
        self.assertEqual(IMAGE_HEIGHT, images.shape[2])
        self.assertEqual(1, images.shape[3])
        self.assertIsInstance(labels, numpy.ndarray)
        self.assertEqual(1, len(labels.shape))
        self.assertEqual(BATCH_SIZE, labels.shape[0])

    def test_next_batch_runs_repeatedly(self) -> None:
        data_set = _create_empty_data_set()
        batch_size = NUM_TRAINING_SAMPLES // 2
        _, _ = data_set.next_batch(batch_size)
        _, _ = data_set.next_batch(batch_size)

    def test_normalize_dtype(self) -> None:
        data = create_empty_image_data()
        normalized = normalize(data)
        self.assertEqual(normalized.dtype, numpy.float32)

    def test_normalize_range(self) -> None:
        data = create_random_image_data(0, 255)
        normalized = normalize(data)
        self.assertLessEqual(normalized.max(), 1.)
        self.assertGreaterEqual(normalized.min(), 0.)


def _create_empty_data_set() -> ImagesLabelsDataSet:
    images = create_empty_image_data()
    labels = create_empty_label_data()
    return ImagesLabelsDataSet(images, labels)


def create_empty_image_data() -> numpy.ndarray:
    return image_data_from_list([0] * NUM_TRAINING_SAMPLES * IMAGE_WIDTH * IMAGE_HEIGHT)


def create_random_image_data(min_val: int, max_val: int) -> numpy.ndarray:
    from random import randrange
    return image_data_from_list(
        [randrange(min_val, max_val + 1) for _ in range(NUM_TRAINING_SAMPLES * IMAGE_WIDTH * IMAGE_HEIGHT)]
    )


def image_data_from_list(buffer: Iterable) -> numpy.ndarray:
    data = numpy.fromiter(buffer, dtype=numpy.uint8)
    return data.reshape(NUM_TRAINING_SAMPLES, IMAGE_WIDTH, IMAGE_HEIGHT, 1)


def create_empty_label_data() -> numpy.ndarray:
    return create_empty_label_data_of_size(NUM_TRAINING_SAMPLES)


def create_empty_label_data_of_size(size: int) -> numpy.ndarray:
    return numpy.fromiter([0] * size, dtype=numpy.uint8)
