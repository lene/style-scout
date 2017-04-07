from data_sets.images_labels_data_set import ImagesLabelsDataSet, normalize
# from .util import create_minimal_input_placeholder

import numpy

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
# pylint: disable=missing-docstring

NUM_TRAINING_SAMPLES = 20
IMAGE_WIDTH = 12
IMAGE_HEIGHT = 10
BATCH_SIZE = 5


class ImagesLabelsDataSetTest(unittest.TestCase):

    def test_init_without_fake_data_runs(self):
        _create_empty_data_set()

    def test_init_length(self):
        images = create_empty_image_data()
        labels = create_empty_label_data()
        data = ImagesLabelsDataSet(images, labels)
        self.assertEquals(NUM_TRAINING_SAMPLES, len(data))

    def test_init_with_different_label_size_fails(self):
        images = create_empty_image_data()
        labels = create_empty_label_data_of_size(NUM_TRAINING_SAMPLES + 1)
        with self.assertRaises(AssertionError):
            ImagesLabelsDataSet(images, labels)

    def test_next_batch_returns_correct_data_format(self):
        data_set = _create_empty_data_set(True)
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

    def test_next_batch_runs_repeatedly(self):
        data_set = _create_empty_data_set()
        batch_size = NUM_TRAINING_SAMPLES//2
        _, _ = data_set.next_batch(batch_size)
        _, _ = data_set.next_batch(batch_size)

    def test_normalize_dtype(self):
        data = create_empty_image_data()
        normalized = normalize(data)
        self.assertEqual(normalized.dtype, numpy.float32)

    def test_normalize_range(self):
        data = create_random_image_data(0, 255)
        normalized = normalize(data)
        self.assertLessEqual(normalized.max(), 1.)
        self.assertGreaterEqual(normalized.min(), 0.)


def _create_empty_data_set(reshape=None):
    images = create_empty_image_data()
    labels = create_empty_label_data()
    return ImagesLabelsDataSet(images, labels, reshape)


def create_empty_image_data():
    return image_data_from_list([0] * NUM_TRAINING_SAMPLES * IMAGE_WIDTH * IMAGE_HEIGHT)


def create_random_image_data(min_val, max_val):
    from random import randrange
    return image_data_from_list(
        [randrange(min_val, max_val+1) for _ in range(NUM_TRAINING_SAMPLES * IMAGE_WIDTH * IMAGE_HEIGHT)]
    )


def image_data_from_list(buf):
    data = numpy.fromiter(buf, dtype=numpy.uint8)
    return data.reshape(NUM_TRAINING_SAMPLES, IMAGE_WIDTH, IMAGE_HEIGHT, 1)


def create_empty_label_data():
    return create_empty_label_data_of_size(NUM_TRAINING_SAMPLES)


def create_empty_label_data_of_size(size):
    return numpy.fromiter([0] * size, dtype=numpy.uint8)
