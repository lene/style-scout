from typing import Tuple, List

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
from functools import partial
from os import sep, makedirs
from os.path import join

import numpy

from acquisition.item import Item
from acquisition.items import Items
from data_sets import EbayDataGenerator
from tests.test_base import TestBase, create_item_dict


class EbayDataGeneratorTest(TestBase):

    NUM_IMAGES = 4

    def setUp(self) -> None:
        super().setUp()
        self.test_pic = join(sep, *__file__.split('/')[:-1], 'data', 'test.jpg')
        self.api.get_item = partial(create_item_dict, picture_url=['file://' + self.test_pic])
        Item.download_root = self.DOWNLOAD_ROOT

    def test_generator_returns_all_images_and_labels_in_data_set(self) -> None:
        items, labels = self._generate_items_with_labels(self.NUM_IMAGES)
        generator = EbayDataGenerator(items, labels, (139, 139), test_share=0)
        for i, (images, labels) in enumerate(generator.train_generator()):
            if i >= self.NUM_IMAGES:
                break
            self.assertEqual((self.NUM_IMAGES, 139, 139, 3), images.shape)
            self.assertIn([0, 0, 0, 1], labels)
            self.assertIn([0, 0, 1, 0], labels)
            self.assertIn([0, 1, 0, 0], labels)
            self.assertIn([1, 0, 0, 0], labels)

    def test_generator_returns_batch_of_correct_size(self) -> None:
        items, labels = self._generate_items_with_labels(self.NUM_IMAGES)
        generator = EbayDataGenerator(items, labels, (139, 139), batch_size=2)

        images, labels = next(generator.train_generator())
        self.assertEqual((2, 139, 139, 3), images.shape)
        self.assertEqual((2, self.NUM_IMAGES), labels.shape)

    def test_generator_repeats_after_returning_full_data_set(self) -> None:
        items, labels = self._generate_items_with_labels(self.NUM_IMAGES)
        generator = EbayDataGenerator(items, labels, (139, 139), batch_size=2)

        images0, labels0 = next(generator.train_generator())
        _, _ = next(generator.train_generator())
        images1, labels1 = next(generator.train_generator())
        for i in range(len(labels0)):
            for j in range(len(labels0[i])):
                self.assertEqual(labels0[i][j], labels1[i][j])

    def test_random_seed(self) -> None:
        self.skipTest("Not yet implemented")

    def test_test_set_share_is_respected(self) -> None:
        items, labels = self._generate_items_with_labels(self.NUM_IMAGES)
        generator = EbayDataGenerator(items, labels, (139, 139), test_share=0)
        test_items = self._get_from_generator(self.NUM_IMAGES, generator)
        self.assertEqual(4, len(set(test_items[0])))

        generator = EbayDataGenerator(items, labels, (139, 139), test_share=0.5)
        test_items = self._get_from_generator(self.NUM_IMAGES, generator)
        self.assertEqual(2, len(set(test_items[0])))

        generator = EbayDataGenerator(items, labels, (139, 139), test_share=0.75)
        test_items = self._get_from_generator(self.NUM_IMAGES, generator)
        self.assertEqual(1, len(set(test_items[0])))

    def test_training_set_share_of_zero_raises(self) -> None:
        items, labels = self._generate_items_with_labels(self.NUM_IMAGES)
        with self.assertRaises(ValueError):
            generator = EbayDataGenerator(items, labels, (139, 139), test_share=1)
            self._get_from_generator(self.NUM_IMAGES, generator)

    def _generate_items_with_labels(self, num_items: int) -> Tuple[Items, numpy.ndarray]:
        items = self._generate_items(num_items)
        for i, item in enumerate(items):
            item.tags = [str(i + 1)]
        return items, [str(i + 1) for i in range(num_items)]

    def _generate_items(self, num_items: int) -> Items:
        raw_items = [Item(self.api, self.category, i + 1) for i in range(num_items)]
        return Items(raw_items)

    @staticmethod
    def _get_from_generator(num_items: int, generator: EbayDataGenerator) -> List:
        return [
            tuple([tuple(line.tolist()) for line in next(generator.train_generator())[1]])
            for _ in range(num_items)
        ]
