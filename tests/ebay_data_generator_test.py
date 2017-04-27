__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
from functools import partial
from os import sep, makedirs
from os.path import join

from acquisition.item import Item
from acquisition.items import Items
from data_sets import EbayDataGenerator
from tests.test_base import TestBase, create_item_dict


class EbayDataGeneratorTest(TestBase):

    NUM_IMAGES = 4

    def setUp(self):
        super().setUp()
        self.test_pic = join(sep, *__file__.split('/')[:-1], 'data', 'test.jpg')
        self.api.get_item = partial(create_item_dict, picture_url=['file://' + self.test_pic])
        Item.download_root = self.DOWNLOAD_ROOT
        self.cache_dir = join(self.DOWNLOAD_ROOT, 'cache')
        makedirs(self.cache_dir)

    def test_generator_returns_all_images_and_labels_in_data_set(self):
        items, labels = self._generate_items_with_labels(self.NUM_IMAGES)
        generator = EbayDataGenerator(Items(items), labels, (139, 139), cache_dir=self.cache_dir)
        for i, (images, labels) in enumerate(generator.train_generator()):
            if i >= self.NUM_IMAGES:
                break
            self.assertEqual((self.NUM_IMAGES, 139, 139, 3), images.shape)
            self.assertEqual(1, labels[i][i])
            self.assertTrue(all(label == 0 for j, label in enumerate(labels[i]) if j != i))

    def test_generator_returns_batch_of_correct_size(self):
        items, labels = self._generate_items_with_labels(self.NUM_IMAGES)
        generator = EbayDataGenerator(Items(items), labels, (139, 139), batch_size=2, cache_dir=self.cache_dir)

        images, labels = next(generator.train_generator())
        self.assertEqual((2, 139, 139, 3), images.shape)
        self.assertEqual((2, self.NUM_IMAGES), labels.shape)

    def test_generator_repeats_after_returning_full_data_set(self):
        items, labels = self._generate_items_with_labels(self.NUM_IMAGES)
        generator = EbayDataGenerator(Items(items), labels, (139, 139), batch_size=2, cache_dir=self.cache_dir)

        images0, labels0 = next(generator.train_generator())
        _, _ = next(generator.train_generator())
        images1, labels1 = next(generator.train_generator())
        for i in range(len(labels0)):
            for j in range(len(labels0[i])):
                self.assertEqual(labels0[i][j], labels1[i][j])

    def _generate_items_with_labels(self, num_items):
        items = self._generate_items(num_items)
        for i, item in enumerate(items):
            item.tags = [str(i + 1)]
        return items, [str(i+1) for i in range(num_items)]

    def _generate_items(self, num_items):
        raw_items = [Item(self.api, self.category, i+1) for i in range(num_items)]
        return Items(raw_items)
