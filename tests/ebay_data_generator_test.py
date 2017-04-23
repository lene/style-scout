__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
from tests.test_base import TestBase, create_item_dict

from data_sets import EbayDataGenerator
from item import Item
from items import Items

from functools import partial
from os.path import join, isfile
from os import sep

class EbayDataGeneratorTest(TestBase):

    def setUp(self):
        super().setUp()
        self.test_pic = join(sep, *__file__.split('/')[:-1], 'data', 'test.jpg')
        self.api.get_item = partial(create_item_dict, picture_url=['file://' + self.test_pic])
        Item.download_root = self.DOWNLOAD_ROOT

    def test_first(self):
        items, labels = self._generate_items_with_labels(4)
        generator = EbayDataGenerator(Items(items), labels, (139, 139))
        for i, (image, labels) in enumerate(generator.generate_arrays()):
            self.assertEqual((139, 139, 3), image.shape)
            self.assertEqual(1, labels[i])
            self.assertTrue(all(label == 0 for j, label in enumerate(labels) if j != i))

    def _generate_items(self, num_items):
        raw_items = [Item(self.api, self.category, i+1) for i in range(num_items)]
        return Items(raw_items)

    def _generate_items_with_labels(self, num_items):
        items = self._generate_items(num_items)
        for i, item in enumerate(items):
            item.tags = [str(i + 1)]
        return items, [str(i+1) for i in range(num_items)]
