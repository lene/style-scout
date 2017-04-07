from test_base import TestBase, create_item_dict

from data_sets import EbayDataSets
from item import Item

import numpy

from functools import partial
from os.path import join, isfile
from os import sep

SIZE = 139


class EbayDataSetsTest(TestBase):

    def setUp(self):
        super().setUp()
        self.test_pic = join(sep, *__file__.split('/')[:-1], 'data', 'test.jpg')
        self.api.get_item = partial(create_item_dict, picture_url=['file://' + self.test_pic])
        Item.download_root = self.DOWNLOAD_ROOT

    def test_basic_create(self):
        items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        data_sets = EbayDataSets(
            items=items, valid_labels=[], size=(SIZE, SIZE)
        )
        self.assertEqual(SIZE*SIZE*3, data_sets.num_features)
        self.assertEqual(0, data_sets.num_classes)

    def test_labels(self):
        self.setUp()

        items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        data_sets = EbayDataSets(
            items=items, valid_labels=['1', '2'], size=(SIZE, SIZE)
        )
        self.assertEqual(2, data_sets.num_classes)
        self.assertDictEqual({'1': 0, '2': 1}, data_sets.labels_to_numbers)
        self.assertDictEqual({0: '1', 1: '2'}, data_sets.numbers_to_labels)

    def test_get_data(self):
        items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        data_sets = EbayDataSets.get_data(
            join(self.DOWNLOAD_ROOT, 'test'), items=items, valid_labels=['1', '2'], image_size=SIZE
        )
        self.assertEqual(2, data_sets.num_classes)
        self.assertDictEqual({'1': 0, '2': 1}, data_sets.labels_to_numbers)
        self.assertDictEqual({0: '1', 1: '2'}, data_sets.numbers_to_labels)
        self.assertEqual(2, len(data_sets.train))
        self.assertEqual(0, len(data_sets.test))
        self.assertEqual(0, len(data_sets.validation))

    def test_get_data_creates_npz(self):
        items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]

        valid_labels = ['1', '2']
        EbayDataSets.get_data(
            join(self.DOWNLOAD_ROOT, 'ebay_data_sets_test'),
            items=items, valid_labels=valid_labels, image_size=SIZE
        )

        self.assertTrue(isfile(join(self.DOWNLOAD_ROOT, 'ebay_data_sets_test.npz')))
        npzfile = numpy.load(join(self.DOWNLOAD_ROOT, 'ebay_data_sets_test.npz'))
        self.assertCountEqual(
            [
                'train_images', 'train_labels', 'test_images', 'test_labels',
                'validation_images', 'validation_labels'
            ], npzfile.files
        )
        self.assertEqual((len(items), SIZE, SIZE, EbayDataSets.DEPTH), npzfile['train_images'].shape)
        self.assertEqual((len(items), len(valid_labels)), npzfile['train_labels'].shape)
