from functools import partial
from os import sep
from os.path import join, isfile
from typing import Tuple, List, Dict

import numpy

from acquisition.item import Item
from acquisition.items import Items
from data_sets import EbayDataSets
from tests.test_base import TestBase, create_item_dict

SIZE = 139


class EbayDataSetsTest(TestBase):

    def setUp(self) -> None:
        super().setUp()
        self.test_pic = join(sep, *__file__.split('/')[:-1], 'data', 'test.jpg')
        self.api.get_item = partial(create_item_dict, picture_url=['file://' + self.test_pic])
        Item.download_root = self.DOWNLOAD_ROOT

    def test_basic_create(self) -> None:
        items = Items([Item(self.api, self.category, 1), Item(self.api, self.category, 2)])
        data_sets = EbayDataSets(
            items=items, valid_labels={}, size=(SIZE, SIZE)
        )
        self.assertEqual(SIZE * SIZE * 3, data_sets.num_features)
        self.assertEqual(0, data_sets.num_classes)

    def test_labels(self) -> None:
        items = Items([Item(self.api, self.category, 1), Item(self.api, self.category, 2)])
        valid_labels = {'1': 1, '2': 1}
        data_sets = EbayDataSets(
            items=items, valid_labels=valid_labels, size=(SIZE, SIZE)
        )
        self.assertEqual(2, data_sets.num_classes)
        self.assertDictEqual({'1': 0, '2': 1}, data_sets.labels_to_numbers)
        self.assertDictEqual({0: '1', 1: '2'}, data_sets.numbers_to_labels)

    def test_get_data(self) -> None:
        items = Items([Item(self.api, self.category, 1), Item(self.api, self.category, 2)])
        valid_labels = {'1': 1, '2': 1}
        data_sets = EbayDataSets.get_data(
            join(self.DOWNLOAD_ROOT, 'test'), items=items, valid_labels=valid_labels, image_size=SIZE
        )
        self.assertDictEqual({'1': 0, '2': 1}, data_sets.labels_to_numbers)
        self.assertDictEqual({0: '1', 1: '2'}, data_sets.numbers_to_labels)
        self.assertEqual(2, len(data_sets.train))
        self.assertEqual(0, len(data_sets.test))
        self.assertEqual(0, len(data_sets.validation))

    def test_get_data_creates_npz(self) -> None:
        items = Items([Item(self.api, self.category, 1), Item(self.api, self.category, 2)])

        valid_labels = {'1': 1, '2': 1}
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

    def test_labels_correctly_associated(self) -> None:
        items, valid_labels = self._create_enough_items()

        data_sets = EbayDataSets.get_data(
            join(self.DOWNLOAD_ROOT, 'ebay_data_sets_test'),
            items=items, valid_labels=valid_labels, image_size=SIZE,
            test_share=0
        )
        self.assertEqual(4, len(data_sets.train.labels))
        self.assertTrue([1., 1., 0., 0., 0., 0., 0., 0.] in data_sets.train.labels.tolist())
        self.assertTrue([0., 0., 1., 1., 0., 0., 0., 0.] in data_sets.train.labels.tolist())
        self.assertTrue([0., 0., 0., 0., 1., 1., 0., 0.] in data_sets.train.labels.tolist())
        self.assertTrue([0., 0., 0., 0., 0., 0., 1., 1.] in data_sets.train.labels.tolist())

    def test_labels_correctly_restored(self) -> None:
        items, valid_labels = self._create_enough_items()

        EbayDataSets.get_data(
            join(self.DOWNLOAD_ROOT, 'ebay_data_sets_test'),
            items=items, valid_labels=valid_labels, image_size=SIZE,
            test_share=0, verbose=True
        )

        other_data_sets = EbayDataSets.get_data(
            join(self.DOWNLOAD_ROOT, 'ebay_data_sets_test'),
            items=items, valid_labels=valid_labels, image_size=SIZE,
            test_share=0, verbose=True
        )
        self.assertEqual(4, len(other_data_sets.train.labels))
        self.assertTrue([1., 1., 0., 0., 0., 0., 0., 0.] in other_data_sets.train.labels.tolist())
        self.assertTrue([0., 0., 1., 1., 0., 0., 0., 0.] in other_data_sets.train.labels.tolist())
        self.assertTrue([0., 0., 0., 0., 1., 1., 0., 0.] in other_data_sets.train.labels.tolist())
        self.assertTrue([0., 0., 0., 0., 0., 0., 1., 1.] in other_data_sets.train.labels.tolist())

    def _create_enough_items(self) -> Tuple[Items, Dict[str, int]]:
        # set up enough items and labels to have a decent probability of not succeeding by chance
        item1 = Item(self.api, self.category, 1)
        item1.tags = ['1', '2']
        item2 = Item(self.api, self.category, 2)
        item2.tags = ['3', '4']
        item3 = Item(self.api, self.category, 3)
        item3.tags = ['5', '6']
        item4 = Item(self.api, self.category, 4)
        item4.tags = ['7', '8']
        valid_labels = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1}
        return Items([item1, item2, item3, item4]), valid_labels
