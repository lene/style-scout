
from item import Item
from shopping_api import ShoppingAPI, Category

import unittest
from unittest.mock import Mock
from functools import partial
from os.path import join, isfile, isdir
from os import sep
from shutil import rmtree

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class ItemTest(unittest.TestCase):

    MOCK_TITLE = 'Mock title'
    MOCK_DESCRIPTION = 'Mock description <strong>with HTML</strong>'
    DOWNLOAD_ROOT = '/tmp/test-style-scout'

    def setUp(self):
        self.api = Mock(spec=ShoppingAPI)
        self.api.get_item = create_item_dict  # Mock(return_value=self.item_data)
        self.category = Mock(spec=Category)
        self.category.name_path = ['0', '1']

    def tearDown(self):
        if isdir(self.DOWNLOAD_ROOT):
            rmtree(self.DOWNLOAD_ROOT)

    def test_simple_create(self):
        item = Item(self.api, self.category, 1)
        self.assertEqual(1, item.id)
        self.assertEqual(self.MOCK_TITLE, item.title)

    def test_html_is_stripped(self):
        item = Item(self.api, self.category, 1)
        self.assertNotEqual(self.MOCK_DESCRIPTION, item.description)
        self.assertEqual('Mock description with HTML', item.description)

    def test_like(self):
        item = Item(self.api, self.category, 1)
        self.assertEqual(set(), item.tags)
        item.like()
        self.assertEqual({'<3'}, item.tags)

    def test_set_liked(self):
        items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        Item.set_liked(items, 1)
        self.assertEqual({'<3'}, items[0].tags)
        self.assertEqual(set(), items[1].tags)

    def test_set_liked_raises_if_not_found(self):
        items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        with self.assertRaises(ValueError):
            Item.set_liked(items, 3)

    def test_get_possible_tags_without_specifics(self):
        item = Item(self.api, self.category, 1)
        tags = item.get_possible_tags()
        for tag_label in Item.TAG_LIST.values():
            self.assertIn('{}:UNDEFINED'.format(tag_label), tags)
        self.assertIn(self.category.name_path[-1], tags)
        self.assertNotIn(self.category.name_path[0], tags)

    def test_get_possible_tags_with_one_specific(self):
        self.api.get_item = partial(
            create_item_dict, specifics={
                'NameValueList': [
                    {'Name': 'stil', 'Value': 'blah'}
                ]
            })
        item = Item(self.api, self.category, 1)
        self.assertIn('style:blah', item.get_possible_tags())

    def test_get_possible_tags_with_multiple_specifics(self):
        self.api.get_item = partial(
            create_item_dict, specifics={
                'NameValueList': [
                    {'Name': 'stil', 'Value': 'blah'},
                    {'Name': 'stil', 'Value': 'blub'}
                ]
            })
        item = Item(self.api, self.category, 1)
        self.assertIn('style:blah', item.get_possible_tags())
        self.assertIn('style:blub', item.get_possible_tags())

    def test_get_possible_tags_with_ignored_specific(self):
        self.api.get_item = partial(
            create_item_dict, specifics={
                'NameValueList': [
                    {'Name': 'ignored', 'Value': 'blah'}
                ]
            })
        item = Item(self.api, self.category, 1)
        for tag in item.get_possible_tags():
            self.assertNotIn('ignored', tag)
            self.assertNotIn('blah', tag)

    def test_download_images_jpg(self):
        test_pic = join(sep, *__file__.split('/')[:-1], 'data', 'test.jpg')
        self.api.get_item = partial(create_item_dict, picture_url=['file://'+test_pic])
        Item.download_root = self.DOWNLOAD_ROOT
        item = Item(self.api, self.category, 1)

        item.download_images()

        self.assertTrue(isfile(self.DOWNLOAD_ROOT + '/style-scout_tests_data_test.jpg'))

    def test_download_images_nonexistent(self):
        self.api.get_item = partial(create_item_dict, picture_url=['file:///tmp/nonexistent'])
        Item.download_root = self.DOWNLOAD_ROOT
        item = Item(self.api, self.category, 1)

        item.download_images()  # should just ignore the error


def create_item_dict(item_id, specifics=None, picture_url=None):
    return {
        'ItemID': item_id,
        'Title': ItemTest.MOCK_TITLE,
        'Description': ItemTest.MOCK_DESCRIPTION,
        'ItemSpecifics': specifics,
        'PictureURL': picture_url
    }
