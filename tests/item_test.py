
from functools import partial
from os import sep
from os.path import join, isfile

from acquisition.item import Item
from tests.test_base import TestBase, create_item_dict

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class ItemTest(TestBase):

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

    def test_get_possible_tags_without_specifics_with_undefined(self):
        item = Item(self.api, self.category, 1)
        tags = item.get_possible_tags(add_undefined=True)
        for tag_label in Item.TAG_LIST.values():
            self.assertIn('{}:UNDEFINED'.format(tag_label), tags)
        self.assertIn(self.category.name_path[-1], tags)
        self.assertNotIn(self.category.name_path[0], tags)

    def test_get_possible_tags_without_undefined(self):
        item = Item(self.api, self.category, 1)
        tags = item.get_possible_tags(add_undefined=False)
        self.assertEquals({self.category.name_path[-1]}, tags)

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
        self.api.get_item = partial(create_item_dict, picture_url=['file://' + test_pic])
        Item.download_root = self.DOWNLOAD_ROOT
        item = Item(self.api, self.category, 1)

        item.download_images()

        self.assertTrue(isfile(self.DOWNLOAD_ROOT + '/style-scout_tests_data_test.jpg'))

    def test_download_images_nonexistent(self):
        self.api.get_item = partial(create_item_dict, picture_url=['file:///tmp/nonexistent'])
        Item.download_root = self.DOWNLOAD_ROOT
        item = Item(self.api, self.category, 1)

        item.download_images()  # should just ignore the error
