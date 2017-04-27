import unittest
from os.path import isdir
from shutil import rmtree
from unittest.mock import Mock

from acquisition.shopping_api import ShoppingAPI
from category import Category


class TestBase(unittest.TestCase):

    DOWNLOAD_ROOT = '/tmp/test-style-scout'
    MOCK_TITLE = 'Mock title'
    MOCK_DESCRIPTION = 'Mock description <strong>with HTML</strong>'

    def setUp(self):
        self.api = Mock(spec=ShoppingAPI)
        self.api.get_item = create_item_dict  # Mock(return_value=self.item_data)
        self.category = Mock(spec=Category)
        self.category.name_path = ['0', '1']

    def tearDown(self):
        if isdir(self.DOWNLOAD_ROOT):
            rmtree(self.DOWNLOAD_ROOT)


def create_item_dict(item_id, specifics=None, picture_url=None):
    return {
        'ItemID': item_id,
        'Title': TestBase.MOCK_TITLE,
        'Description': TestBase.MOCK_DESCRIPTION,
        'ItemSpecifics': specifics,
        'PictureURL': picture_url
    }
