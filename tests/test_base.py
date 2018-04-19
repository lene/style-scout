import unittest
from os.path import isdir
from os import makedirs
from shutil import rmtree
from typing import Dict, Any
from unittest.mock import Mock

from acquisition.item import Item
from acquisition.items import Items
from acquisition.shopping_api import ShoppingAPI
from category import Category


class TestBase(unittest.TestCase):

    DOWNLOAD_ROOT = '/tmp/test-style-scout'
    MOCK_TITLE = 'Mock title'
    MOCK_DESCRIPTION = 'Mock description <strong>with HTML</strong>'

    def setUp(self) -> None:
        self.api = Mock(spec=ShoppingAPI)
        self.api.get_item = create_item_dict  # Mock(return_value=self.item_data)
        self.category = Mock(spec=Category)
        self.category.name_path = ['0', '1']
        makedirs(self.DOWNLOAD_ROOT, exist_ok=False)

    def tearDown(self) -> None:
        if isdir(self.DOWNLOAD_ROOT):
            rmtree(self.DOWNLOAD_ROOT)

    def generate_items(self, num_items: int) -> Items:
        raw_items = [Item(self.api, self.category, i + 1) for i in range(num_items)]
        return Items(raw_items)


def create_item_dict(item_id: int, specifics: str=None, picture_url: str=None) -> Dict[str, Any]:
    return {
        'ItemID': item_id,
        'Title': TestBase.MOCK_TITLE,
        'Description': TestBase.MOCK_DESCRIPTION,
        'ItemSpecifics': specifics,
        'PictureURL': picture_url
    }
