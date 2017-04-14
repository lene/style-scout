__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

from tests.test_base import TestBase, create_item_dict
from item import Item

# from ebay_download import filter_items_without_complete_tags


class EbayDownloadTest(TestBase):

    def setUp(self):
        super().setUp()
        self.category.necessary_tags = ['style']

