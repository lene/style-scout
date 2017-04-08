__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

from tests.test_base import TestBase, create_item_dict
from item import Item

from ebay_download import filter_items_without_complete_tags


class EbayDownloadTest(TestBase):

    def setUp(self):
        super().setUp()
        self.category.necessary_tags = ['style']

    def test_filter_items_without_complete_tags(self):
        item1 = Item(self.api, self.category, 1)
        item1.tags = {'blah:blub'}
        item2 = Item(self.api, self.category, 2)
        item2.tags = {'blah:blub', 'style:cool_shit'}
        print(self.category.necessary_tags)

        self.assertEqual([item2], filter_items_without_complete_tags([item1, item2]))

