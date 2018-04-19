from typing import Dict, List

from ebaysdk.finding import Connection as Finding
from ebaysdk.shopping import Connection as Shopping

from acquisition.item import EbayItem, Item
from acquisition.items import Items
from category import Category


class ShoppingAPI:

    def __init__(
            self, auth: Dict[str, str], siteid: int=77, config_file: str=None,
            debug: bool=True, warnings: bool=True, timeout: int=20
    ) -> None:
        self._appid = auth['app_id']
        self._certid = auth['cert_id']
        self._devid = auth['dev_id']
        self._siteid = str(siteid)
        self._api = Shopping(
            appid=self._appid, certid=self._certid, devid=self._devid, siteid=self._siteid,
            config_file=config_file, debug=debug, warnings=warnings, timeout=timeout
        )
        self._search_api = Finding(
            appid=self._appid, siteid=self.get_site_code(siteid),
            config_file=config_file, debug=debug, warnings=warnings, timeout=timeout
        )

    @staticmethod
    def get_site_code(siteid: int) -> str:
        return {77: 'EBAY-DE'}[siteid]

    def categories(self, root_id: int) -> List[Category]:
        call_data = {
            'CategoryID': root_id,
            'IncludeSelector': 'ChildCategories'
        }
        response = self._api.execute('GetCategoryInfo', call_data)
        return [Category(c) for c in response.dict()['CategoryArray']['Category']]

    def get_category_items(self, category: Category, limit: int=100, page: int=1) -> Items:
        assert limit <= 100, 'Not yet implemented: Searching for more than one page'
        query = {
            'categoryId': [category.id],
            # 'sortOrder': 'EndTimeSoonest',
            'paginationInput': {'entriesPerPage': limit, 'pageNumber': page},
        }
        response = self._search_api.execute('findItemsAdvanced', query)
        try:
            return Items([
                EbayItem(self, category, result['itemId'])
                for result in response.dict()['searchResult'].get('item', [])
            ])
        except KeyError:
            from pprint import pprint
            print('Query failed: ', category.name)
            pprint(response.dict())
            raise

    def get_item(self, item_id: int) -> Dict:
        query = {
            'ItemID': item_id,
            'IncludeSelector': 'Description,ItemSpecifics'
        }

        response = self._api.execute('GetSingleItem', query)
        return response.dict()['Item']
