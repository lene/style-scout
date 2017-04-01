from ebaysdk.finding import Connection as Finding
from ebaysdk.shopping import Connection as Shopping

from item import Item


class Category:
    def __init__(self, data):
        self.id = data['CategoryID']
        self.id_path = data.get('CategoryIDPath', '').split(':')
        self.name = data['CategoryName']
        self.name_path = data.get('CategoryNamePath', '').split(':')
        self.is_leaf = data['LeafCategory'] == 'true'
        # also present, but currently not useful:
        # {'CategoryParentID': '15724', 'CategoryLevel': '3'}


class ShoppingAPI:

    def __init__(
            self, auth, siteid=77, config_file=None,
            debug=True, warnings=True, timeout=20
    ):
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
    def get_site_code(siteid):
        return {77: 'EBAY-DE'}[siteid]

    def categories(self, root_id):
        call_data = {
            'CategoryID': root_id,
            'IncludeSelector': 'ChildCategories'
        }
        response = self._api.execute('GetCategoryInfo', call_data)
        return [Category(c) for c in response.dict()['CategoryArray']['Category']]

    def get_category_items(self, category, limit=100, page=1):
        assert limit <= 100, 'Not yet implemented: Searching for more than one page'
        query = {
            'categoryId': [category.id],
            'paginationInput': {'entriesPerPage': limit, 'pageNumber': page},
        }
        response = self._search_api.execute('findItemsAdvanced', query)
        try:
            return [Item(self, category, result['itemId']) for result in response.dict()['searchResult']['item']]
        except KeyError:
            print('Query failed: ', query, category.name)
            return []

    def get_item(self, item_id):
        query = {
            'ItemID': item_id,
            'IncludeSelector': 'Description,ItemSpecifics'
        }

        response = self._api.execute('GetSingleItem', query)
        return response.dict()['Item']

