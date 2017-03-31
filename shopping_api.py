from ebaysdk.finding import Connection as Finding
from ebaysdk.shopping import Connection as Shopping

from item import Item


class ShoppingAPI:

    def __init__(
            self, appid, certid, devid, siteid=77,
            debug=True, config_file=None, warnings=True, timeout=20
    ):
        self._appid = appid
        self._certid = certid
        self._devid = devid
        self._siteid = str(siteid)
        self._debug = debug
        self._warnings = warnings
        self._config_file = config_file
        self._timeout = timeout
        self._api = Shopping(
            debug=debug, config_file=config_file, appid=appid, certid=certid,
            devid=devid, warnings=warnings, timeout=timeout, siteid=self._siteid
        )
        self._search_api = Finding(
            appid=appid, config_file=None,
            siteid='EBAY-DE'
        )

    def categories(self, root_id):
        call_data = {
            'CategoryID': root_id,
            'IncludeSelector': 'ChildCategories'
        }
        response = self._api.execute('GetCategoryInfo', call_data)
        return response.dict()['CategoryArray']['Category']

    def get_category_items(self, category, limit=100):
        query = {
            'categoryId': [category],
            'paginationInput': {'entriesPerPage': limit, 'pageNumber': '1'},
        }
        response = self._search_api.execute('findItemsAdvanced', query)
        return [Item(self, result['itemId']) for result in response.dict()['searchResult']['item']]

    def get_item(self, item_id):
        query = {
            'ItemID': item_id,
            'IncludeSelector': 'Description,ItemSpecifics'
        }

        response = self._api.execute('GetSingleItem', query)
        return response.dict()['Item']

