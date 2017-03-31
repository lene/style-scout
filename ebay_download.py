
from datetime import datetime
from pprint import pprint
import re
import json

from ebaysdk.exception import ConnectionError
from ebaysdk.finding import Connection as Finding
from ebaysdk.shopping import Connection as Shopping

NUM_PER_PAGE = 200
DEFAULT_SEARCH_TERMS = {
    1: ('Kleidung',),
    2: ('Damenmode', 'Damenschuhe'),
    3: (
        'Anzüge & Kombinationen', 'Blusen, Tops & Shirts',
        'Jacken & Mäntel', 'Kleider', 'Röcke', 'Damenunterwäsche',
        'Halbschuhe & Ballerinas', 'Pumps', 'Sandalen & Badeschuhe',
        'Stiefel & Stiefeletten'
    ),
    4: ('Bodys', 'Nachtwäsche')
}

with open('ebay_auth.json') as file:
    auth = json.load(file)

query = {
    'keywords': 'laptop',
    'categoryId': ['177', '111422'],
    'itemFilter': [
        {
            'name': 'Condition', 'value': 'Used'
        },
        {
            'name': 'MinPrice', 'value': '200',
            'paramName': 'Currency', 'paramValue': 'GBP'
        },
        {
            'name': 'MaxPrice', 'value': '400',
            'paramName': 'Currency', 'paramValue': 'GBP'
        }
    ],
    'paginationInput': {'entriesPerPage': NUM_PER_PAGE, 'pageNumber': '1'},
    'sortOrder': 'CurrentPriceHighest'
}


def search(categories, limit=100):
    assert len(categories) < 4
    try:
        api = Finding(appid=auth['production']['app_id'], config_file=None, siteid='EBAY-DE')
        query = {
            'categoryId': categories,
            'paginationInput': {'entriesPerPage': limit, 'pageNumber': '1'},
        }

        response = api.execute('findItemsAdvanced', query)
        return response.dict()['searchResult']['item']

    except ConnectionError as e:
        print(e)
        pprint(e.response.dict())


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

    def categories(self, root_id):
        call_data = {
            'CategoryID': root_id,
            'IncludeSelector': 'ChildCategories'
        }
        response = self._api.execute('GetCategoryInfo', call_data)
        return response.dict()['CategoryArray']['Category']

    def get_item(self, item_id):
        query = {
            'ItemID': item_id,
            'IncludeSelector': 'Description,ItemSpecifics'
        }

        response = self._api.execute('GetSingleItem', query)
        return response.dict()['Item']


class Item:

    download_root = 'data'
    account = 'eBay'

    def __init__(self, api, item_id):
        item = api.get_item(item_id)
        self.id = item['ItemID']
        self.title = item['Title']
        self.description = self.clean_description(item['Description'])
        self.item_specifics = item.get('ItemSpecifics', {})
        self.picture_urls = item['PictureURL']
        self.picture_files = []
        for i, picture_url in enumerate(self.picture_urls):
            self.download_image(picture_url, i == 0)

    def __str__(self):
        return """Id: {}
Title: {}
{}
Specifics: {}
Pix: {}""".format(self.id, self.title, self.description, self.item_specifics, self.picture_urls)

    @classmethod
    def clean_description(cls, description):
        description = re.sub('<[^<]+?>', '', description)
        description = description.replace('&nbsp;', ' ')
        description = description.replace('\n', ' ')
        return description
    
    @classmethod
    def download_image(cls, url, show=False):
        from os.path import join, isfile
        from os import makedirs
        from PIL import Image
        from urllib.request import urlretrieve

        makedirs(join(cls.download_root, cls.account), exist_ok=True)
        filename = join(cls.download_root, cls.account, '_'.join(url.split('/')[-4:]))
        if not isfile(filename):
            urlretrieve(url, filename)
        # for debugging/following the status, mostly
        with Image.open(filename) as image:
            if show:
                image.show()
            # print(filename, image.size)


def search_categories(search_term_filter, root_category=-1):
    category_ids = [root_category]
    leaf_categories = []
    for level in range(1, 5):
        next_level_cat_ids = []
        for id in category_ids:
            categories = [
                category for category in api.categories(id)
                if any(term.lower() in category['CategoryName'].lower() for term in search_term_filter[level])
            ]
            leaf_categories += [c for c in categories if c['LeafCategory'] != 'false']
            next_level_cat_ids += [c['CategoryID'] for c in categories if c['LeafCategory'] == 'false']
        category_ids = next_level_cat_ids
    return leaf_categories


try:

    api = ShoppingAPI(
        auth['production']['app_id'], auth['production']['dev_id'],
        auth['production']['cert_id'], 77, debug=False
    )

    categories = search_categories(DEFAULT_SEARCH_TERMS)

    results = search([c['CategoryID'] for c in categories][:3], 10)

    for result in results:
        item = Item(api, result['itemId'])
        print(str(item))

except ConnectionError as e:
    print(e)
    print(e.response.dict())
