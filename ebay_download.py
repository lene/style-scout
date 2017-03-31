import json

from ebaysdk.exception import ConnectionError

from shopping_api import ShoppingAPI

NUM_PER_PAGE = 200
DEFAULT_CATEGORIES = {
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

    categories = search_categories(DEFAULT_CATEGORIES)

    results = api.get_category_items(categories[0]['CategoryID'], 10)

    for result in results:
        print(str(result))

except ConnectionError as e:
    print(e)
    print(e.response.dict())
