import json
from pickle import dump, load

from collections import defaultdict
from operator import itemgetter
from ebaysdk.exception import ConnectionError
from os.path import isfile

from shopping_api import ShoppingAPI

PICKLE = 'items.pickle'

ITEMS_PER_CATEGORY = 100
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
MIN_TAG_NUM = 10

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
                if any(term.lower() in category.name.lower() for term in search_term_filter[level])
            ]
            leaf_categories += [c for c in categories if c.is_leaf]
            next_level_cat_ids += [c.id for c in categories if not c.is_leaf]
        category_ids = next_level_cat_ids
    return leaf_categories


def print_tags(tags, num_most_popular=30):
    for k in sorted(tags.keys()):
        if 'style:' in k or 'heel height:' in k or 'length:' in k:
            continue
        print(k, tags[k])
    print()
    print(num_most_popular, 'most popular tags:')
    for k, v in sorted(tags.items(), key=itemgetter(1))[-num_most_popular:]:
        print(k, v)
    print(len(tags), 'tags')


if isfile(PICKLE):
    with open(PICKLE, 'rb') as file:
        items = load(file)
else:
    items = []


def remove_duplicate_items(items):
    print(len(items))
    ids = set()
    new_items = []
    for item in items:
        if item.id not in ids:
            new_items.append(item)
            ids.add(item.id)
    print(len(new_items), 'tags')
    return new_items


for page in range(0, 100):
    print('\nPage', page)
    try:

        api = ShoppingAPI(
            auth['production'], 77, debug=False
        )

        categories = search_categories(DEFAULT_CATEGORIES)

        if ITEMS_PER_CATEGORY:
            for category in categories:
                items += api.get_category_items(category, limit=ITEMS_PER_CATEGORY, page=page)
                print(category.name)

        items = remove_duplicate_items(items)

        tags = defaultdict(int)
        for item in items:
            for tag in item.get_tag_suggestions():
                tags[tag] += 1
        tags = {t: n for t, n in tags.items() if n >= MIN_TAG_NUM}
        print_tags(tags)

        for item in items:
            item.set_tags(tags)

    except ConnectionError as e:
        print(e)
        print(e.response.dict())
    finally:
        with open(PICKLE, 'wb') as file:
            dump(items, file)
