import json
from pickle import dump, load

from argparse import ArgumentParser, ArgumentTypeError
from collections import defaultdict
from operator import itemgetter
from ebaysdk.exception import ConnectionError
from os.path import isfile

from os import rename, remove

from shopping_api import ShoppingAPI
from category import Category

PICKLE = 'items.pickle'

ITEMS_PER_CATEGORY = 20
MIN_TAG_NUM = 10
WEIGHTS_FILE = 'ebay.hdf5'
IMAGES_FILE = 'ebay_images.pickle'
DEFAULT_SIZE = 139


def parse_command_line():
    parser = ArgumentParser(
        description="Download information about eBay items as training data for style neural network"
    )
    parser.add_argument(
        '--items-per-page', default=ITEMS_PER_CATEGORY, type=int,
        help="Page size (once per every category)"
    )
    parser.add_argument(
        '--page-from', default=1, type=int, help="Page number from which to start"
    )
    parser.add_argument(
        '--page-to', default=1, type=int, help="Page number up to which to read"
    )
    parser.add_argument(
        '--min-valid-tag', default=MIN_TAG_NUM, type=int,
        help="Minimum number of times a tag has to occur to be considered valid"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help="Print info about extracted tags"
    )
    parser.add_argument(
        '--item-file', default=PICKLE, help="Pickle file in which downloaded items are stored"
    )    
    parser.add_argument(
        '--ebay-auth-file', default='ebay_auth.json',
        help="JSON file containing the eBay authorization IDs"
    )    
    parser.add_argument(
        '--likes-file', help="JSON file containing the liked item IDs"
    )
    parser.add_argument(
        '--ebay-site_id', type=int, default=77, help="eBay site ID (77 for Germany)"
    )    
    parser.add_argument(
        '--download-images', action='store_true', help="Download images"
    )
    parser.add_argument(
        '--images-file', default=IMAGES_FILE,
        help='Pickle file containing precomputed image data set'
    )
    parser.add_argument(
        '--weights-file', '-w', default=WEIGHTS_FILE,
        help='HDF5 file containing a precomputed set of weights for this neural network.'
    )
    parser.add_argument(
        '--num-epochs', '-n', type=int, default=1,
        help='How many times to iterate.'
    )
    parser.add_argument(
        '--image-size', '-s', type=int, default=DEFAULT_SIZE,
        help='Size (both width and height) to which images are resized.'
    )
    return parser.parse_args()


def print_tags(tags, num_most_popular=50):
    for k in sorted(tags.keys()):
        if 'style:' in k or 'heel height:' in k or 'length:' in k or 'pattern:' in k or True: #  completely disabled for now
            continue
        print(k, tags[k])
    print()
    print(num_most_popular, 'most popular tags:')
    for k, v in sorted(tags.items(), key=itemgetter(1))[-num_most_popular:]:
        print(k, v)
    print(len(tags), 'distinct tags')


def remove_duplicate_items(items):
    ids = set()
    new_items = []
    for item in items:
        if item.id not in ids:
            if isinstance(item.category, int):
                print(item.title, item.category)
                continue
            new_items.append(item)
            ids.add(item.id)
    print(len(items), '->', len(new_items), 'items')
    return new_items


def count_all_tags(items):
    counted_tags = defaultdict(int)
    for item in items:
        for tag in item.get_possible_tags():
            counted_tags[tag] += 1
    return counted_tags


def dump_objects_to_file(filename, objects):
    if isfile(filename):
        if isfile(filename + '.bak'):
            remove(filename + '.bak')
        rename(filename, filename + '.bak')
    with open(filename, 'wb') as file:
        dump(objects, file)


def get_valid_tags(items, min_count):
    return {
        t: n for t, n in count_all_tags(items).items()
        if n >= min_count
    }


def load_objects_from_file(filename):
    if isfile(filename):
        with open(filename, 'rb') as file:
            return load(file)
    return []


def update_items(items, page, per_page):
    if per_page:
        for category in categories:
            items += api.get_category_items(category, limit=per_page, page=page)
            print('{} done, {} items in total'.format(category.name, len(items)))
    return remove_duplicate_items(items)


def add_liked_items(api, items, category, liked_item_ids):
    from item import Item
    present_item_ids = set(i.id for i in items)
    for liked in liked_item_ids:
        if liked in present_item_ids:
            Item.set_liked(items, liked)
        else:
            new_item = Item(api, category, liked)
            new_item.like()
            items.append(new_item)


def import_likes(api, filename, items):

    if not isfile(filename):
        return

    with open(filename, 'r') as f:
        liked = json.load(f)

    for category_id, item_ids in liked.items():
        if category_id.isdigit():
            category = Category.by_id(category_id)
            print(category.name)
            add_liked_items(api, items, category, item_ids)


args = parse_command_line()

with open(args.ebay_auth_file) as file:
    auth = json.load(file)

items = load_objects_from_file(args.item_file)

api = ShoppingAPI(auth['production'], args.ebay_site_id, debug=False)
categories = Category.search_categories(api)

if args.likes_file:
    import_likes(api, args.likes_file, items)

if args.download_images:
    for i, item in enumerate(items):
        print(i, '/', len(items), end=' ')
        item.download_images(verbose=True)

for page in range(args.page_from, args.page_to + 1):

    print('\nPage {}, {} distinct items'.format(page, len(items)))

    try:

        items = update_items(items, page, args.items_per_page)

        valid_tags = get_valid_tags(items, args.min_valid_tag)

        if args.verbose:
            print_tags(valid_tags)

        for item in items:
            item.set_tags(set(valid_tags.keys()))

    finally:
        dump_objects_to_file(args.item_file, items)

from data_sets.ebay_data_sets import EbayDataSets
from variable_inception import variable_inception

data = EbayDataSets.get_data(args.images_file, items, valid_tags, args.image_size)

model = variable_inception(input_shape=(*data.size, data.depth), classes=data.num_classes)

model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])

if isfile(args.weights_file):
    print('Loading ' + args.weights_file)
    model.load_weights(args.weights_file)

train = data.train.input.reshape(len(data.train.input), args.image_size, args.image_size, 3)

print(train.shape, data.train.labels.shape)
if args.num_epochs:
    model.fit(train, data.train.labels, epochs=args.num_epochs)
    model.save_weights(args.weights_file)
 
test = data.test.input.reshape(len(data.test.input), args.image_size, args.image_size, 3)
loss_and_metrics = model.evaluate(test, data.test.labels)
print()
print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])
