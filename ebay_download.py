import json

from argparse import ArgumentParser
from collections import defaultdict
from operator import itemgetter
from os.path import isfile


from shopping_api import ShoppingAPI
from category import Category
from ebay_downloader_io import EbayDownloaderIO


ITEMS_PER_CATEGORY = 20
MIN_TAG_NUM = 10
SAVE_FOLDER = 'data'
DEFAULT_SIZE = 139


def parse_command_line():
    parser = ArgumentParser(
        description="Download information about eBay items as training data for neural network recognizing style"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help="Print info about extracted tags"
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
        '--save-folder', default=SAVE_FOLDER,
        help='Folder under which to store items, images and weights'
    )
    parser.add_argument(
        '--item-file', default=None, help="Pickle file from which to load downloaded items"
    )    
    parser.add_argument(
        '--images-file', default=None, help='Pickle file from which to load precomputed image data set'
    )
    parser.add_argument(
        '--weights-file', '-w', default=None, help='HDF5 file from which to load precomputed set of weights'
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
        '--min-valid-tag', default=MIN_TAG_NUM, type=int,
        help="Minimum number of times a tag has to occur to be considered valid"
    )
    parser.add_argument(
        '--download-images', action='store_true', help="Download images"
    )
    parser.add_argument(
        '--complete-tags-only', action='store_true', help="Filter out incomplete tags"
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



def get_valid_tags(items, min_count):
    return {
        t: n for t, n in count_all_tags(items).items()
        if n >= min_count
        if 'UNDEFINED' not in t
    }


def update_items(items, page, per_page):
    if per_page:
        for category in categories:
            items += api.get_category_items(category, limit=per_page, page=page)
            if args.verbose:
                print('{} done, {} items in total'.format(category.name, len(items)))
    return remove_duplicate_items(items)


def filter_items_without_complete_tags(items):
    def has_complete_tags(item):
        def has_tag_category(item, tag_category):
            return any(tag_category in tag for tag in item.tags)

        return all(has_tag_category(item, tag_category) for tag_category in item.category.necessary_tags)

    print(len(items))
    items = [item for item in items if has_complete_tags(item)]
    print(len(items))
    return items


def update_tags(items, valid_tags):
    for item in items:
        item.set_tags(set(valid_tags.keys()))


if __name__ == '__main__':
    args = parse_command_line()

    with open(args.ebay_auth_file) as file:
        auth = json.load(file)

    io = EbayDownloaderIO(
        args.save_folder, args.image_size, args.item_file, args.images_file, args.weights_file,
        args.likes_file, args.verbose
    )
    items = io.load_items()

    api = ShoppingAPI(auth['production'], args.ebay_site_id, debug=False)
    categories = Category.search_categories(api)

    items = io.import_likes(api, items)

    for page in range(args.page_from, args.page_to + 1):

        if args.verbose:
            print('\nPage {}, {} distinct items'.format(page, len(items)))

        try:
            items = update_items(items, page, args.items_per_page)
        finally:
            valid_tags = get_valid_tags(items, args.min_valid_tag)
            if args.verbose:
                print_tags(valid_tags)
            update_tags(items, valid_tags)
            if args.complete_tags_only:
                items = filter_items_without_complete_tags(items)
            io.save_items(items)

    if args.download_images:
        for i, item in enumerate(items):
            if args.verbose:
                print(i+1, '/', len(items), end=' ')
            item.download_images(verbose=args.verbose)

    from variable_inception import variable_inception

    image_data = io.get_images(items, valid_tags, args.image_size)

    model = variable_inception(input_shape=(*image_data.size, image_data.DEPTH), classes=image_data.num_classes)

    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])

    io.load_weights(model)

    train = image_data.train.input.reshape(len(image_data.train.input), args.image_size, args.image_size, 3)

    print(train.shape, image_data.train.labels.shape)
    if args.num_epochs:
        model.fit(train, image_data.train.labels, epochs=args.num_epochs)
        io.save_weights(model)

    test = image_data.test.input.reshape(len(image_data.test.input), args.image_size, args.image_size, 3)
    loss_and_metrics = model.evaluate(test, image_data.test.labels)
    print()
    print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])
