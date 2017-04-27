import json

from argparse import ArgumentParser
from operator import itemgetter

from shopping_api import ShoppingAPI
from category import Category
from ebay_downloader_io import EbayDownloaderIO


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
        '--items-per-page', default=0, type=int,
        help="Page size (once per every category) (0 to disable downloading items)"
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


def update_items(items, page, per_page):
    if per_page:
        for category in categories:
            items.extend(api.get_category_items(category, limit=per_page, page=page))
            if args.verbose:
                print('{} done, {} items in total'.format(category.name, len(items)))
    items.remove_duplicates()


def download_item_page(items, io):
    if args.verbose:
        print('\nPage {}, {} distinct items'.format(page, len(items)))
    try:
        update_items(items, page, args.items_per_page)
    finally:
        valid_tags = items.get_valid_tags(args.min_valid_tag)
        if args.verbose:
            print_tags(valid_tags)
        items.update_tags(valid_tags)
        if args.complete_tags_only:
            items = items.filter_items_without_complete_tags()
        io.save_items(items)
        return valid_tags


if __name__ == '__main__':
    args = parse_command_line()

    with open(args.ebay_auth_file) as file:
        auth = json.load(file)

    api = ShoppingAPI(auth['production'], args.ebay_site_id, debug=False)
    categories = Category.search_categories(api)

    io = EbayDownloaderIO(
        args.save_folder, items_file=args.item_file, likes_file=args.likes_file, verbose=args.verbose
    )
    items = io.load_items()
    items = io.import_likes(api, items)

    for page in range(args.page_from, args.page_to + 1):

        valid_tags = download_item_page(items, io)

    if args.download_images:
        for i, item in enumerate(items):
            if args.verbose:
                print(i+1, '/', len(items), end=' ')
            item.download_images(verbose=args.verbose)
