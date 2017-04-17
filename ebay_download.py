import json

from argparse import ArgumentParser
from operator import itemgetter
from pprint import pprint
from random import randrange

from os.path import isfile

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
    parser.add_argument(
        '--demo', type=int, default=10,
        help='Number of images from the test set to try to predict as demo.'
    )
    parser.add_argument(
        '--test', '-t', action='store_true', help="Run evaluation on test data set"
    )
    parser.add_argument(
        '--predict', help='Image or comma-separated list to get predictions on'
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


def print_prediction(data_set, model):
    i = randrange(len(data_set.input))
    image = data_set.input[i]
    label = data_set.labels[i]
    print('actual values:')
    pprint(data_set.labels_sorted_by_probability(label))
    data_set.show_image(image)
    print('predictions:')
    pprint(
        [(label, prob) for label, prob in data_set.labels_sorted_by_probability(
            model.predict(data_set.input[i:i + 1], batch_size=1, verbose=1)[0]
        ).items() if prob > 0.01]
    )


def predict_images(data_set, model, filenames):
    image_data = data_set.image_data(filenames)
    pprint(
        [(label, prob) for label, prob in data_set.labels_sorted_by_probability(
            model.predict(image_data, batch_size=len(image_data), verbose=1)[0]
        ).items() if prob > 0.01]
    )

if __name__ == '__main__':
    args = parse_command_line()

    with open(args.ebay_auth_file) as file:
        auth = json.load(file)

    api = ShoppingAPI(auth['production'], args.ebay_site_id, debug=False)
    categories = Category.search_categories(api)

    io = EbayDownloaderIO(
        args.save_folder, args.image_size, args.item_file, args.images_file, args.weights_file,
        args.likes_file, args.verbose
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

    from variable_inception import variable_inception

    image_data = io.get_images(items, valid_tags, args.image_size)

    model = variable_inception(input_shape=(*image_data.size, image_data.DEPTH), classes=image_data.num_classes)

    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])

    io.load_weights(model)

    if args.num_epochs:
        train = image_data.train.input.reshape(
            len(image_data.train.input), args.image_size, args.image_size, 3
        )
        print(train.shape, image_data.train.labels.shape)
        model.fit(train, image_data.train.labels, epochs=args.num_epochs)
        io.save_weights(model)

    if args.test:
        test = image_data.test.input.reshape(len(image_data.test.input), args.image_size, args.image_size, 3)
        loss_and_metrics = model.evaluate(test, image_data.test.labels)
        print()
        print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])

    for _ in range(args.demo):
        print_prediction(image_data.train, model)
        # print_prediction(image_data.test)

    predict_images(image_data, model, [f for f in args.predict.split(',') if isfile(f)])
