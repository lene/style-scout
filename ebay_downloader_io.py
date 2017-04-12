
from os import makedirs, rename, remove
from os.path import isfile, join
import pickle
import json

from data_sets import EbayDataSets
from category import Category
from item import Item


class EbayDownloaderIO:

    def __init__(
            self, base_dir, image_size=None, items_file=None, images_file=None, weights_file=None,
            likes_file=None, verbose=True
    ):
        makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.items_file = join(self.base_dir, items_file or _filename('items', 'pickle', image_size))
        self.images_file = join(self.base_dir, images_file or _filename('images', 'npz', image_size))
        self.weights_file = join(self.base_dir, weights_file or _filename('weights', 'hdf5', image_size))
        self.likes_file = likes_file if isfile(likes_file) \
            else join(self.base_dir, likes_file) if isfile(join(self.base_dir, likes_file)) \
            else None
        self.verbose = verbose

    def load_items(self):
        if isfile(self.items_file):
            self._print_status('Loading', self.items_file)
            with open(self.items_file, 'rb') as file:
                return pickle.load(file)
        return []

    def save_items(self, objects):
        self._print_status('Saving', self.items_file)
        if isfile(self.items_file):
            if isfile(self.items_file + '.bak'):
                remove(self.items_file + '.bak')
            rename(self.items_file, self.items_file + '.bak')
        with open(self.items_file, 'wb') as file:
            pickle.dump(objects, file)

    def import_likes(self, api, items):

        if not self.likes_file or not isfile(self.likes_file):
            return items

        self._print_status('Loading', self.likes_file)

        with open(self.likes_file, 'r') as f:
            liked = json.load(f)

        for category_id, item_ids in liked.items():
            if category_id.isdigit():
                category = Category.by_id(category_id)
                self._print_status('{} ({})'.format(category.name, len(item_ids)))
                add_liked_items(api, items, category, item_ids)

        return items

    def get_images(self, items, valid_tags, image_size):
        return EbayDataSets.get_data(self.images_file, items, valid_tags, image_size, self.verbose)

    def load_weights(self, model):
        if isfile(self.weights_file):
            self._print_status('Loading', self.weights_file)
            model.load_weights(self.weights_file)

    def save_weights(self, model):
        self._print_status('Saving', self.weights_file)
        model.save_weights(self.weights_file)

    def _print_status(self, *args):
        if self.verbose:
            print(*args)


def add_liked_items(api, items, category, liked_item_ids):
    present_item_ids = set(i.id for i in items)
    for liked in liked_item_ids:
        if liked in present_item_ids:
            Item.set_liked(items, liked)
        else:
            new_item = Item(api, category, liked)
            new_item.like()
            items.append(new_item)


def _filename(what, extension, image_size):
    return "{}_{}.{}".format(what, image_size, extension) if image_size else "{}.{}".format(what, extension)
