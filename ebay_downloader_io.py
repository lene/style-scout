
from os import makedirs, rename, remove
from os.path import isfile, join
import pickle
import json

from data_sets import EbayDataSets
from category import Category
from item import Item
from items import Items


class EbayDownloaderIO:

    def __init__(
            self, base_dir, image_size=None, items_file=None, images_file=None, weights_file=None,
            likes_file=None, verbose=False
    ):
        makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.items_file = join(self.base_dir, items_file or _filename('items', 'pickle', None))
        self.images_file = join(self.base_dir, images_file or _filename('images', 'npz', image_size))
        self.weights_file = join(self.base_dir, weights_file or _filename('weights', 'hdf5', image_size))
        self.likes_file = self._likes_filename(likes_file)
        self.verbose = verbose

    def load_items(self):
        """
        Load items already downloaded from pickle file, if present
        :return: Items object containing previously downloaded Item objects
        """
        if isfile(self.items_file):
            self._print_status('Loading', self.items_file)
            with open(self.items_file, 'rb') as file:
                return Items(pickle.load(file), self.verbose)
        return Items([], self.verbose)

    def save_items(self, items):
        """
        Store given Items object to pickle file
        :param items: Items to store
        :return: None
        """
        assert isinstance(items, Items)
        self._print_status('Saving', self.items_file)
        if isfile(self.items_file):
            if isfile(self.items_file + '.bak'):
                remove(self.items_file + '.bak')
            rename(self.items_file, self.items_file + '.bak')
        with open(self.items_file, 'wb') as file:
            pickle.dump(items, file)

    def import_likes(self, api, items):
        """
        Loads liked Item objects from the configured likes file and adds them to items, ensuring 
        each Item object is present only once.
        :param api: API object from which Item objects are read
        :param items: Item objects already present
        :return: Items object containing both the previous and the liked Item objects
        """
        if not self.likes_file or not isfile(self.likes_file):
            return items

        self._print_status('Loading', self.likes_file)

        with open(self.likes_file, 'r') as f:
            liked = json.load(f)

        for category_id, item_ids in liked.items():
            if category_id.isdigit():
                category = Category.by_id(category_id)
                self._print_status('{} ({})'.format(category.name, len(item_ids)))
                _add_liked_items(api, items, category, item_ids)

        return items

    def get_images(self, items, valid_tags, image_size):
        """
        Loads or creates image data set with all images belonging to the given items and the labels
        determined by the passed valid_tags
        :param items: Items object for which the images and labels are returned
        :param valid_tags: List of tags which may be present in the labels of the returned data set 
        :param image_size: size to which the images are resized
        :return: data set for the requested parameters
        """
        return EbayDataSets.get_data(self.images_file, items, valid_tags, image_size, verbose=self.verbose)

    def load_weights(self, model):
        """
        Load the precomputed weights for the given neural network
        :param model: Keras model for the neural network
        :return: None
        """
        if isfile(self.weights_file):
            self._print_status('Loading', self.weights_file)
            model.load_weights(self.weights_file)

    def save_weights(self, model):
        """
        Save the computed weights for the given neural network
        :param model: Keras model for the neural network
        :return: None
        """
        self._print_status('Saving', self.weights_file)
        model.save_weights(self.weights_file)

    def _print_status(self, *args):
        if self.verbose:
            print(*args)

    def _likes_filename(self, likes_file):
        return None if not likes_file \
            else likes_file if isfile(likes_file) \
            else join(self.base_dir, likes_file) if isfile(join(self.base_dir, likes_file)) \
            else None


def _add_liked_items(api, items, category, liked_item_ids):
    present_item_ids = set(i.id for i in items)
    for liked in liked_item_ids:
        if liked in present_item_ids:
            items.set_liked(liked)
        else:
            new_item = Item(api, category, liked)
            new_item.like()
            items.append(new_item)


def _filename(what, extension, image_size):
    return "{}_{}.{}".format(what, image_size, extension) if image_size else "{}.{}".format(what, extension)
