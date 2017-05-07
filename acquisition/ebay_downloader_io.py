
import json
import pickle
from os import makedirs, rename, remove
from os.path import isfile, join

from acquisition.item import Item
from acquisition.items import Items
from category import Category
from data_sets import EbayDataSets
from utils.with_verbose import WithVerbose
from ebaysdk.exception import ConnectionError


class EbayDownloaderIO(WithVerbose):

    def __init__(
            self, base_dir, image_size=None, items_file=None, images_file=None, weights_file=None,
            likes_file=None, verbose=False
    ):
        _check_constructor_arguments_valid(image_size, items_file, images_file, weights_file, likes_file)
        WithVerbose.__init__(self, verbose)
        makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.image_size = image_size
        self.items_file = join(self.base_dir, items_file or _filename('items', 'pickle', None))
        self.images_file = join(self.base_dir, images_file or _filename('images', 'npz', image_size))
        self.weights_file_base = self._weights_file_base(weights_file)
        self.likes_file = self._likes_filename(likes_file)

    def load_items(self):
        """
        Load items already downloaded from pickle file, if present
        :return: Items object containing previously downloaded Item objects
        """
        if isfile(self.items_file):
            self._print_status('Loading', self.items_file)
            with open(self.items_file, 'rb') as file:
                items = pickle.load(file)
                if isinstance(items, Items):
                    return items
                return Items(items, self.verbose)
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

    def get_images(self, items, valid_tags, image_size, test_share=0.2):
        """
        Loads or creates image data set with all images belonging to the given items and the labels
        determined by the passed valid_tags
        :param items: Items object for which the images and labels are returned
        :param valid_tags: List of tags which may be present in the labels of the returned data set 
        :param image_size: size to which the images are resized
        :param test_share: portion of the data set which is used as test data
        :return: data set for the requested parameters
        """
        return EbayDataSets.get_data(
            self.images_file, items, valid_tags, image_size, test_share=test_share, verbose=self.verbose
        )

    def load_weights(self, model, fit_type='', num_items=0):
        """
        Load the precomputed weights for the given neural network
        :param model: Keras model for the neural network
        :param fit_type: currently, 'full' or 'liked'
        :param num_items: number of items in the full data set
        :return: None
        """
        weights_file = self.weights_file(fit_type, num_items)
        if isfile(weights_file):
            self._print_status('Loading', weights_file)
            model.load_weights(weights_file)

    def save_weights(self, model, fit_type='', num_items=0):
        """
        Save the computed weights for the given neural network
        :param model: Keras model for the neural network
        :param fit_type: currently, 'full' or 'liked'
        :param num_items: number of items in the full data set
        :return: None
        """
        weights_file = self.weights_file(fit_type, num_items)
        self._print_status('Saving', weights_file)
        model.save_weights(weights_file)

    def weights_file(self, fit_type='', num_items=0):
        return _filename(
            self.weights_file_base, 'hdf5', fit_type, self._number_to_string(num_items), self.image_size
        )

    @staticmethod
    def _number_to_string(num_items):
        if not num_items:
            return ''
        if 1000*(num_items//1000) == num_items:
            return str(num_items)[:-3] + 'k'
        return str(num_items)

    def _weights_file_base(self, weights_file):
        if weights_file is None:
            weights_file = 'weights'
        return join(self.base_dir, weights_file.replace('.hdf5', ''))

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
            try:
                new_item = Item(api, category, liked)
                new_item.like()
                items.append(new_item)
            except ConnectionError as e:
                print(e)


def _filename(what, extension, *args):
    return "_".join(str(arg) for arg in [what] + list(args) if arg) + ".{}".format(extension)


def _check_constructor_arguments_valid(image_size, items_file, images_file, weights_file, likes_file):
    if images_file or weights_file:
        assert isinstance(image_size, int)

