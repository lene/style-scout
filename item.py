from os import remove

from tag_processor import TagProcessor

import re
from collections import defaultdict

from items import Items
class Item:
    """
    An eBay item along with some utility functions to make it useful as input to a neural network.
    """
    download_root = 'data/eBay'
    TAG_LIST = {
        'farbe': 'color', 'länge': 'length', 'stil': 'style', 'anlass': 'occasion',
        'muster': 'pattern', 'absatzhöhe': 'heel height',
        # 'material': 'material', 'obermaterial': 'material',
    }

    def __init__(self, api, category, item_id):
        try:
            item = api.get_item(item_id)
            self.id = item['ItemID']
            self.title = item['Title']
            self.description = self._clean_description(item['Description'])
            self.item_specifics = self._get_specifics(item.get('ItemSpecifics', {}))
            self.picture_urls = item.get('PictureURL', [])
            self._valid = True
        except AttributeError:
            self.item_specifics = {}
            self._valid = False
        finally:
            self.category = category
            self.picture_files = []
            self.tags = set()

    def like(self):
        """Set this Item to liked."""
        self.tags.add('<3')

    @property
    def valid(self):
        return self._valid

    def download_images(self, verbose=False):
        """
        Download the images associated with this Item to self.download_root.
        :param verbose: If set, print a progress message
        :return: None
        """
        try:
            for i, picture_url in enumerate(self.picture_urls):
                downloaded = self.download_image(picture_url, verbose and i == 0)
                if downloaded:
                    self.picture_files.append(downloaded)
        except AttributeError:
            return

    def set_tags(self, all_available_tags):
        """
        Out of all available tags, set those that have a value on this Item.
        :param all_available_tags: all tags that may be set
        :return: None
        """
        self.tags = all_available_tags & self.get_possible_tags()

    def get_possible_tags(self, add_undefined=False):
        """
        Get all tags that have a value on this Item. Possible tags are returned as a set where each
        tag has the form "tag_type:value".
        :param add_undefined: If True, add tags that are unset on this Item as "tag_type:UNDEFINED"
        :return: All tags as  a set where each tag has the form "tag_type:value"
        """
        tags = self.tags or set()
        tags |= set(self.category.name_path[1:])
        for property in self.TAG_LIST.keys():
            tags |= self._tags_for_tag_type(property, add_undefined)

        return tags

    def process_tag(self, tag_label, tag_value):
        """
        Convert the tag value given in proprietary format into the value required for training the
        neural network.
        :param tag_label: Label for the tag
        :param tag_value: Tag value as returned by the eBay API
        :return: Converted value
        """
        processor = TagProcessor(self.TAG_LIST)
        return processor.process_tag(tag_label, tag_value)

    @classmethod
    def download_image(cls, url, show=False):
        """
        Downloads the single image at the URL url to cls.download_root
        :param url: URL to download the image from
        :param show: If True, show (nonblocking) the image for debugging purposes
        :return: None
        """
        from os.path import join, isfile
        from os import makedirs
        from urllib.request import urlretrieve
        from urllib.error import URLError

        makedirs(join(cls.download_root), exist_ok=True)
        filename = join(cls.download_root, '_'.join(url.split('/')[-4:]))
        try:
            if not isfile(filename):
                urlretrieve(url, filename)
            cls._show_image(filename, show)
            return filename
        except URLError:
            return None

    @classmethod
    def _show_image(cls, filename, show):
        from PIL import Image
        if show:
            try:
                # for debugging/following the status, mostly
                with Image.open(filename) as image:
                    # image.show()
                    print(filename, image.size)
            except OSError:
                try:
                    remove(filename)
                except FileNotFoundError:
                    pass

    def __str__(self):
            return """Id: {}
    Title: {}
    Specifics: {}
    Tags: {}
    Pix: {}""".format(self.id, self.title, self.item_specifics, self.tags, self.picture_urls)

    @staticmethod
    def _clean_description(description):
        description = re.sub('<[^<]+?>', '', description)
        description = description.replace('&nbsp;', ' ')
        description = description.replace('\n', ' ')
        return description

    @staticmethod
    def _get_specifics(item_specifics):
        if not item_specifics:
            return {}
        if isinstance(item_specifics['NameValueList'], dict):  # because some people just cannot stick to schemata
            return {
                item_specifics['NameValueList']['Name'].lower(): item_specifics['NameValueList']['Value'].lower()
            }

        specifics = defaultdict(list)
        try:
            for pair in item_specifics.get('NameValueList', []):
                if isinstance(pair['Value'], list):
                    specifics[pair['Name'].lower()].extend(v.lower() for v in pair['Value'])
                else:
                    specifics[pair['Name'].lower()].append(pair['Value'].lower())
            return dict(specifics)
        except (TypeError, AttributeError):
            print(item_specifics)
            raise

    def _tags_for_tag_type(self, property, add_undefined):
        tag_label = self.TAG_LIST[property]
        if property in self.item_specifics:
            return self._tags_for_property(property, tag_label)
        elif add_undefined:
                return {'{}:UNDEFINED'.format(tag_label)}
        return set()

    def _tags_for_property(self, property, tag_label):
        tags = set()
        if not isinstance(self.item_specifics[property], list):
            self.item_specifics[property] = [self.item_specifics[property]]
        tag_values = [s.lower() for s in self.item_specifics[property]]
        for tag_value in tag_values:
            tag_value = self.process_tag(tag_label, tag_value)
            if tag_value:
                if not isinstance(tag_value, list):
                    tag_value = [tag_value]
                tags |= set('{}:{}'.format(tag_label, v) for v in tag_value)
        return tags


class EbayItem(Item):
    pass
