import re
from os.path import join, isfile
from collections import defaultdict
from os import remove, makedirs
from urllib.request import urlretrieve
from urllib.error import URLError, ContentTooShortError
from http.client import RemoteDisconnected
import concurrent.futures
import asyncio

from acquisition.tag_processor import TagProcessor


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
    MAX_DOWNLOAD_THREADS = 18  # limit imposed by the eBay API (actually we don't use the API, but BSTS)

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

    def download_images(self):
        """
        Download the images associated with this Item to self.download_root.
        :param verbose: If set, print a progress message
        :return: None
        """
        makedirs(self.download_root, exist_ok=True)
        if len(self.picture_files) == len(self.picture_urls) and all(is_image_file(f) for f in self.picture_files):
            return

        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self._download_images(max_threads=self.MAX_DOWNLOAD_THREADS))
        except ContentTooShortError as e:
            print('\n{}, retrying...'.format(e))
            loop.run_until_complete(self._download_images(max_threads=self.MAX_DOWNLOAD_THREADS//4))
        except (URLError, RemoteDisconnected):
            pass

        self.picture_files = [
            self.url_to_file(url) for url in self.picture_urls if is_image_file(self.url_to_file(url))
        ]

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
    def url_to_file(cls, url):
        return join(cls.download_root, '_'.join(url.split('/')[-4:]))

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

    async def _download_images(self, max_threads):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(executor, urlretrieve, url, self.url_to_file(url))
                for url in self.picture_urls if not is_image_file(self.url_to_file(url))
            ]
            await asyncio.gather(*futures)

    def __str__(self):
            return """Id: {}
    Title: {} {}
    Specifics: {}
    Tags: {}
    Pix: {}""".format(
                self.id, self.title,
                '\n    <3' if '<3' in self.tags else '',
                self.item_specifics, self.tags, self.picture_urls
            )

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


def is_image_file(filename):
    from PIL import Image
    if not isfile(filename):
        return False
    try:
        Image.open(filename)
        return True
    except OSError:
        return False

