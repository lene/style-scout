import re
from os.path import join, isfile
from collections import defaultdict
from os import remove, makedirs
from typing import Set, Dict, List
from urllib.request import urlretrieve
from urllib.error import URLError, ContentTooShortError
from http.client import RemoteDisconnected
import concurrent.futures
import asyncio

# from acquisition.shopping_api import ShoppingAPI
from acquisition.tag_processor import TagProcessor
from category import Category


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

    def __init__(self, api: 'ShoppingAPI', category: Category, item_id: int) -> None:
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
            self.picture_files = []  # type: List[str]
            self.tags = set()  # type: Set[str]

    def like(self) -> None:
        """Set this Item to liked."""
        self.tags.add('<3')

    def unlike(self) -> None:
        """Set this Item to liked."""
        self.tags.discard('<3')

    @property
    def is_liked(self) -> bool:
        return '<3' in self.tags

    @property
    def valid(self) -> bool:
        return self._valid

    def download_images(self) -> None:
        """
        Download the images associated with this Item to self.download_root.
        :param verbose: If set, print a progress message
        :return: None
        """
        makedirs(self.download_root, exist_ok=True)
        if len(self.picture_files) == len(self.picture_urls) \
                and all(is_image_file(f) for f in self.picture_files):
            return

        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self._download_images(max_threads=self.MAX_DOWNLOAD_THREADS))
        except ContentTooShortError as e:
            print('\n{}, retrying...'.format(e))
            loop.run_until_complete(self._download_images(max_threads=self.MAX_DOWNLOAD_THREADS // 4))
        except (URLError, RemoteDisconnected):
            pass

        self.picture_files = [
            self.url_to_file(url) for url in self.picture_urls if is_image_file(self.url_to_file(url))
        ]

    def set_tags(self, all_available_tags: Set[str]) -> None:
        """
        Out of all available tags, set those that have a value on this Item.
        :param all_available_tags: all tags that may be set
        :return: None
        """
        self.tags = all_available_tags & self.get_possible_tags()

    def get_possible_tags(self, add_undefined: bool=False) -> Set[str]:
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

    def process_tag(self, tag_label: str, tag_value: str) -> Set[str]:
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
    def url_to_file(cls, url: str) -> str:
        return join(cls.download_root, '_'.join(url.split('/')[-4:]))

    @classmethod
    def _show_image(cls, filename: str, show: bool) -> None:
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

    async def _download_images(self, max_threads: int) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(executor, urlretrieve, url, self.url_to_file(url))
                for url in self.picture_urls if not is_image_file(self.url_to_file(url))
            ]
            await asyncio.gather(*futures)

    def __str__(self) -> str:
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
    def _clean_description(description: str) -> str:
        description = re.sub('<[^<]+?>', '', description)
        description = description.replace('&nbsp;', ' ')
        description = description.replace('\n', ' ')
        return description

    @staticmethod
    def _get_specifics(item_specifics: Dict) -> Dict:
        if not item_specifics:
            return {}
        if isinstance(item_specifics['NameValueList'], dict):  # some people just cannot stick to schemata
            return {
                item_specifics['NameValueList']['Name'].lower():
                    item_specifics['NameValueList']['Value'].lower()
            }

        specifics = defaultdict(list)  # type: Dict[str, List]
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

    def _tags_for_tag_type(self, property: str, add_undefined: bool) -> Set[str]:
        tag_label = self.TAG_LIST[property]
        if property in self.item_specifics:
            return self._tags_for_property(property, tag_label)
        elif add_undefined:
                return {'{}:UNDEFINED'.format(tag_label)}
        return set()

    def _tags_for_property(self, property: str, tag_label: str) -> Set[str]:
        if not isinstance(self.item_specifics[property], list):
            self.item_specifics[property] = [self.item_specifics[property]]
        tag_values = {s.lower() for s in self.item_specifics[property]}
        return {
            '{}:{}'.format(tag_label, v)
            for tag_value in tag_values for v in self.process_tag(tag_label, tag_value)
        }


class EbayItem(Item):
    pass


def is_image_file(filename: str) -> bool:
    from PIL import Image
    if not isfile(filename):
        return False
    try:
        Image.open(filename)
        return True
    except OSError:
        return False
