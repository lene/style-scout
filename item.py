from tag_processor import TagProcessor

import re
from collections import defaultdict


class Item:

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
        self.tags.add('<3')

    @property
    def valid(self):
        return self._valid

    def download_images(self, verbose=False):
        try:
            for i, picture_url in enumerate(self.picture_urls):
                downloaded = self.download_image(picture_url, verbose and i == 0)
                if downloaded:
                    self.picture_files.append(downloaded)
        except AttributeError:
            return

    def set_tags(self, all_available_tags):
        self.tags = all_available_tags & self.get_possible_tags()

    def get_possible_tags(self, add_undefined=False):
        tags = set(self.category.name_path[1:])
        for specifics in self.TAG_LIST.keys():
            tag_label = self.TAG_LIST[specifics]
            if specifics in self.item_specifics:
                if not isinstance(self.item_specifics[specifics], list):
                    self.item_specifics[specifics] = [self.item_specifics[specifics]]
                tag_values = [s.lower() for s in self.item_specifics[specifics]]
                for tag_value in tag_values:
                    tag_value = self.process_tag(tag_label, tag_value)
                    if tag_value:
                        if not isinstance(tag_value, list):
                            tag_value = [tag_value]
                        for v in tag_value:
                            tags.add('{}:{}'.format(tag_label, v))
            else:
                if add_undefined:
                    tags.add('{}:UNDEFINED'.format(tag_label))

        return tags | self.tags

    def process_tag(self, tag_label, tag_value):
        processor = TagProcessor(self.TAG_LIST)
        return processor.process_tag(tag_label, tag_value)

    @classmethod
    def download_image(cls, url, show=False):
        from os.path import join, isfile
        from os import makedirs
        from PIL import Image
        from urllib.request import urlretrieve
        from urllib.error import URLError

        try:
            makedirs(join(cls.download_root), exist_ok=True)
            filename = join(cls.download_root, '_'.join(url.split('/')[-4:]))
            if not isfile(filename):
                urlretrieve(url, filename)
            # for debugging/following the status, mostly
            with Image.open(filename) as image:
                if show:
                    # image.show()
                    print(filename, image.size)
            return filename
        except URLError:
            return None

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

