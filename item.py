import re


class Item:

    download_root = 'data'
    account = 'eBay'
    TAG_LIST = {
        'Farbe': 'color', 'Länge': 'length', 'Stil': 'style', 'Anlass': 'occasion',
        'Muster': 'pattern', 'Material': 'material', 'Obermaterial': 'material',
        'Absatzhöhe': 'heel height', 'Wäschegröße': 'size', 'Größe': 'size',
        'Schuhgröße': 'shoe size'
    }

    def __init__(self, api, category, item_id):
        item = api.get_item(item_id)
        self.category = category
        self.id = item['ItemID']
        self.title = item['Title']
        self.description = self._clean_description(item['Description'])
        self.item_specifics = self._get_specifics(item.get('ItemSpecifics', {}))
        self.picture_urls = item['PictureURL']
        self.picture_files = []

    def __str__(self):
        return """Id: {}
Title: {}
{}
Specifics: {}
Pix: {}""".format(self.id, self.title, self.description, self.item_specifics, self.picture_urls)

    def download_images(self):
        for i, picture_url in enumerate(self.picture_urls):
            self.download_image(picture_url, i == 0)

    def get_tags(self):
        tags = set(self.category.name_path)
        for specifics in self.TAG_LIST.keys():
            if specifics in self.item_specifics:
                tags.add(
                    '{}:{}'.format(
                        self.TAG_LIST[specifics], self.item_specifics[specifics].lower()
                    )
                )
        return tags

    @classmethod
    def download_image(cls, url, show=False):
        from os.path import join, isfile
        from os import makedirs
        from PIL import Image
        from urllib.request import urlretrieve

        makedirs(join(cls.download_root, cls.account), exist_ok=True)
        filename = join(cls.download_root, cls.account, '_'.join(url.split('/')[-4:]))
        if not isfile(filename):
            urlretrieve(url, filename)
        # for debugging/following the status, mostly
        with Image.open(filename) as image:
            if show:
                image.show()
                # print(filename, image.size)

    @staticmethod
    def _clean_description(description):
        description = re.sub('<[^<]+?>', '', description)
        description = description.replace('&nbsp;', ' ')
        description = description.replace('\n', ' ')
        return description

    @staticmethod
    def _get_specifics(item_specifics):
        try:
            return {pair['Name']: pair['Value'] for pair in item_specifics.get('NameValueList', [])}
        except TypeError:
            return {item_specifics['NameValueList']['Name']: item_specifics['NameValueList']['Value']}
