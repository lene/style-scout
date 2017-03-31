import re

class Item:
    download_root = 'data'
    account = 'eBay'

    def __init__(self, api, item_id):
        item = api.get_item(item_id)
        self.id = item['ItemID']
        self.title = item['Title']
        self.description = self._clean_description(item['Description'])
        self.item_specifics = item.get('ItemSpecifics', {})
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

    @staticmethod
    def _clean_description(description):
        description = re.sub('<[^<]+?>', '', description)
        description = description.replace('&nbsp;', ' ')
        description = description.replace('\n', ' ')
        return description

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
