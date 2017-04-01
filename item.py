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
            self.category = category
            self.id = item['ItemID']
            self.title = item['Title']
            self.description = self._clean_description(item['Description'])
            self.item_specifics = self._get_specifics(item.get('ItemSpecifics', {}))
            self.picture_urls = item.get('PictureURL', [])
            self.picture_files = []
            self.tags = set()
        except AttributeError:
            self.item_specifics = {}

    def __str__(self):
        return """Id: {}
Title: {}
Specifics: {}
Tags: {}
Pix: {}""".format(self.id, self.title, self.item_specifics, self.tags, self.picture_urls)

    def like(self):
        self.tags.add('<3')

    def download_images(self, verbose=False):
        for i, picture_url in enumerate(self.picture_urls):
            self.download_image(picture_url, verbose and i == 0)

    def set_tags(self, all_available_tags):
        self.tags = all_available_tags & self.get_possible_tags()

    def get_possible_tags(self):
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
                tags.add('{}:UNDEFINED'.format(tag_label))

        return tags

    @classmethod
    def set_liked(cls, items, item_id):
        for item in items:
            if item.id == item_id:
                item.like()
                return
        raise ValueError("Item {} not in items".format(item_id))

    def process_tag(self, tag_label, tag_value):
        if tag_label == 'color':
            tag_value = self.process_color_tag(tag_value)
        elif tag_label == 'length':
            tag_value = self.process_length_tag(tag_value)
        elif tag_label == 'style':
            tag_value = self.process_style_tag(tag_value)
        elif tag_label == 'occasion':
            tag_value = self.process_occasion_tag(tag_value)
        elif tag_label == 'pattern':
            tag_value = self.process_pattern_tag(tag_value)
        elif tag_label == 'heel height':
            tag_value = self.process_heel_height_tag(tag_value)
        return tag_value

    @staticmethod
    def process_color_tag(tag_value):
        if 'hell' == tag_value[:4]:
            tag_value = tag_value[4:]
        elif 'dunkel' == tag_value[:6]:
            tag_value = tag_value[6:].strip()

        if 'schwarz' in tag_value:
            if 'weiß' in tag_value or 'weiss' in tag_value:
                tag_value = 'schwarz-weiß'
            elif 'grau' in tag_value:
                tag_value = 'schwarz-grau'
            elif 'rot' in tag_value:
                tag_value = 'schwarz-rot'
            elif 'gold' in tag_value:
                tag_value = 'schwarz-gold'
            elif 'silber' in tag_value:
                tag_value = 'schwarz-silber'
            elif 'beige' in tag_value:
                tag_value = 'schwarz-beige'
            elif 'töne' in tag_value:
                tag_value = 'schwarz'
        elif 'weiss' in tag_value or 'weißtöne' in tag_value or 'weiã' in tag_value or 'wollweiß' in tag_value:
            return 'weiß'
        elif 'braun' in tag_value:
            return 'braun'
        elif 'rot' in tag_value:
            return 'rot'
        elif 'blau' in tag_value or 'marine' == tag_value:
            return 'blau'
        elif 'mint' in tag_value:
            return 'mint'
        elif 'grün' in tag_value or 'grã¼n' in tag_value:
            return 'grün'
        elif tag_value[:3] == 'ros' or tag_value[-4:-1] == 'ros':
            return 'rosa'
        elif tag_value[:4] == 'oliv':
            return 'oliv'
        elif 'grau' in tag_value:
            return 'grau'
        elif 'beige' in tag_value:
            return 'beige'
        elif 'bunt' == tag_value:
            return 'mehrfarbig'
        elif 'bordo' == tag_value:
            return 'bordeaux'
        elif 'creme' in tag_value or 'ecru' == tag_value:
            return 'creme'
        elif 'haut' == tag_value:
            return 'nude'

        return tag_value

    @staticmethod
    def process_length_tag(tag_value):
        return tag_value

    @staticmethod
    def process_style_tag(tag_value):
        tag_value = tag_value.replace('//', '/')
        if tag_value[:9] == 'nachthemd':
            tag_value = 'nachthemden & -shirts'
        elif tag_value[:5] == 'bluse':
            tag_value = 'blusen'
        elif tag_value[:9] == 'halbschuh':
            tag_value = 'halbschuhe'
        elif tag_value == 'kostã¼m':
            tag_value = 'kostüm'
        elif 'mokassin' in tag_value:
            tag_value = 'loafers, mokassins'
        elif 'pantolette' in tag_value:
            tag_value = 'pantolette'
        elif tag_value == 'sandale':
            tag_value = 'sandalen'
        elif tag_value == 'sandalette':
            tag_value = 'sandaletten'
        elif tag_value == 'slipper schuhe':
            tag_value = 'slipper'
        elif tag_value == 'sneaker':
            tag_value = 'sneakers'
        elif tag_value == 'stiefelette':
            tag_value = 'stiefeletten'
        elif tag_value == 'tunika':
            tag_value = 'tuniken'
        elif tag_value == 'shirt':
            return 'shirts'
        elif tag_value[:5] == 'jeans':
            return 'jeans'
        elif tag_value[:5] == 'stepp':
            return 'stepp'
        elif tag_value == 'kleid':
            return None
        elif tag_value[:8] == 'sonstige':
            return None
        return tag_value

    @staticmethod
    def process_occasion_tag(tag_value):
        if tag_value == 'formal' or tag_value == 'büro':
            return 'business'
        elif 'business' in tag_value and 'freizeit' in tag_value:
            return ['business', 'freizeit']
        elif tag_value == 'abendlich':
            return 'festlich'
        elif tag_value == 'wandern/trekking':
            return 'outdoor'
        elif tag_value == 'clubwear':
            return 'party'
        elif tag_value[:8] == 'hochzeit':
            return 'spezieller anlass'
        elif tag_value == 'immer' or tag_value == 'alles' or tag_value == 'freizeit, besondere anlässe, party, arbeit':
            return None
        return tag_value

    @staticmethod
    def process_pattern_tag(tag_value):
        if tag_value == 'ohne' or tag_value == 'einfarbig' or tag_value == 'kein muster' or tag_value[:3] == 'uni':
            return 'ohne muster'
        elif tag_value[:6] == 'siehe ' or tag_value == 'mit muster':
            return None
        elif tag_value[:6] == 'blumen' or tag_value == 'floral':
            return 'geblümt'
        elif tag_value[:8] == 'streifen':
            return 'gestreift'
        elif tag_value == 'bedruckt':
            return 'mit motiv'
        return tag_value

    @staticmethod
    def process_heel_height_tag(tag_value):
        if 'ca.' == tag_value[:3]:
            tag_value = tag_value[3:].strip()
        height = None
        if re.match('\d', tag_value[0]):
            height = int(tag_value[0])
        if height is not None:
            if height < 3:
                return 'kleiner absatz (kleiner als 3 cm)'
            elif height <= 5:
                return 'mittlerer absatz (3-5 cm)'
            elif height <= 8:
                return 'hoher absatz (5-8 cm)'
            else:
                return 'sehr hoher absatz (größer als 8 cm)'
        return tag_value

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
                    image.show()
                    # print(filename, image.size)
        except URLError:
            pass

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