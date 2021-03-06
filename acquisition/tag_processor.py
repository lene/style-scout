import re
from typing import Dict, Set


class TagProcessor:
    """
    Given tag values in an arbitrary, proprietary format, convert them to values in accordance with
    what the neural network requires/expects.
    """

    def __init__(self, tag_list: Dict[str, str]) -> None:
        self.tag_list = tag_list

    def process_tag(self, tag_label: str, tag_value: str) -> Set[str]:
        if tag_label == 'color':
            return {self.process_color_tag(tag_value)}
        elif tag_label == 'length':
            return {self.process_length_tag(tag_value)}
        elif tag_label == 'style':
            return self.process_style_tag(tag_value)
        elif tag_label == 'occasion':
            return self.process_occasion_tag(tag_value)
        elif tag_label == 'pattern':
            return self.process_pattern_tag(tag_value)
        elif tag_label == 'heel height':
            return {self.process_heel_height_tag(tag_value)}
        return {tag_value}

    @staticmethod
    def process_color_tag(tag_value: str) -> str:
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
        elif 'weiss' in tag_value or 'weißtöne' in tag_value or 'weiã' in tag_value \
                or 'wollweiß' in tag_value:
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
    def process_length_tag(tag_value: str) -> str:
        return tag_value

    @staticmethod
    def process_style_tag(tag_value: str) -> Set[str]:
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
            return {'shirts'}
        elif tag_value[:5] == 'jeans':
            return {'jeans'}
        elif tag_value[:5] == 'stepp':
            return {'stepp'}
        elif tag_value == 'kleid':
            return set()
        elif tag_value[:8] == 'sonstige':
            return set()
        return {tag_value}

    @staticmethod
    def process_occasion_tag(tag_value: str) -> Set[str]:
        if tag_value == 'formal' or tag_value == 'büro':
            return {'business'}
        elif 'business' in tag_value and 'freizeit' in tag_value:
            return {'business', 'freizeit'}
        elif tag_value == 'abendlich':
            return {'festlich'}
        elif tag_value == 'wandern/trekking':
            return {'outdoor'}
        elif tag_value == 'clubwear':
            return {'party'}
        elif tag_value[:8] == 'hochzeit':
            return {'spezieller anlass'}
        elif tag_value == 'immer' or tag_value == 'alles' \
                or tag_value == 'freizeit, besondere anlässe, party, arbeit':
            return set()
        return {tag_value}

    @staticmethod
    def process_pattern_tag(tag_value: str) -> Set[str]:
        if tag_value == 'ohne' or tag_value == 'einfarbig' or tag_value == 'kein muster' \
                or tag_value[:3] == 'uni':
            return {'ohne muster'}
        elif tag_value[:6] == 'siehe ' or tag_value == 'mit muster':
            return set()
        elif tag_value[:6] == 'blumen' or tag_value == 'floral':
            return {'geblümt'}
        elif tag_value[:8] == 'streifen':
            return {'gestreift'}
        elif tag_value == 'bedruckt':
            return {'mit motiv'}
        return {tag_value}

    @staticmethod
    def process_heel_height_tag(tag_value: str) -> str:
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
