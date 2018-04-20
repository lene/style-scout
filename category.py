from typing import Dict, List, Tuple


DEFAULT_CATEGORIES = {
    1: ('Kleidung',),
    2: ('Damenmode', 'Damenschuhe'),
    3: (
        'Anzüge & Kombinationen', 'Blusen, Tops & Shirts',
        'Jacken & Mäntel', 'Kleider', 'Röcke', 'Damenunterwäsche',
        'Halbschuhe & Ballerinas', 'Pumps', 'Sandalen & Badeschuhe',
        'Stiefel & Stiefeletten'
    ),
    4: ('Bodys', 'Nachtwäsche')
}  # type: Dict[int, Tuple[str, ...]]


class Category:

    NECESSARY_TAGS = {
        'Anzüge & Kombinationen': ['color', 'style', ],
        'Blusen, Tops & Shirts': ['color', 'style', ],
        'Jacken & Mäntel': ['color', 'length', 'style', ],
        'Kleider': ['color', 'length', 'style', ],
        'Röcke': ['color', 'length', 'style', ],
        'Halbschuhe & Ballerinas': ['color', 'heel height', 'style', ],
        'Pumps': ['color', 'heel height', 'style', ],
        'Sandalen & Badeschuhe': ['color', 'heel height', 'style', ],
        'Stiefel & Stiefeletten': ['color', 'heel height', 'style', ],
        'Bodys': ['color', ],
        'Nachtwäsche': ['color', 'style', 'length', ]
    }
    _by_id = {}  # type: Dict[str, Category]

    def __init__(self, data: Dict) -> None:
        self.id = data['CategoryID']
        self.id_path = data.get('CategoryIDPath', '').split(':')
        self.name = data['CategoryName']
        self.name_path = data.get('CategoryNamePath', '').split(':')
        self.is_leaf = data['LeafCategory'] == 'true'
        self._by_id[self.id] = self
        # self.parent = self.by_id(data.get('CategoryParentID')) if int(data.get('CategoryParentID', 0))
        #     else None

    @property
    def necessary_tags(self) -> List[str]:
        return self.NECESSARY_TAGS.get(self.name, [])

    @classmethod
    def by_id(cls, id: int) -> 'Category':
        return cls._by_id[str(id)]

    @classmethod
    def search_categories(
            cls, api: 'ShoppingAPI', search_term_filter: Dict[int, Tuple[str, ...]]=DEFAULT_CATEGORIES,
            root_category: int=-1
    ) -> List['Category']:
        category_ids = [root_category]
        leaf_categories = []  # type: List[Category]
        for level in range(1, 5):
            next_level_cat_ids = []  # type: List[int]
            for id in category_ids:
                categories = [
                    category for category in api.categories(id)
                    if any(term.lower() in category.name.lower() for term in search_term_filter[level])
                ]
                leaf_categories += [c for c in categories if c.is_leaf]
                next_level_cat_ids += [c.id for c in categories if not c.is_leaf]
            category_ids = next_level_cat_ids
        return leaf_categories
