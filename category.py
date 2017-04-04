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
}

class Category:

    _by_id = {}

    def __init__(self, data):
        self.id = data['CategoryID']
        self.id_path = data.get('CategoryIDPath', '').split(':')
        self.name = data['CategoryName']
        self.name_path = data.get('CategoryNamePath', '').split(':')
        self.is_leaf = data['LeafCategory'] == 'true'
        # also present, but currently not useful:
        # {'CategoryParentID': '15724', 'CategoryLevel': '3'}
        self._by_id[self.id] = self

    @classmethod
    def by_id(cls, id):
        return cls._by_id[str(id)]

    @classmethod
    def search_categories(cls, api, search_term_filter=DEFAULT_CATEGORIES, root_category=-1):
        category_ids = [root_category]
        leaf_categories = []
        for level in range(1, 5):
            next_level_cat_ids = []
            for id in category_ids:
                categories = [
                    category for category in api.categories(id)
                    if any(term.lower() in category.name.lower() for term in search_term_filter[level])
                ]
                leaf_categories += [c for c in categories if c.is_leaf]
                next_level_cat_ids += [c.id for c in categories if not c.is_leaf]
            category_ids = next_level_cat_ids
        return leaf_categories

