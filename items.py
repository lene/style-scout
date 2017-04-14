from collections import defaultdict


class Items:

    def __init__(self, raw_items, verbose=False):
        self.items = raw_items
        self.verbose = verbose

    def __iter__(self):
        return self.items.__iter__()

    def __next__(self):
        return next(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def append(self, item):
        self.items.append(item)

    def extend(self, items):
        self.items.extend(items if isinstance(items, list) else items.items)

    def remove_duplicates(self):
        ids = set()
        new_items = []
        for item in self.items:
            if item.id not in ids:
                new_items.append(item)
                ids.add(item.id)
        if self.verbose:
            print(len(self), '->', len(new_items), 'items')
        self.items = new_items

    def get_valid_tags(self, min_count):
        return {
            t: n for t, n in self.count_all_tags().items()
            if n >= min_count
            if 'UNDEFINED' not in t
        }

    def count_all_tags(self):
        counted_tags = defaultdict(int)
        for item in self.items:
            for tag in item.get_possible_tags():
                counted_tags[tag] += 1
        return counted_tags

    def set_liked(self, item_id):
        for item in self.items:
            if item.id == item_id:
                item.like()
                return
        raise ValueError("Item {} not in items".format(item_id))

    def filter_items_without_complete_tags(self):
        def has_complete_tags(item):
            def has_tag_category(item, tag_category):
                return any(tag_category in tag for tag in item.tags)
            return all(has_tag_category(item, tag_category) for tag_category in item.category.necessary_tags)

        old_length = len(self)
        items = [item for item in self.items if has_complete_tags(item)]
        if self.verbose:
            print(old_length, '->', len(items))
        return Items(items, self.verbose)
