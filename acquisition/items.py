from collections import defaultdict

from utils.with_verbose import WithVerbose


class Items(WithVerbose):
    """
    A set of Item objects along with some utility functions.
    """
    def __init__(self, raw_items, verbose=False):
        WithVerbose.__init__(self, verbose)
        self.items = raw_items

    def __iter__(self):
        return self.items.__iter__()

    def __next__(self):
        return next(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def append(self, item):
        """Adds an Item to this item set."""
        self.items.append(item)

    def extend(self, items):
        """Adds a list of Item objects or an Items object to this item set."""
        self.items.extend(items if isinstance(items, list) else items.items)

    def remove_duplicates(self):
        """
        Remove all duplicate occurrences of an Item in this item set. 
        :return: None
        """
        # Since Item is not hashable we need to do this manually
        ids = set()
        new_items = []
        for item in self.items:
            if item.id not in ids:
                new_items.append(item)
                ids.add(item.id)
        self._print_status(len(self), '->', len(new_items), 'items')
        self.items = new_items

    def download_images(self):
        for i, item in enumerate(self.items):
            self._print_status('Downloading images ({}/{})'.format(i + 1, len(self.items)), end='\r')
            item.download_images()
        self._print_status()


    def get_valid_tags(self, min_count):
        """
        Returns the tags in this item set which occur at least min_count times, along with the 
        number of times each tag occurs.
        :param min_count: minimum number of occurrences for a tag to be included
        :return: a dict of the form {tag: number_it_occurs} 
        """
        return {
            t: n for t, n in self.count_all_tags().items()
            if n >= min_count
            if 'UNDEFINED' not in t
        }

    def count_all_tags(self):
        """
        Returns the tags in this item set along with the number of times each tag occurs.
        :return: a dict of the form {tag: number_it_occurs} 
        """
        counted_tags = defaultdict(int)
        for item in self.items:
            for tag in item.get_possible_tags():
                counted_tags[tag] += 1
        return counted_tags

    def set_liked(self, item_id):
        """
        Sets an Item in this item set with the specified ID to liked.
        :param item_id: The ID of the item to be liked.
        :return: None
        """
        for item in self.items:
            if item.id == item_id:
                item.like()
                return
        raise ValueError("Item {} not in items".format(item_id))

    def filter_items_without_complete_tags(self):
        """
        Removes all Item object in this item set which do not have all tags set that are required
        for successful training of the neural network.
        The exact necessary tags vary by Category.
        :return: An Items object containing only Item objects with all necessary tags.
        """
        def has_complete_tags(item):
            def has_tag_category(item, tag_category):
                return any(tag_category in tag for tag in item.tags)
            return all(has_tag_category(item, tag_category) for tag_category in item.category.necessary_tags)

        old_length = len(self)
        items = [item for item in self.items if has_complete_tags(item)]
        self._print_status(old_length, '->', len(items))
        return Items(items, self.verbose)

    def update_tags(self, valid_tags):
        for item in self.items:
            item.set_tags(set(valid_tags.keys()))
