from collections import defaultdict
from datetime import timedelta
from random import sample, seed, shuffle
from time import time
from typing import List, Union, Dict, Sized, Iterable, Set, Iterator, overload

from acquisition.item import Item
from category import Category
from utils.with_verbose import WithVerbose


class Items(WithVerbose, Sized, Iterable):
    """
    A set of Item objects along with some utility functions.
    """
    def __init__(
            self, raw_items: Union['Items', List[Item]], verbose: bool=False, is_download_complete: bool=False
    ) -> None:
        WithVerbose.__init__(self, verbose)
        self.items = raw_items
        self.is_download_complete = is_download_complete

    def __iter__(self) -> Iterator:
        return self.items.__iter__()

    def __next__(self):
        return next(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @overload
    def __getitem__(self, item: slice) -> 'Items':
        pass

    @overload
    def __getitem__(self, item: int) -> Item:
        pass

    def __getitem__(self, item):
        return self.items[item]

    def append(self, item: Item) -> None:
        """Adds an Item to this item set."""
        self.items.append(item)

    def extend(self, items: 'Items') -> None:
        """Adds a list of Item objects or an Items object to this item set."""
        self.items.extend(items)

    def categories(self) -> Set[Category]:
        return {i.category for i in self.items}

    def filter(self, category: Category=None) -> 'Items':
        if not category:
            raise ValueError()
        return Items(
            [item for item in self.items if item.category.name.lower().startswith(category.name.lower())],
            self.verbose, self.is_download_complete
        )

    def remove_duplicates(self) -> None:
        """
        Remove all duplicate occurrences of an Item in this item set.
        :return: None
        """
        # Since Item is not hashable we need to do this manually
        ids = set()  # type: Set[int]
        new_items = []
        for item in self.items:
            if item.id not in ids:
                new_items.append(item)
                ids.add(item.id)
        self._print_status(len(self), '->', len(new_items), 'items')
        self.items = new_items

    def remove_crap(self) -> None:
        new_items = [item for item in self.items if hasattr(item, 'id') and hasattr(item, 'picture_urls')]
        self._print_status(len(self), '->', len(new_items), 'items')
        self.items = new_items

    def download_images(self) -> None:
        if self.is_download_complete:
            return
        start_time = time()
        elapsed_time = 0.
        for i, item in enumerate(self.items):
            elapsed_time = time() - start_time
            self._print_status(
                'Downloading images ({}/{}) ETA: {}'.format(
                    i + 1, len(self.items),
                    timedelta(seconds=int(elapsed_time * (len(self) - i) / (i + 1)))
                ), end='\r'
            )
            item.download_images()
        self.items = [item for item in self.items if item.picture_files]
        self.is_download_complete = True
        self._print_status(
            '{} items downloaded in {}'.format(len(self), timedelta(seconds=int(elapsed_time))) + ' ' * 12
        )

    def get_valid_tags(self, min_count: int) -> Dict[str, int]:
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

    def count_all_tags(self) -> Dict[str, int]:
        """
        Returns the tags in this item set along with the number of times each tag occurs.
        :return: a dict of the form {tag: number_it_occurs}
        """
        counted_tags = defaultdict(int)  # type: Dict[str, int]
        for item in self.items:
            for tag in item.get_possible_tags():
                counted_tags[tag] += 1
        return counted_tags

    def set_liked(self, item_id: int) -> None:
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

    def filter_items_without_complete_tags(self) -> 'Items':
        """
        Removes all Item object in this item set which do not have all tags set that are required
        for successful training of the neural network.
        The exact necessary tags vary by Category.
        :return: An Items object containing only Item objects with all necessary tags.
        """
        def has_complete_tags(item: Item) -> bool:
            def has_tag_category(item: Item, tag_category: str) -> bool:
                return any(tag_category in tag for tag in item.tags)
            return all(has_tag_category(item, tag_category) for tag_category in item.category.necessary_tags)

        old_length = len(self)
        items = [item for item in self.items if has_complete_tags(item)]
        self._print_status(old_length, '->', len(items))
        return Items(items, self.verbose)

    def update_tags(self, valid_tags: Dict[str, int]) -> None:
        for item in self.items:
            item.set_tags(set(valid_tags.keys()))

    def equal_number_of_liked_and_unliked(self, random_seed: int=None) -> 'Items':
        seed(random_seed)
        liked = list(filter(lambda item: '<3' in item.tags, self.items))
        unliked = sample(list(filter(lambda item: '<3' not in item.tags, self.items)), len(liked))
        all_items = liked + unliked
        shuffle(all_items)
        return Items(all_items)
