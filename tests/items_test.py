
from acquisition.item import Item
from acquisition.items import Items
from tests.test_base import TestBase


class ItemsTest(TestBase):

    def test_getitem(self) -> None:
        raw_items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        items = Items(raw_items)
        self.assertEquals(raw_items[0], items[0])
        self.assertEquals(raw_items[1], items[1])

    def test_getitem_index_error(self) -> None:
        items = self.generate_items(2)
        with self.assertRaises(IndexError):
            items[2]

    def test_len(self) -> None:
        self.assertEquals(0, len(Items([])))
        self.assertEquals(2, len(Items([Item(self.api, self.category, 1), Item(self.api, self.category, 2)])))

    def test_append(self) -> None:
        items = Items([])
        items.append(Item(self.api, self.category, 1))
        self.assertEquals(1, len(items))
        self.assertEquals(1, items[0].id)

    def test_extend_list(self) -> None:
        items = Items([])
        items.extend(Items([Item(self.api, self.category, 1)]))
        self.assertEquals(1, len(items))
        self.assertEquals(1, items[0].id)

    def test_extend_items(self) -> None:
        items = Items([])
        items.extend(self.generate_items(1))
        self.assertEquals(1, len(items))
        self.assertEquals(1, items[0].id)

    def test_set_liked(self) -> None:
        items = self.generate_items(2)
        items.set_liked(1)
        self.assertEqual({'<3'}, items[0].tags)
        self.assertEqual(set(), items[1].tags)

    def test_set_liked_raises_if_not_found(self) -> None:
        items = self.generate_items(2)
        with self.assertRaises(ValueError):
            items.set_liked(3)

    def test_remove_duplicates(self) -> None:
        raw_items = [Item(self.api, self.category, 1), Item(self.api, self.category, 1)]
        items = Items(raw_items)
        items.remove_duplicates()
        self.assertEqual(1, len(items))
        self.assertEqual(raw_items[0], items[0])

    def test_remove_duplicates_without_duplicates(self) -> None:
        raw_items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        items = Items(raw_items)
        items.remove_duplicates()
        self.assertCountEqual(raw_items, items)

    def test_get_valid_tags_returns_category(self) -> None:
        items = self.generate_items(3)
        tags = items.get_valid_tags(2)
        self.assertEqual({self.category.name_path[1]: len(items)}, tags)

    def test_get_valid_tags_returns_tags_with_enough_count(self) -> None:
        items = self.generate_items(3)

        for i in items[:2]:
            items.set_liked(i.id)

        tags = items.get_valid_tags(2)
        self.assertIn('<3', tags.keys())

    def test_get_valid_tags_filters_tags_with_not_enough_count(self) -> None:
        items = self.generate_items(3)

        for i in items[:2]:
            items.set_liked(i.id)

        tags = items.get_valid_tags(3)
        self.assertNotIn('<3', tags.keys())

    def test_filter_items_without_complete_tags(self) -> None:
        item1 = Item(self.api, self.category, 1)
        item1.tags = {'blah:blub'}
        item2 = Item(self.api, self.category, 2)
        item2.tags = {'blah:blub', 'style:cool_shit'}
        self.category.necessary_tags = ['style']

        filtered = Items([item1, item2]).filter_items_without_complete_tags()
        self.assertEqual(1, len(filtered))
        self.assertEqual(item2, filtered[0])

    def test_split_liked_and_unliked_items_evenly(self) -> None:
        items = self.generate_items(10)
        items[0].like()
        self.assertEqual(2, len(items.equal_number_of_liked_and_unliked()))
        liked = [i for i in items if i.is_liked]
        self.assertEqual(1, len(liked))
