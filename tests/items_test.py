
from tests.test_base import TestBase
from item import Item
from items import Items


class ItemsTest(TestBase):

    def test_getitem(self):
        raw_items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        items = Items(raw_items)
        self.assertEquals(raw_items[0], items[0])
        self.assertEquals(raw_items[1], items[1])

    def test_getitem_index_error(self):
        raw_items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        items = Items(raw_items)
        with self.assertRaises(IndexError):
            items[2]

    def test_len(self):
        self.assertEquals(0, len(Items([])))
        self.assertEquals(2, len(Items([Item(self.api, self.category, 1), Item(self.api, self.category, 2)])))

    def test_append(self):
        items = Items([])
        items.append(Item(self.api, self.category, 1))
        self.assertEquals(1, len(items))
        self.assertEquals(1, items[0].id)

    def test_extend_list(self):
        items = Items([])
        items.extend([Item(self.api, self.category, 1)])
        self.assertEquals(1, len(items))
        self.assertEquals(1, items[0].id)

    def test_extend_items(self):
        items = Items([])
        items.extend(Items([Item(self.api, self.category, 1)]))
        self.assertEquals(1, len(items))
        self.assertEquals(1, items[0].id)

    def test_set_liked(self):
        items = Items([Item(self.api, self.category, 1), Item(self.api, self.category, 2)])
        items.set_liked(1)
        self.assertEqual({'<3'}, items[0].tags)
        self.assertEqual(set(), items[1].tags)

    def test_set_liked_raises_if_not_found(self):
        items = Items([Item(self.api, self.category, 1), Item(self.api, self.category, 2)])
        with self.assertRaises(ValueError):
            items.set_liked(3)

    def test_remove_duplicates(self):
        raw_items = [Item(self.api, self.category, 1), Item(self.api, self.category, 1)]
        items = Items(raw_items)
        items.remove_duplicates()
        self.assertEqual(raw_items[0], items[0])

    def test_remove_duplicates_without_duplicates(self):
        raw_items = [Item(self.api, self.category, 1), Item(self.api, self.category, 2)]
        items = Items(raw_items)
        items.remove_duplicates()
        self.assertCountEqual(raw_items, items)

    def test_get_valid_tags(self):
        raw_items = [
            Item(self.api, self.category, 1),
            Item(self.api, self.category, 2),
            Item(self.api, self.category, 3)
        ]
        items = Items(raw_items)
        for i in raw_items:
            items.set_liked(i.id)

        tags = items.get_valid_tags(2)
        print(tags)
        self.fail('To do')

