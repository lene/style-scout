from acquisition.items import Items
from category import Category

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

from os.path import dirname, isfile, join
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.test_base import TestBase
from acquisition.ebay_downloader_io import EbayDownloaderIO


class EbayDownloaderIOTest(TestBase):

    ITEMS_FILE = 'items.pickle'
    WEIGHTS_FILE_BASE = 'weights'
    IMAGE_SIZE = 16

    def test_init_with_all_none(self) -> None:
        EbayDownloaderIO(self.DOWNLOAD_ROOT)

    def test_init_with_items_file(self) -> None:
        io = EbayDownloaderIO(self.DOWNLOAD_ROOT, items_file=self.ITEMS_FILE)
        self.assertEqual(io.items_file, join(self.DOWNLOAD_ROOT, self.ITEMS_FILE))

    def test_init_with_items_file_and_image_size(self) -> None:
        io = EbayDownloaderIO(self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, items_file=self.ITEMS_FILE)
        self.assertEqual(io.items_file, join(self.DOWNLOAD_ROOT, self.ITEMS_FILE))

    def test_init_with_weights_file_also_needs_image_size(self) -> None:
        with self.assertRaises(AssertionError):
            EbayDownloaderIO(self.DOWNLOAD_ROOT, weights_file=self.WEIGHTS_FILE_BASE)

    def test_init_with_weights_file_and_image_size(self) -> None:
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, weights_file=self.WEIGHTS_FILE_BASE
        )
        self.assertEqual(io.weights_file_base, join(self.DOWNLOAD_ROOT, self.WEIGHTS_FILE_BASE))
        self.assertEqual(
            join(self.DOWNLOAD_ROOT, '{}_{}.hdf5'.format(self.WEIGHTS_FILE_BASE, self.IMAGE_SIZE)),
            io.weights_file()
        )

    def test_init_with_additional_info(self) -> None:
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, weights_file=self.WEIGHTS_FILE_BASE
        )
        self.assertEqual(
            join(self.DOWNLOAD_ROOT, '{}_full_10_{}.hdf5'.format(self.WEIGHTS_FILE_BASE, self.IMAGE_SIZE)),
            io.weights_file('full', 10)
        )

    def test_thousands_are_replaced_with_k(self) -> None:
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, weights_file=self.WEIGHTS_FILE_BASE
        )
        self.assertEqual(
            join(self.DOWNLOAD_ROOT, '{}_full_1k_{}.hdf5'.format(self.WEIGHTS_FILE_BASE, self.IMAGE_SIZE)),
            io.weights_file('full', 1000)
        )
        self.assertEqual(
            join(self.DOWNLOAD_ROOT, '{}_full_1001_{}.hdf5'.format(self.WEIGHTS_FILE_BASE, self.IMAGE_SIZE)),
            io.weights_file('full', 1001)
        )

    def test_existing_weight_file_name_is_left_intact(self) -> None:
        Path(self.DOWNLOAD_ROOT, 'test.hdf5').touch(exist_ok=True)
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, weights_file='test.hdf5'
        )
        self.assertEqual(join(self.DOWNLOAD_ROOT, 'test.hdf5'), io.weights_file('full', 1000))

    def XXXtest_load_items_from_existing_file(self) -> None:
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, items_file=join(dirname(__file__), 'data', 'items.pickle')
        )
        items = io.load_items()
        self.assertEqual(2, len(items))
        self.assertEqual(1, num_liked_items(items))

    def test_load_items_without_file(self) -> None:
        io = EbayDownloaderIO(self.DOWNLOAD_ROOT)
        items = io.load_items()
        self.assertEqual(0, len(items))

    def test_save_items(self) -> None:
        items_file = join(self.DOWNLOAD_ROOT, 'test_items.pickle')
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE,
            items_file=items_file
        )
        items = self.generate_items(10)
        io.save_items(items, protocol=1)
        self.assertTrue(isfile(items_file))

    def test_save_items_creates_backup(self) -> None:
        items_file = join(self.DOWNLOAD_ROOT, 'test_items.pickle')
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE,
            items_file=items_file
        )
        items = self.generate_items(10)
        io.save_items(items, protocol=1)
        io.save_items(items, protocol=1)
        self.assertTrue(isfile(items_file + '.bak'))

    def XXXtest_import_likes(self) -> None:
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, items_file=join(dirname(__file__), 'data', 'items.pickle'),
            likes_file=join(dirname(__file__), 'data', 'likes.json')
        )
        items = io.load_items()
        self.assertEqual(2, len(items))
        self.assertEqual(1, num_liked_items(items))
        build_category_structure()
        io.import_likes(self.api, items)
        self.assertEqual(2, num_liked_items(items))

    def test_explicitly_specified_folder_overrides_base_dir(self) -> None:
        with TemporaryDirectory() as tempdir:
            weights_file = join(tempdir, 'test.hdf5')
            items_file = join(tempdir, 'images.pickle')
            io = EbayDownloaderIO(
                self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE,
                weights_file=weights_file, items_file=items_file
            )
            self.assertNotIn(self.DOWNLOAD_ROOT, io.weights_file('full', 1000))
            self.assertIn(tempdir, io.weights_file('full', 1000))
            self.assertNotIn(self.DOWNLOAD_ROOT, io.items_file)
            self.assertIn(tempdir, io.items_file)


def num_liked_items(items: Items) -> int:
    return len([i for i in items if i.is_liked])


def build_category_structure() -> None:
    Category(
        {
            'CategoryID': '62107',
            'CategoryName': 'Sandals',
            'LeafCategory': 'true'
        }
    )
