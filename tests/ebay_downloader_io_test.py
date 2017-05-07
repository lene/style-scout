__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

from os.path import join
from pathlib import Path

from tests.test_base import TestBase
from acquisition.ebay_downloader_io import EbayDownloaderIO


class EbayDownloaderIOTest(TestBase):

    ITEMS_FILE = 'items.pickle'
    WEIGHTS_FILE_BASE = 'weights'
    IMAGE_SIZE = 16

    def test_init_with_all_none(self):
        EbayDownloaderIO(self.DOWNLOAD_ROOT)

    def test_init_with_items_file(self):
        io = EbayDownloaderIO(self.DOWNLOAD_ROOT, items_file=self.ITEMS_FILE)
        self.assertEqual(io.items_file, join(self.DOWNLOAD_ROOT, self.ITEMS_FILE))

    def test_init_with_items_file_and_image_size(self):
        io = EbayDownloaderIO(self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, items_file=self.ITEMS_FILE)
        self.assertEqual(io.items_file, join(self.DOWNLOAD_ROOT, self.ITEMS_FILE))

    def test_init_with_weights_file_also_needs_image_size(self):
        with self.assertRaises(AssertionError):
            EbayDownloaderIO(self.DOWNLOAD_ROOT, weights_file=self.WEIGHTS_FILE_BASE)

    def test_init_with_weights_file_and_image_size(self):
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, weights_file=self.WEIGHTS_FILE_BASE
        )
        self.assertEqual(io.weights_file_base, join(self.DOWNLOAD_ROOT, self.WEIGHTS_FILE_BASE))
        self.assertEqual(
            join(self.DOWNLOAD_ROOT, '{}_{}.hdf5'.format(self.WEIGHTS_FILE_BASE, self.IMAGE_SIZE)),
            io.weights_file()
        )

    def test_init_with_additional_info(self):
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, weights_file=self.WEIGHTS_FILE_BASE
        )
        self.assertEqual(
            join(self.DOWNLOAD_ROOT, '{}_full_10_{}.hdf5'.format(self.WEIGHTS_FILE_BASE, self.IMAGE_SIZE)),
            io.weights_file('full', 10)
        )

    def test_thousands_are_replaced_with_k(self):
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

    def test_existing_weight_file_name_is_left_intact(self):
        Path(self.DOWNLOAD_ROOT, 'test.hdf5').touch(exist_ok=True)
        io = EbayDownloaderIO(
            self.DOWNLOAD_ROOT, image_size=self.IMAGE_SIZE, weights_file='test.hdf5'
        )
        self.assertEqual(join(self.DOWNLOAD_ROOT, 'test.hdf5'), io.weights_file('full', 1000))

    def test_load_items_from_existing_file(self):
        self.skipTest('Test not yet implemented')

    def test_load_items_without_file(self):
        self.skipTest('Test not yet implemented')

    def test_save_items(self):
        self.skipTest('Test not yet implemented')

    def test_import_likes(self):
        self.skipTest('Test not yet implemented')

    def test_get_images_from_existing_file(self):
        self.skipTest('Test not yet implemented')

    def test_get_images_without_file(self):
        self.skipTest('Test not yet implemented')

    def test_load_weights_from_saved_weights_equal_original_weights(self):
        self.skipTest('Test not yet implemented')

    def test_images_filename_contains_num_items_and_image_size(self):
        self.skipTest('Functionality not yet implemented')

    def test_weights_filename_contains_num_items_image_size_and_epoch_number(self):
        self.skipTest('Functionality not yet implemented')

