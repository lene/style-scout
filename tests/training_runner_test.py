__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

import unittest
from collections import namedtuple

from tests.test_base import TestBase
from train import TrainingRunner

Args = namedtuple(
    'Args', [
        'verbose', 'image_size', 'min_valid_tag', 'likes_only', 'category', 'batch_size', 'demo',
        'num_epochs', 'test', 'save_folder', 'item_file', 'images_file', 'weights_file', 'type'
    ]
)


class TrainingRunnerTest(TestBase):

    def test_init_existing_network_types(self):
        for type in ('inception', 'xception', 'vgg16', 'vgg19', 'resnet50'):
            TrainingRunner(
                Args(
                    verbose=False, image_size=0, min_valid_tag=0, likes_only=False, category='', batch_size=1,
                    demo=False, num_epochs=0, test=False, save_folder=self.DOWNLOAD_ROOT,
                    item_file='', images_file='', weights_file='',
                    type=type
                )
            )

    def test_init_nonexisting_network_type(self):
        with self.assertRaises(ValueError):
            TrainingRunner(
                Args(
                    verbose=False, image_size=0, min_valid_tag=0, likes_only=False, category='', batch_size=1,
                    demo=False, num_epochs=0, test=False, save_folder=self.DOWNLOAD_ROOT,
                    item_file='', images_file='', weights_file='',
                    type='BWAHAHAH FAIL!'
                )
            )
