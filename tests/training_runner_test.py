__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

from recordclass import recordclass
from tests.test_base import TestBase
from train import TrainingRunner


class Args(
    recordclass(
        'Args', [
            'verbose', 'image_size', 'min_valid_tag', 'likes_only', 'category', 'batch_size', 'demo',
            'num_epochs', 'test', 'save_folder', 'item_file', 'weights_file', 'type',
            'optimizer', 'layers'
        ]
    )
):
    @staticmethod
    def default_args():
        return Args(
            verbose=False, image_size=139, min_valid_tag=0, likes_only=False, category='', batch_size=1,
            demo=False, num_epochs=0, test=False, save_folder=TestBase.DOWNLOAD_ROOT,
            item_file='', weights_file='',
            type='inception', optimizer='adam', layers=1
        )


class TrainingRunnerTest(TestBase):

    def test_init_existing_network_types(self):
        for nn_type in ('inception', 'xception', 'vgg16', 'vgg19', 'resnet50'):
            args = Args.default_args()
            args.type = nn_type
            TrainingRunner(args)

    def test_init_nonexisting_network_type(self):
        with self.assertRaises(ValueError):
            args = Args.default_args()
            args.type = 'BWAHAHAH FAIL!'
            TrainingRunner(args)

    def test_different_optimizers(self):
        self.skipTest("Test not yet implemented")

    def test_different_fully_connected_layers(self):
        self.skipTest("Test not yet implemented")
        for nn_type in ('inception', 'xception', 'vgg16', 'vgg19', 'resnet50'):
            args = Args.default_args()
            args.type = nn_type
            layers = TrainingRunner(args).instantiate_model(None).layers
            args.layers = [1024, 1024]
            # this instantiates a full session. better solution needed.
            self.assertEqual(len(TrainingRunner(args).instantiate_model(None).layers), len(layers) + 1)


def _create_empty_data_set(reshape=None):
    # TODO remove duplicate, move to TestBase
    from tests.images_labels_data_set_test import create_empty_image_data, create_empty_label_data
    from data_sets.images_labels_data_set import ImagesLabelsDataSet
    images = create_empty_image_data()
    labels = create_empty_label_data()
    return ImagesLabelsDataSet(images, labels, reshape)
