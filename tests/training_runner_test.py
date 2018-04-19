__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

from recordclass import recordclass
from tests.test_base import TestBase
from train import TrainingRunner


class Args(
    recordclass(  # type: ignore
        'Args', [
            'verbose', 'image_size', 'min_valid_tag', 'likes_only', 'category', 'batch_size', 'demo',
            'num_epochs', 'test', 'save_folder', 'item_file', 'weights_file', 'type',
            'optimizer', 'layers'
        ]
    )
):
    @staticmethod
    def default_args() -> 'Args':
        return Args(
            verbose=False, image_size=139, min_valid_tag=0, likes_only=False, category='', batch_size=1,
            demo=False, num_epochs=0, test=False, save_folder=TestBase.DOWNLOAD_ROOT,
            item_file='', weights_file='',
            type='inception', optimizer='adam', layers=1
        )


class TrainingRunnerTest(TestBase):

    def test_init_existing_network_types(self) -> None:
        for nn_type in ('inception', 'xception', 'vgg16', 'vgg19', 'resnet50'):
            args = Args.default_args()
            args.type = nn_type
            TrainingRunner(args)

    def test_init_nonexisting_network_type(self) -> None:
        with self.assertRaises(ValueError):
            args = Args.default_args()
            args.type = 'BWAHAHAH FAIL!'
            TrainingRunner(args)
