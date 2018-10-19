from argparse import ArgumentParser, Namespace
from pprint import pprint
from typing import Callable, Tuple, Dict

from PIL import Image
import numpy
from keras import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

from acquisition.ebay_downloader_io import EbayDownloaderIO
from acquisition.items import Items
from data_sets.ebay_data_generator import EbayDataGenerator
from utils.with_verbose import WithVerbose
from network_types import (
    inception, xception, vgg16, vgg19, resnet50, inception_resnet_v2,
    densenet121, densenet169, densenet201, nasnet
)
from data_sets.contains_images import add_border

MIN_TAG_NUM = 10
SAVE_FOLDER = 'data'
DEFAULT_IMAGE_SIZE = 139
DEFAULT_TEST_SET_SHARE = 0.2


def parse_command_line() -> Namespace:
    parser = ArgumentParser(
        description="Train neural networks recognizing style from liked eBay items"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help="Print info about extracted tags"
    )
    parser.add_argument(
        '--save-folder', default=SAVE_FOLDER,
        help='Folder under which to store items, images and weights'
    )
    parser.add_argument(
        '--item-file', '-i', default=None, help="Pickle file from which to load downloaded items"
    )
    parser.add_argument(
        '--min-valid-tag', default=MIN_TAG_NUM, type=int,
        help="Minimum number of times a tag has to occur to be considered valid"
    )
    parser.add_argument(
        '--images-file', default=None,
        help='Pickle file from which to load precomputed image data set'
    )
    parser.add_argument(
        '--weights-file', '-w', default=None,
        help='HDF5 file from which to load precomputed set of weights'
    )
    parser.add_argument(
        '--num-epochs', '-n', type=int, default=1, help='How many times to iterate'
    )
    parser.add_argument(
        '--image-size', '-s', type=int, default=DEFAULT_IMAGE_SIZE,
        help='Size (both width and height) to which images are resized'
    )
    parser.add_argument(
        '--test-set-share', type=float, default=DEFAULT_TEST_SET_SHARE,
        help='Share of the data used as test set'
    )
    parser.add_argument(
        '--random-seed', type=int, default=None, help='Random seed used in train-test split'
    )
    parser.add_argument(
        '--demo', type=int, default=0, help='Number of images to try to predict as demo'
    )
    parser.add_argument(
        '--test', '-t', action='store_true', help="Run evaluation on test data set"
    )
    parser.add_argument('--likes-only', '-l', action='store_true', help="Only train against likes")
    parser.add_argument('--category', help='Only train against items of this category')
    parser.add_argument(
        '--batch-size', type=int, default=32, help='Batch size used in fitting the model'
    )
    parser.add_argument(
        '--optimizer', default='adam', help='Optimizer used to fit the model',
        choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']
    )
    parser.add_argument(
        '--type', default='inception', help='Type of neural network used',
        choices=list(TrainingRunner.NETWORK_TYPES.keys())
    )
    parser.add_argument(
        '--layers', default=[1024], type=int, nargs='+',
        help='Additional fully connected layers before the output layer'
    )

    return parser.parse_args()


class TrainingRunner(WithVerbose):

    NETWORK_TYPES = {
        'inception': inception,
        'xception': xception,
        'vgg16': vgg16,
        'vgg19': vgg19,
        'resnet50': resnet50,
        'inception_resnet': inception_resnet_v2,
        'densenet121': densenet121,
        'densenet169': densenet169,
        'densenet201': densenet201,
        'nasnet': nasnet,
    }

    def __init__(self, args: Namespace) -> None:
        WithVerbose.__init__(self, args.verbose)
        self.image_size = args.image_size
        self.min_valid_tag = args.min_valid_tag
        self.likes_only = args.likes_only
        self.category = args.category
        self.batch_size = args.batch_size
        self.demo = args.demo
        self.num_epochs = args.num_epochs
        self.test = args.test
        self.io = EbayDownloaderIO(
            args.save_folder, args.image_size, args.item_file, args.weights_file,
            verbose=self.verbose
        )
        self.loss_function = 'mean_squared_error'
        self.optimizer = args.optimizer
        self.neural_network_type = self.decode_network_name(args.type)
        self.fully_connected_layers = args.layers
        self.log_dir = './logs'  # TODO: CLI arg
        self.image_data = self._get_image_data(args.test_set_share, args. random_seed)
        self.model = self.setup_model()

    def run(self) -> None:
        self.run_training()
        self.run_test()
        self.run_demo()

    def run_training(self):
        if self.num_epochs:
            self.model.fit_generator(
                self.image_data.train_generator(),
                steps_per_epoch=self.image_data.train_length(), epochs=self.num_epochs,
                callbacks=[
                    # save weights after every iteration
                    ModelCheckpoint(self.io.weights_file_base + '.{epoch:02d}.hdf5', verbose=self.verbose),
                    # if loss does not change for 2 iterations, change learning rate
                    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=self.verbose),
                    # if loss does not change for 4 iterations, finish training
                    EarlyStopping(monitor='loss', min_delta=0, patience=4, verbose=self.verbose),
                    # write log for visualization in TensorBoard
                    TensorBoard(
                        log_dir=self.log_dir, batch_size=self.batch_size, histogram_freq=0,
                        write_graph=True, write_grads=False, write_images=True
                    )
                ]
            )
            self.io.save_weights(self.model, self._fit_type(), self._num_items)

    def run_test(self):
        if self.test:
            loss_and_metrics = self.model.evaluate_generator(
                self.image_data.test_generator(), steps=self.image_data.test_length(), max_queue_size=2
            )
            print()
            print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])

    def run_demo(self):
        for item in [i for i in self._prepare_items()[0] if '<3' in i.tags][:self.demo]:
            images = numpy.asarray([
                self.image_data.downscale(Image.open(file).convert('RGB'), method=add_border)
                for file in item.picture_files
            ])
            for image in images:
                self.image_data.show_image(image)
            predictions = self.model.predict(images, batch_size=len(images), verbose=1)
            for prediction in predictions:
                pprint(
                    [
                        (label, prob)
                        for label, prob in self.image_data.labels_sorted_by_probability(prediction).items()
                        if prob > 0.01
                    ]
                )

    def _get_image_data(self, test_set_share: float, random_seed: int) -> EbayDataGenerator:
        items, valid_tags = self._prepare_items()
        return EbayDataGenerator(
            items, valid_tags, (self.image_size, self.image_size),
            batch_size=self.batch_size, random_seed=random_seed, test_share=test_set_share,
            verbose=self.verbose
        )

    def setup_model(self) -> Model:
        model = self.neural_network_type(
            input_shape=(*self.image_data.size, self.image_data.DEPTH),
            classes=self.image_data.num_classes,
            connected_layers=self.fully_connected_layers
        )
        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        num_layers = len(model.layers)
        self._print_status(f'Model compiled - {self.neural_network_type.__name__}, {num_layers} layers')
        self.io.load_weights(model, self._fit_type(), self._num_items)
        return model

    def _prepare_items(self) -> Tuple[Items, Dict[str, int]]:
        items = self.io.load_items()
        self._num_items = len(items)
        if self.likes_only:
            for item in items:
                if '<3' not in item.tags:
                    item.tags.add(':-(')
            valid_tags = {'<3': 0, ':-(': 0}
        else:
            valid_tags = items.get_valid_tags(self.min_valid_tag)
        if self.category:
            items = items.filter(category=self.category)
            if len(items) == 0:
                raise ValueError('No items of category {}: {}'.format(self.category, items.categories()))
        items.update_tags(valid_tags)
        category_string = '{}: '.format(self.category) if self.category else ''
        self._print_status(
            '{}{} items, {} liked'.format(
                category_string,
                len(items),
                len([i for i in items if '<3' in i.tags])
            )
        )
        return items, valid_tags

    def _fit_type(self) -> str:
        type = 'likes' if self.likes_only else 'full'
        return self.category.lower() + '_' + type if self.category else type

    @staticmethod
    def decode_network_name(network_type: str) -> Callable:
        try:
            return TrainingRunner.NETWORK_TYPES[network_type]
        except KeyError:
            raise ValueError('Invalid Neural Network name "{}"'.format(network_type))


if __name__ == '__main__':
    runner = TrainingRunner(args=parse_command_line())
    runner.run()
