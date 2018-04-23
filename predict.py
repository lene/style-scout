from argparse import ArgumentParser, Namespace
from typing import List

import numpy
from keras import Model
from PIL import Image
from pprint import pprint

from acquisition.ebay_downloader_io import EbayDownloaderIO
from data_sets.contains_images import ContainsImages
from utils.with_verbose import WithVerbose
from train import TrainingRunner

SAVE_FOLDER = 'data'
DEFAULT_SIZE = 139


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
        '--weights-file', '-w', default=None, help='HDF5 file from which to load precomputed set of weights'
    )
    parser.add_argument(
        '--images-file', default=None, help='Pickle file from which to load precomputed image data set'
    )
    parser.add_argument(
        '--image-size', '-s', type=int, default=DEFAULT_SIZE,
        help='Size (both width and height) to which images are resized'
    )
    parser.add_argument(
        '--demo', type=int, default=0, help='Number of images to try to predict as demo'
    )
    parser.add_argument(
        '--predict-image', action='append',
        help='Image which is evaluated'
    )
    parser.add_argument(
        '--predict-item-url', action='append',
        help='URL of eBay item which is evaluated'
    )
    parser.add_argument(
        '--type', default='inception', help='Type of neural network used',
        choices=list(TrainingRunner.NETWORK_TYPES.keys())
    )

    return parser.parse_args()


def get_items(item_urls: List[str]) -> None:
    pass


class Predictor(WithVerbose, ContainsImages):

    def __init__(self, args: Namespace) -> None:
        self.io = EbayDownloaderIO(
            args.save_folder, args.image_size, weights_file=args.weights_file, verbose=args.verbose
        )
        self.verbose = args.verbose
        self.size = (args.image_size, args.image_size)
        self.demo = args.demo
        self.neural_network_type = TrainingRunner.decode_network_name(args.type)
        self.model = self.setup_model()

    def setup_model(self, loss_function: str='mean_squared_error', optimizer: str='sgd') -> Model:
        model = self.neural_network_type(
            input_shape=(*self.size, 3), classes=2  # image_data.num_classes
        )
        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

        self._print_status('Model compiled')
        self.io.load_weights(model)
        return model

    def predict(self, image_files: List[str]) -> None:
        images = numpy.asarray([self.downscale(Image.open(file).convert('RGB')) for file in image_files])
        for image in images:
            self.show_image(image)
        predictions = self.model.predict(images, batch_size=len(images), verbose=1)
        for prediction in predictions:
            # pprint(
            #     [
            #         (label, prob)
            #         for label, prob in image_data.labels_sorted_by_probability(prediction).items()
            #         if prob > 0.01
            #     ]
            # )
            pprint(prediction)


if __name__ == '__main__':
    args = parse_command_line()
    predictor = Predictor(args)
    predictor.predict(args.predict_image)
