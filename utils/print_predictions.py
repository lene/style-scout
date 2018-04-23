from argparse import ArgumentParser, Namespace
from pickle import load
from pprint import pprint
from random import randrange

import numpy
from keras import Model

from acquisition.ebay_downloader_io import EbayDownloaderIO
from acquisition.items import Items
from data_sets.ebay_data_generator import EbayDataGenerator
from train import TrainingRunner


def print_predictions(image_data: EbayDataGenerator, model: Model) -> None:
    images, labels = next(image_data.test_generator())
    i = randrange(len(images))
    print_prediction(images[i:i + 1], labels[i], image_data, model)


def print_prediction(
        images: numpy.ndarray, label: str, image_data: EbayDataGenerator, model: Model
) -> None:
    print('actual values:')
    # pprint(image_data.labels_sorted_by_probability(label))
    image_data.show_image(images)
    print('predictions:')
    pprint(
        [
            (label, prob)
            for label, prob in image_data.labels_sorted_by_probability(
                model.predict(
                    images.reshape(1, *image_data.size, image_data.DEPTH), batch_size=1, verbose=1
                )[0]
            ).items()
            if prob > 0.01
        ]
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Train neural networks recognizing style from liked eBay items"
    )
    parser.add_argument('--items', required=True, type=str, help='Path to items file')
    parser.add_argument('--weights', required=True, type=str, help='Path to weights file')
    parser.add_argument(
        '--type', required=True, type=str, help='Type of neural network used',
        choices=list(TrainingRunner.NETWORK_TYPES.keys())
    )
    parser.add_argument(
        '--layers', required=True, type=int, nargs='+',
        help='Additional fully connected layers before the output layer'
    )
    parser.add_argument(
        '--size', required=True, type=int, help='Image size used in training'
    )
    return parser.parse_args()


def load_items(io: EbayDownloaderIO) -> Items:
    items = io.load_items()
    for item in items:
        if '<3' not in item.tags:
            item.tags.add(':-(')
    return items


def load_model(args: Namespace) -> Model:
    model = TrainingRunner.decode_network_name(args.type)(
        input_shape=(args.size, args.size, 3), classes=2, connected_layers=args.layers
    )
    model.compile(metrics=['accuracy'], optimizer='sgd', loss='mean_squared_error')
    model.load_weights(args.weights)
    return model


if __name__ == '__main__':
    args = parse_args()
    with open(args.items, 'rb') as file:
        items = load(file)

    valid_tags = {'<3': 0, ':-(': 0}
    model = load_model(args)
    generator = EbayDataGenerator(items, valid_tags, (args.size, args.size))

    print_predictions(generator, model)
