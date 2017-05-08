from argparse import ArgumentParser
from pprint import pprint
from random import randrange
from PIL import Image
import numpy

from acquisition.ebay_downloader_io import EbayDownloaderIO
from data_sets.ebay_data_generator import EbayDataGenerator
from utils.with_verbose import WithVerbose
from variable_inception import variable_inception
from data_sets.contains_images import add_border

MIN_TAG_NUM = 10
SAVE_FOLDER = 'data'
DEFAULT_SIZE = 139


def parse_command_line():
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
        '--item-file', default=None, help="Pickle file from which to load downloaded items"
    )
    parser.add_argument(
        '--min-valid-tag', default=MIN_TAG_NUM, type=int,
        help="Minimum number of times a tag has to occur to be considered valid"
    )
    parser.add_argument(
        '--images-file', default=None, help='Pickle file from which to load precomputed image data set'
    )
    parser.add_argument(
        '--weights-file', '-w', default=None, help='HDF5 file from which to load precomputed set of weights'
    )
    parser.add_argument(
        '--num-epochs', '-n', type=int, default=1, help='How many times to iterate'
    )
    parser.add_argument(
        '--image-size', '-s', type=int, default=DEFAULT_SIZE,
        help='Size (both width and height) to which images are resized'
    )
    parser.add_argument(
        '--demo', type=int, default=0, help='Number of images to try to predict as demo'
    )
    parser.add_argument(
        '--test', '-t', action='store_true', help="Run evaluation on test data set"
    )
    parser.add_argument('--likes-only', action='store_true', help="Only train against likes")
    parser.add_argument('--category', help='Only train against items of this category')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size used in fitting the model')
    parser.add_argument('--optimizer', default='adam', help='Optimizer used to fit the model')

    return parser.parse_args()


def print_predictions(image_data, model):
    images, labels = next(image_data.test_generator())
    i = randrange(len(images))
    print_prediction(images[i:i + 1], labels[i], image_data, model)


def print_prediction(images, label, image_data, model):
    print('actual values:')
    pprint(image_data.labels_sorted_by_probability(label))
    image_data.show_image(images)
    print('predictions:')
    pprint(
        [
            (label, prob)
            for label, prob in image_data.labels_sorted_by_probability(
                model.predict(images.reshape(1, *image_data.size, image_data.DEPTH), batch_size=1, verbose=1)[0]
            ).items()
            if prob > 0.01
        ]
    )


class TrainingRunner(WithVerbose):

    def __init__(self, args):
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
            args.save_folder, args.image_size, args.item_file, args.images_file, args.weights_file,
            verbose=self.verbose
        )
        self.loss_function = 'mean_squared_error'
        self.optimizer = 'sgd'

    def run(self):

        image_data = self.get_image_data()

        model = self.setup_model(image_data)

        if self.num_epochs:
            model.fit_generator(
                image_data.train_generator(),
                steps_per_epoch=image_data.train_length(), epochs=self.num_epochs
            )
            self.io.save_weights(model, self._fit_type(), self._num_items)

        if self.test:
            loss_and_metrics = model.evaluate_generator(
                image_data.test_generator(), steps=image_data.test_length(), max_q_size=2
            )
            print()
            print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])

        for item in [i for i in self._prepare_items()[0] if '<3' in i.tags][:self.demo]:
            images = numpy.asarray([
                image_data.downscale(Image.open(file).convert('RGB'), method=add_border)
                for file in item.picture_files
            ])
            for image in images:
                image_data.show_image(image)
            predictions = model.predict(images, batch_size=len(images), verbose=1)
            for prediction in predictions:
                pprint(
                    [
                        (label, prob)
                        for label, prob in image_data.labels_sorted_by_probability(prediction).items()
                        if prob > 0.01
                    ]
                )

    def get_image_data(self):
        items, valid_tags = self._prepare_items()
        image_data = EbayDataGenerator(
            items, valid_tags, (self.image_size, self.image_size),
            batch_size=self.batch_size, verbose=self.verbose
        )

        return image_data

    def _prepare_items(self):
        items = self.io.load_items()
        self._num_items = len(items)
        if self.likes_only:
            for item in items:
                if '<3' not in item.tags:
                    item.tags.add('</3')
            valid_tags = {'<3': None, '</3': None}
        else:
            valid_tags = items.get_valid_tags(self.min_valid_tag)
        if self.category:
            items = items.filter(category=self.category)
            if len(items) == 0:
                raise ValueError('No items of category ' + self.category)
        items.update_tags(valid_tags)
        self._print_status('{} items, {} liked'.format(len(items), len([i for i in items if '<3' in i.tags])))
        return items, valid_tags

    def setup_model(self, image_data):
        model = variable_inception(
            input_shape=(*image_data.size, image_data.DEPTH), classes=image_data.num_classes
        )
        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        self._print_status('Model compiled')
        self.io.load_weights(model, self._fit_type(), self._num_items)
        return model

    def _fit_type(self):
        type = 'likes' if self.likes_only else 'full'
        return self.category.lower() + '_' + type if self.category else type


if __name__ == '__main__':
    runner = TrainingRunner(args=parse_command_line())
    runner.run()
