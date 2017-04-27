from argparse import ArgumentParser
from pprint import pprint
from random import randrange

from acquisition.ebay_downloader_io import EbayDownloaderIO
from data_sets.ebay_data_generator import EbayDataGenerator
from utils.with_verbose import WithVerbose
from variable_inception import variable_inception

MIN_TAG_NUM = 10
SAVE_FOLDER = 'data'
DEFAULT_SIZE = 139


def parse_command_line():
    parser = ArgumentParser(
        description="Download information about eBay items as training data for neural network recognizing style"
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
        '--num-epochs', '-n', type=int, default=1, help='How many times to iterate.'
    )
    parser.add_argument(
        '--image-size', '-s', type=int, default=DEFAULT_SIZE,
        help='Size (both width and height) to which images are resized.'
    )
    parser.add_argument(
        '--demo', type=int, default=5, help='Number of images to try to predict as demo.'
    )
    parser.add_argument(
        '--test', '-t', action='store_true', help="Run evaluation on test data set"
    )
    parser.add_argument(
        '--use-single-batch', action='store_true', help="Read all data in advance"
    )
    parser.add_argument(
        '--batch-size', type=int, default=32, help='Batch size used in fitting the model.'
    )
    parser.add_argument(
        '--cache-dir', default='/tmp', help="Directory to contain the cached image data"
    )

    return parser.parse_args()


def print_prediction(data_set, image_data, model):
    i = randrange(len(data_set.input))
    image = data_set.input[i]
    label = data_set.labels[i]
    print('actual values:')
    pprint(image_data.labels_sorted_by_probability(label))
    image_data.show_image(image)
    print('predictions:')
    pprint(
        [(label, prob) for label, prob in image_data.labels_sorted_by_probability(
            model.predict(data_set.input[i:i + 1], batch_size=1, verbose=1)[0]
        ).items() if prob > 0.01]
    )


class TrainingRunner(WithVerbose):

    def __init__(self, args):
        WithVerbose.__init__(self, args.verbose)
        self.image_size = args.image_size
        self.min_valid_tag = args.min_valid_tag
        self.use_single_batch = args.use_single_batch
        self.batch_size = args.batch_size
        self.cache_dir = args.cache_dir
        self.demo = args.demo
        self.num_epochs = args.num_epochs
        self.test = args.test
        self.io = EbayDownloaderIO(
            args.save_folder, args.image_size, args.item_file, args.images_file, args.weights_file,
            verbose=self.verbose
        )

    def run(self):

        image_data = self.get_image_data()

        model = self.setup_model(image_data)

        if self.num_epochs:
            if self.use_single_batch:
                model.fit(
                    image_data.train.input, image_data.train.labels,
                    epochs=self.num_epochs, batch_size=self.batch_size
                )
            else:
                model.fit_generator(
                    image_data.train_generator(), steps_per_epoch=len(image_data), epochs=self.num_epochs
                )
            self.io.save_weights(model)

    def get_image_data(self):
        items = self.io.load_items()
        valid_tags = items.get_valid_tags(self.min_valid_tag)
        items.update_tags(valid_tags)
        if self.use_single_batch:
            image_data = self.io.get_images(items, valid_tags, self.image_size, test_share=0)
        else:
            image_data = EbayDataGenerator(
                items, valid_tags, (self.image_size, self.image_size),
                batch_size=self.batch_size, cache_dir=self.cache_dir, verbose=self.verbose
            )
        self._print_status('Image data loaded')
        self._see_if_labels_are_correct(image_data)

        return image_data

    def setup_model(self, image_data):
        model = variable_inception(input_shape=(*image_data.size, image_data.DEPTH),
                                   classes=image_data.num_classes)
        model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
        self._print_status('model compiled')
        self.io.load_weights(model)
        self._print_status('weights loaded')
        return model

    def run_test(self):
        if self.test:
            test = image_data.test.input
            loss_and_metrics = model.evaluate(test, image_data.test.labels)
            print()
            print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])

    def run_predictions(self):
        for _ in range(self.demo):
            print_prediction(image_data.train, image_data, model)
            print_prediction(image_data.test, image_data, model)

    def _see_if_labels_are_correct(self, image_data):
        for _ in range(self.demo):
            i = randrange(len(image_data.train.input))
            image = image_data.train.input[i]
            label = image_data.train.labels[i]
            print('Labels before fitting:')
            pprint(image_data.labels_sorted_by_probability(label))
            image_data.show_image(image)


if __name__ == '__main__':
    TrainingRunner(args=parse_command_line()).run()
