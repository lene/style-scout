from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Tuple, List

from os import makedirs, remove
from os.path import isfile
from subprocess import run, PIPE

makedirs('log', exist_ok=True)

BATCH_SIZE = 1
EPOCHS = (1, 2, 3, 5, 10)
ALGOS = (
    'vgg16', 'vgg19',
    'resnet50', 'inception', 'inception_resnet',
    # 'xception', 'densenet121', 'densenet169', 'densenet201', 'nasnet'
)
OPTIMIZERS = (
    'sgd', 'rmsprop', 'adagrad',
    # 'adadelta', 'adam', 'adamax', 'nadam'
)
SIZES = (200, 300, 400)
LAYERS = (
    (800,),
    (1024, 256),
    (1024, 1024, 512, 256)
)

ITEM_FILE = 'data/items_liked_unliked_equal.pickle'
OUTPUT_FILE = 'perf.md'


def read_tuple(
        cast: Callable[[str], Any], delimiter: str=','
) -> Callable[[str], Tuple[Any, ...]]:
    def do_read(args: str) -> Tuple[Any, ...]:
        parts = args.split(delimiter)
        return tuple([cast(p) for p in parts])
    return do_read


def tuple_string(args: Tuple) -> str:
    return str(args).replace(' ', '').replace('(', '').replace(')', '').replace("'", '')


def layers_string(layers: Tuple[Tuple[int, ...], ...]) -> str:
    return str(layers).\
        replace(' ', '').\
        replace(',)', ')').\
        replace('),(', '/').\
        replace('(', '').replace(')', '')


def parse_command_line() -> Namespace:
    parser = ArgumentParser(
        description="Record runtime and accuracies for several algorithms and configurations"
    )
    parser.add_argument('--item-file', type=str, default=ITEM_FILE)
    parser.add_argument('--output-file', type=str, default=OUTPUT_FILE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--likes-only', action='store_true')
    parser.add_argument(
        '--epochs', type=read_tuple(int), default=EPOCHS,
        help=f"List of number of evaluated epochs (default: {tuple_string(EPOCHS)})"
    )
    parser.add_argument(
        '--image-sizes', type=read_tuple(int), default=SIZES,
        help=f"List of evaluated image sizes (default: {tuple_string(SIZES)})"
    )
    parser.add_argument(
        '--algorithms', type=read_tuple(str), default=ALGOS,
        help=f"List of evaluated algorithms (default: {tuple_string(ALGOS)})"
    )
    parser.add_argument(
        '--optimizers', type=read_tuple(str), default=OPTIMIZERS,
        help=f"List of evaluated optimizers (default: {tuple_string(OPTIMIZERS)})"
    )
    parser.add_argument(
        '--layers', type=read_tuple(read_tuple(int), delimiter='/'), default=LAYERS,
        help=f"List of evaluated fully connected layers (default: {layers_string(LAYERS)})"
    )
    return parser.parse_args()


def time_string(time_line):
    string = time_line[0:time_line.find('elapsed') + len('elapsed')]
    user = float(string[0:string.find('user')])
    sys = float(string[string.find('user') + len('user '):string.find('system')])
    wall = string[string.find('system') + len('system '):string.find('elapsed')]
    return f'{wall} wall, {user + sys:7.2f}s CPU'


def training_run(
        epochs: int, weights_file: str, size: int, layers: Tuple[int, ...], algo: str,
        optimizer: str, item_file: str, batch_size: int, likes_only: bool
) -> Tuple[float, str]:
    command = [
        'nice', 'time', 'python', 'train.py',
        '--type', algo,
        '--optimizer', optimizer,
        '--num-epochs', str(epochs),
        '--layer', *[str(l) for l in layers],
        '--image-size', str(size),
        '--weights-file', weights_file,
        '--item-file', item_file,
        '--demo', '0',
        '--batch-size', str(batch_size),
        '--test',
        '--random-seed', '0',
        '-v'
    ]
    if likes_only:
        command.append('--likes-only')

    result = run(
        command,
        stdout=PIPE, stderr=PIPE
    )
    stdout = result.stdout.decode('utf-8').split('\n')
    stderr = result.stderr.decode('utf-8').split('\n')
    try:
        accuracy_line = stdout[-4]
        test_set_accuracy = float(
            accuracy_line[
                accuracy_line.find('test set accuracy: ') + len('test set accuracy: '):-1
            ]
        )
        return test_set_accuracy, time_string(stderr[-3])
    except (IndexError, ValueError):
        return 0, '---'


def current_num_epochs(epochs: Tuple[int, ...], current_epoch: int) -> int:
    index = epochs.index(current_epoch)
    return epochs[0] if index == 0 else epochs[index] - epochs[index - 1]


def evaluate(
        size: int, layers: Tuple[int, ...], algo: str, optimizer: str, epochs: Tuple[int, ...],
        item_file: str, batch_size: int, likes_only: bool
) -> str:
    weights_file = f'/tmp/{algo}_{size}_{layers}.hdf5'
    if isfile(weights_file):
        remove(weights_file)

    test_set_accuracy = []
    total_run_time = ""
    for e in epochs:
        accuracy, run_time = training_run(
            current_num_epochs(epochs, e), weights_file, size, layers, algo, optimizer,
            item_file, batch_size, likes_only
        )
        test_set_accuracy.append(accuracy)
        total_run_time = run_time if not total_run_time else total_run_time

    return f'|{algo:18}|{optimizer:8}|{size}|{layers}|{total_run_time}|' + \
           '|'.join([f'{acc:.4f}' for acc in test_set_accuracy]) + '|\n'


def main(args: Namespace) -> None:
    print(args)
    with open(args.output_file, 'w') as f:
        f.writelines(header(args.epochs))

    for size in args.image_sizes:
        for layers in args.layers:
            for algo in args.algorithms:
                for optimizer in args.optimizers:
                    with open(args.output_file, 'a') as f:
                        f.write(
                            evaluate(
                                size, layers, algo, optimizer, args.epochs,
                                args.item_file, args.batch_size, args.likes_only
                            )
                        )


def header(epochs: Tuple[int, ...]) -> List[str]:
    return [
        '|Algorithm|Optimizer|Size|Extra layers|Run time/epoch|' +
        '|'.join([f"Test set acc. ({e} epochs)" for e in epochs]) +
        '|\n',
        '|:--------|--------:|---:|-----------:|-------------:' +
        '|:-----------------------' * len(epochs) +
        '|\n'
    ]


if __name__ == '__main__':
    main(parse_command_line())
