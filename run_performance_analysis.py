from argparse import ArgumentParser, Namespace
from typing import Tuple

from os import makedirs, remove
from os.path import isfile
from subprocess import run, PIPE

makedirs('log', exist_ok=True)

BATCH_SIZE = 1
ALGOS = (
    'vgg16', 'vgg19', 'xception', 'resnet50', 'inception', 'inception_resnet',
    'densenet121', 'densenet169', 'densenet201', 'nasnet'
)
SIZES = (200, 300, 400)
ITEM_FILE = 'data/items_liked_unliked_equal.pickle'
LAYERS = (
    (1024,), (1024, 1024), (1024, 1024, 512, 256), (1024, 1024, 1024, 512, 256, 128, 64)
)  # type: Tuple[Tuple[int, ...], ...]
OUTPUT_FILE = 'perf.md'


def parse_command_line() -> Namespace:
    parser = ArgumentParser(
        description="Record runtime and accuracies for several algorithms and configurations"
    )
    parser.add_argument(
        '--item-file', type=str, default=ITEM_FILE
    )
    parser.add_argument(
        '--output-file', type=str, default=OUTPUT_FILE
    )
    parser.add_argument(
        '--batch-size', type=int, default=BATCH_SIZE
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
        item_file: str, batch_size: int
) -> Tuple[float, str]:
    result = run(
        [
            'nice', 'time', 'python', 'train.py',
            '-n', str(epochs), '-v', '--weights-file', weights_file,
            '--item-file', item_file, '--image-size', str(size),
            '--likes-only', '--demo', '0', '--batch-size', str(batch_size), '--optimizer', 'sgd',
            '--layer', *[str(l) for l in layers],
            '--type', algo, '--test'
        ],
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


def evaluate(size: int, layers: Tuple[int, ...], algo: str, args: Namespace):
    weights_file = f'/tmp/{algo}_{size}_{layers}.hdf5'
    if isfile(weights_file):
        remove(weights_file)
    test_set_accuracy_1, run_time_1 = training_run(
        1, weights_file, size, layers, algo, args.item_file, args.batch_size
    )
    test_set_accuracy_2, _ = training_run(
        1, weights_file, size, layers, algo, args.item_file, args.batch_size
    )
    test_set_accuracy_3, _ = training_run(
        1, weights_file, size, layers, algo, args.item_file, args.batch_size
    )
    test_set_accuracy_5, _ = training_run(
        2, weights_file, size, layers, algo, args.item_file, args.batch_size
    )
    test_set_accuracy_10, _ = training_run(
        5, weights_file, size, layers, algo, args.item_file, args.batch_size
    )
    # test_set_accuracy_3 = test_set_accuracy_5 = test_set_accuracy_10 = 0
    with open(args.output_file, 'a') as f:
        f.write(
            f'|{algo:18}|{size}|{layers}|{run_time_1}'
            f'|{test_set_accuracy_1:.4f}|{test_set_accuracy_2:.4f}'
            f'|{test_set_accuracy_3:.4f}|{test_set_accuracy_5:.4f}|{test_set_accuracy_10:.4f}|\n'
        )
    if isfile(weights_file):
        remove(weights_file)


args = parse_command_line()

with open(args.output_file, 'w') as f:
    f.writelines([
        '|Algorithm|Size|Extra layers|Run time/epoch|Test set acc. (1 epoch)'
        '|Test set acc. (2 epochs)|Test set acc. (3 epochs)|Test set acc. (5 epochs)|'
        'Test set acc. (10 epochs)|\n',
        '|:--------|---:|-----------:|-------------:|:----------------------'
        '|:-----------------------|:-----------------------|:-----------------------'
        '|:------------------------|\n'
    ])

for size in SIZES:
    for layers in LAYERS:
        for algo in ALGOS:
            evaluate(size, layers, algo, args)
