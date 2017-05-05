from appJar import gui
from pickle import load, dump
from argparse import ArgumentParser, RawTextHelpFormatter
from os.path import join, isfile
from PIL import Image
from pprint import pprint

from data_sets.contains_images import add_border

STATE_FILE = 'data/like_items.state'

SAVE_FOLDER = 'data'


def parse_command_line():
    parser = ArgumentParser(
        description="""Simple GUI to like/dislike eBay items.
Use 'l' key to like an item, 'd' to dislike it, 'n' to skip it (leaving the like status
untouched), 'b' to view the previous item and 's' or 'q' to stop the program.""",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--save-folder', default=SAVE_FOLDER, help='Folder under which items are stored')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument(
        '--item-file', default=None, help="Pickle file from which to load downloaded items"
    )
    parser.add_argument(
        '--liked-only', action='store_true', help="Only display liked items"
    )
    parser.add_argument('--size', type=int, default=400, help="Size of the displayed images")

    return parser.parse_args()


class LikeItemsUI:

    def __init__(self, app, items, start, size, liked_only=False):
        self.item_no = start
        self.items = [i for i in items if '<3' in i.tags] if liked_only else items
        self.app = app
        self.image_size = (size, size)
        self.liked = len([i for i in self.items[:start] if '<3' in i.tags])

        self._setup_ui()

        self._setup_key_bindings()

        self.update_content()

    def _setup_key_bindings(self):
        self.app.bindKey('l', self.like)
        self.app.bindKey('d', self.dislike)
        self.app.bindKey('s', self.stop)
        self.app.bindKey('q', self.stop)
        self.app.bindKey('b', self.back)
        self.app.bindKey('n', self.next)

    def _setup_ui(self):
        self.app.addLabel('title', 'Like or dislike eBay items here')
        self.app.startLabelFrame("Pictures", 0, 0)
        for image_no in range(4):
            image = Image.new('RGB', self.image_size)
            image.save('/tmp/i{}.gif'.format(image_no))
            self.app.addImage(
                'image_{}'.format(image_no), '/tmp/i{}.gif'.format(image_no), image_no // 2, image_no % 2
            )
        self.app.stopLabelFrame()
        self.app.addLabel('item_no', '{}/{}'.format(self.item_no, len(self.items)))
        self.app.addListBox('tags', [])
        self.app.setListBoxRows('tags', 8)
        self.app.addButtons(["Like", "Dislike", 'Stop'], self.press, 3, 0, 2)  # Row 3,Column 0,Span 2

    def downscale(self, image):
        w, h = image.size
        image = add_border(image, w, h)
        return image.resize(self.image_size, Image.BICUBIC)

    def press(self, btn):
        if btn == "Stop":
            pprint([i.tags for i in self.items[:self.item_no]])
            self.app.stop()
        elif btn == 'Like':
            self.items[self.item_no].like()
            self.next_item()
        else:
            self.items[self.item_no].unlike()
            self.next_item()

    def stop(self, _):
        self.press('Stop')

    def like(self, _):
        self.liked += 1
        self.press('Like')

    def dislike(self, _):
        self.press('Dislike')

    def back(self, _):
        self.item_no -= 1
        self.update_content()

    def next(self, _):
        self.next_item()

    def next_item(self):
        self.item_no += 1
        if self.item_no >= len(self.items):
            self.stop(None)
        self.update_content()

    def update_content(self):
        item = self.items[self.item_no]
        self.app.setLabel('item_no', '{}/{} ({} Liked)'.format(self.item_no, len(self.items), self.liked))
        self.app.updateListItems('tags', self.items[self.item_no].tags)
        for image_no in range(4):
            try:
                image = self.downscale(Image.open(item.picture_files[image_no]))
            except IndexError:
                image = Image.new('RGB', self.image_size)
            image.save('/tmp/i{}.gif'.format(image_no))
            app.reloadImage('image_{}'.format(image_no), '/tmp/i{}.gif'.format(image_no))


args = parse_command_line()
if isfile(STATE_FILE):
    with open(STATE_FILE, 'rb') as pickle_file:
        params = load(pickle_file)
    item_file_name = params['item_file']
    start = params['start']
else:
    item_file_name = join(args.save_folder, args.item_file)
    start = args.start

with open(item_file_name, 'rb') as pickle_file:
    items = load(pickle_file)

try:
    app = gui()

    ui = LikeItemsUI(app, items, start, args.size, liked_only=args.liked_only)

    app.go()
finally:
    item_file_name = '.'.join(item_file_name.split('.')[:2] + [str(ui.item_no)])
    with open(join(item_file_name), 'wb') as pickle_file:
        dump(items, pickle_file)

    with open(STATE_FILE, 'wb') as pickle_file:
        dump({'item_file': item_file_name, 'start': ui.item_no}, pickle_file)
