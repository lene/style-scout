from appJar import gui
from pickle import load, dump
from argparse import ArgumentParser
from os.path import join
from PIL import Image
from pprint import pprint

from data_sets.contains_images import add_border

SAVE_FOLDER = 'data'


def parse_command_line():
    parser = ArgumentParser(description="Simple GUI to like/dislike eBay items")
    parser.add_argument('--save-folder', default=SAVE_FOLDER, help='Folder under which items are stored')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument(
        '--item-file', default=None, help="Pickle file from which to load downloaded items"
    )
    parser.add_argument(
        '--likes-file', help="JSON file containing the liked item IDs"
    )

    return parser.parse_args()


class LikeItemsUI:

    IMAGE_SIZE = (400, 400)

    def __init__(self, app, items, start):
        self.item_no = start
        self.items = items
        self.app = app

        self.app.addLabel('title', 'Like or dislike eBay items here')

        self.app.addLabel('item_no', '{}/{}'.format(self.item_no, len(self.items)))
        self.app.startLabelFrame("Simple", 0, 0)
        for image_no in range(4):
            image = Image.new('RGB', self.IMAGE_SIZE)
            image.save('/tmp/i{}.gif'.format(image_no))
            self.app.addImage(
                'image_{}'.format(image_no), '/tmp/i{}.gif'.format(image_no), image_no // 2, image_no % 2
            )
        self.app.stopLabelFrame()

        self.app.addListBox('tags', [])
        self.app.setListBoxRows('tags', 8)

        self.app.addButtons(["Like", "Dislike", 'Stop'], self.press, 3, 0, 2)  # Row 3,Column 0,Span 2

        self.update_content()

    def downscale(self, image, size):
        w, h = image.size
        image = add_border(image, w, h)
        return image.resize(size, Image.BICUBIC)

    def press(self, btn):
        if btn == "Stop":
            pprint([str(i) for i in self.items[:self.item_no]])
            self.app.stop()
        elif btn == 'Like':
            self.items[self.item_no].like()
            self.next_item()
        else:
            self.next_item()

    def next_item(self):
        self.item_no += 1
        self.update_content()

    def update_content(self):
        item = self.items[self.item_no]
        self.app.setLabel('item_no', '{}/{}'.format(self.item_no, len(self.items)))
        self.app.updateListItems('tags', self.items[self.item_no].tags)
        for image_no in range(4):
            try:
                image = self.downscale(Image.open(item.picture_files[image_no]), self.IMAGE_SIZE)
            except IndexError:
                image = Image.new('RGB', self.IMAGE_SIZE)
            image.save('/tmp/i{}.gif'.format(image_no))
            app.reloadImage('image_{}'.format(image_no), '/tmp/i{}.gif'.format(image_no))


args = parse_command_line()

with open(join(args.save_folder, args.item_file), 'rb') as pickle_file:
    items = load(pickle_file)

app = gui()

ui = LikeItemsUI(app, items, args.start)

app.go()

with open(join(args.save_folder, args.item_file + '.{}.liked'.format(ui.item_no)), 'wb') as pickle_file:
    dump(items, pickle_file)
