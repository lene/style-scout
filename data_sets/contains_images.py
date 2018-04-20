from typing import Callable

from PIL import Image
import numpy


def crop_bottom(image: Image.Image, w: int, h: int) -> Image.Image:
    if w > h:
        image = image.crop(((w - h) / 2, 0, w - (w - h) / 2, h))
    elif h > w:
        image = image.crop((0, 0, w, w))
    return image


def add_border(image: Image.Image, w: int, h: int) -> Image.Image:
    if w == h:
        return image
    new_image = Image.new("RGB", (max(w, h), max(w, h)))
    new_image.paste(image, (0, 0))
    return new_image


class ContainsImages:

    DEPTH = 3

    def __init__(self, x_size: int, y_size: int) -> None:
        """
        :param x_size: width of the images
        :param y_size: height of the images
        """
        self.size = (x_size, y_size)
        self.num_features = x_size * y_size * self.DEPTH

    def downscale(
            self, image: Image.Image, method: Callable[[Image.Image, int, int], Image.Image]=add_border
    ) -> numpy.array:
        w, h = image.size
        image = method(image, w, h)
        return numpy.asarray(image.resize(self.size, Image.BICUBIC))

    @classmethod
    def show_image(cls, rgb_values: numpy.ndarray, label: str='') -> None:
        import matplotlib.pyplot as plt
        plt.imshow(rgb_values, cmap='gray' if cls.DEPTH == 1 else None)
        plt.title(label)
        plt.show()
