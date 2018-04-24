from typing import Tuple

from keras.applications import (
    InceptionV3, Xception, VGG16, VGG19, ResNet50, InceptionResNetV2, DenseNet121, DenseNet169, DenseNet201,
    NASNetLarge
)
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model


def model(
        network_type: Model, input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]
) -> Model:
    base_model = network_type(include_top=False, weights=None, input_shape=input_shape, classes=classes)

    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # add some fully-connected layers
    for layer_size in connected_layers:
        x = Dense(layer_size, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def inception(input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)) -> Model:
    return model(InceptionV3, input_shape, classes, connected_layers)


def xception(input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)) -> Model:
    return model(Xception, input_shape, classes, connected_layers)


def vgg16(input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)) -> Model:
    return model(VGG16, input_shape, classes, connected_layers)


def vgg19(input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)) -> Model:
    return model(VGG19, input_shape, classes, connected_layers)


def resnet50(input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)) -> Model:
    return model(ResNet50, input_shape, classes, connected_layers)


def inception_resnet_v2(
        input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)
) -> Model:
    return model(InceptionResNetV2, input_shape, classes, connected_layers)


def densenet121(
        input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)
) -> Model:
    return model(DenseNet121, input_shape, classes, connected_layers)


def densenet169(
        input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)
) -> Model:
    return model(DenseNet169, input_shape, classes, connected_layers)


def densenet201(
        input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)
) -> Model:
    return model(DenseNet201, input_shape, classes, connected_layers)


def nasnet(
        input_shape: Tuple[int, ...], classes: int, connected_layers: Tuple[int, ...]=(1024,)
) -> Model:
    return model(NASNetLarge, input_shape, classes, connected_layers)
