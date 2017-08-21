from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model


def model(network_type, input_shape, classes):
    base_model = network_type(include_top=False, weights=None, input_shape=input_shape, classes=classes)

    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def inception(input_shape, classes):
    return model(InceptionV3, input_shape, classes)


def xception(input_shape, classes):
    return model(Xception, input_shape, classes)


def vgg16(input_shape, classes):
    return model(VGG16, input_shape, classes)


def vgg19(input_shape, classes):
    return model(VGG19, input_shape, classes)


def resnet50(input_shape, classes):
    return model(ResNet50, input_shape, classes)
