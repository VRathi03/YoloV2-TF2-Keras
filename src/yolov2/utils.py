from pathlib import Path
import numpy as np


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def normalize(image):
    return image / 255.


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def set_pre_trained_weights(model, nb_conv, path_to_pre_trained_weights):
    weight_reader = WeightReader(path_to_pre_trained_weights)
    weight_reader.reset()
    print(nb_conv)
    for i in range(1, nb_conv + 1):
        conv_layer = model.get_layer('conv_' + str(i))

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            print("Bias: ", bias.shape)
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            print("Kernel before reshape:", kernel.shape)
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            print("Kernel after Reshape:", kernel.shape)
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    return model


def initialize_weights(layer, sd):
    weights = layer.get_weights()
    new_kernel = np.random.normal(size=weights[0].shape, scale=sd)
    new_bias = np.random.normal(size=weights[1].shape, scale=sd)
    layer.set_weights([new_kernel, new_bias])
