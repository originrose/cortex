from __future__ import print_function
'''
This script produces an hdf5 file with outputs for each layer.

Inputs: 1) Model configuration (json, written out from keras using model.to_json())
        2) weights (HDF5 written out from keras using model.save_weights())

Outputs:  An HDF5 file containing
        1) the output at each layer produced by a forward pass of the test-image
        2) test-image
'''
import h5py
import argparse
import numpy as np
from keras import backend as K
from keras.models import model_from_json, load_model


MODEL_DEFAULTS = {
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
}


def get_activations(model, layer, X_batch):
    """Adapted from: https://github.com/fchollet/keras/issues/41 as a
    workaround for perceived softmax issue. """
    # phase 0 is test/inference, 1 is learning
    phase = K.learning_phase()
    get_activations = K.function(
        [model.layers[0].input, phase],
        [model.layers[layer].output],
    )
    activations = get_activations([X_batch, 0])
    return activations


def load_sidecar_model(arch_fname, weights_fname):
    """Return a model saved by the previous keras standard format of
    architecture in json, weights in h5 sidecar."""
    with open(arch_fname, 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(weights_fname)
    # compile with sensible detfaults for typical n-way classifier, although
    # we could possibly detect and handle different cases.
    model.compile(**MODEL_DEFAULTS)
    return model


def load_single_model(fname):
    """Wrapper for load_model that ensures model is compiled."""
    m = load_model(fname)
    try:
        m.compile(**MODEL_DEFAULTS)
    except:
        return m


def get_test_image(input_shape):
    """Generates a test image of ones using the input_shape"""
    input_shape[0] = 1
    return np.ones(input_shape)


def layer_outputs(h5file, json_file=None, out_fname='layer_outputs.h5'):
    """Writes an HDF5 file containing outputs at each layer."""
    if json_file:
        model = load_sidecar_model(json_file, h5file)
    else:
        model = load_single_model(h5file)

    input_dims = model.input.get_shape().as_list()
    test_image = get_test_image(input_dims)

    with h5py.File(out_fname, 'w') as outf:
        outf['test_image'] = test_image
        layer_group = outf.create_group('layer_outputs')

        for i, lyr in enumerate(model.layers):
            output = get_activations(model, i, test_image)

            # output 0 is kind of a hack here, keras actually supports
            # multiple outputs from a layer, although not common in modeling
            # practice
            layer_group[lyr.name] = output[0]

        # Write out the original weights to the new hdf5 file
        print("HDF5 file written: {0}".format(out_fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("h5_file",
        help="The h5 model or weights file. If weights, specify --json")
    arg("out_h5_file",
        help="The h5 model to be generated that contains outputs.")
    arg("--json", help="A json file containing the model's architecture.",
        action="store", type=str, default=None)
    args = parser.parse_args()

    if not (args.h5_file and args.out_h5_file):
        print("Missing required arguments, see help.")
        exit()
    if args.json:
        layer_outputs(
            args.h5_file,
            json_file=args.json,
            out_fname=args.out_h5_file,
        )
    else:
        layer_outputs(
            args.h5_file,
            out_fname=args.out_h5_file,
        )
