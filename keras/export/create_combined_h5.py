'''
This script produces an hdf5 file with outputs for each layer
Will force inputs to be the 'tf' dimensionality ordering but supports both theano and tensorflow backends
Inputs: 1) Model configuration (json, written out from keras using model.to_json())
        2) weights (HDF5 written out from keras using model.save_weights())
Outputs:  An HDF5 file containing
        1) the output at each layer produced by a forward pass of the test-image
        2) model configuration
        3) weights
        4) test-image
'''
import sys
import json
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import layer_from_config

def load_json(filename):
    "Loads the model configuration file"
    try:
        with open (filename, 'r') as f:
            data = f.read()
        model_config = json.loads(data)
        return model_config
    except:
        print "Error: Check if file exists and/or if it is valid json format"


def create_model(model_config):
    "Creates a base model off of the class_name in the model config"
    if (model_config.has_key('class_name') and model_config.has_key('config')):
        model = eval(model_config['class_name'])()
    else:
        print "This json file has not been generated using keras - REDO and RERUN"
        exit
    return model


def load_weights(model, weights_path):
    "Loads weights from the HDF5 file to the relevant layers"
    f = h5py.File(weights_path)
    layer_attr = f.attrs.keys()[0]
    for k, layer in enumerate(f.attrs['layer_names']):
        if k >= len(model.layers):
            break
        g = f[layer]
        weights = [g[attr] for attr in g.keys()]
        if len(weights) > 0:
            model.layers[k].set_weights([weights[0], weights[1]])
    f.close()
    return model


def construct_layers(layer_config, model, weights_path):
    "Adds layers to the base model and loads weights onto each of those layers"
    for config in layer_config:
        model.add(layer_from_config(config))

    print "Number of layers loaded {}".format(len(model.layers))
    model_with_weights = load_weights(model, weights_path)
    return model_with_weights


def get_test_image(input_shape):
    "Generates a test image of ones using the input_shape"
    input_shape.insert(0, 1)
    return np.ones(input_shape)


def copy_weights(output_file, weights_file):
    output_file.create_group('model_weights')

    f = h5py.File(weights_file)
    for layer in f.attrs['layer_names']:
        f.copy(f[layer], output_file['model_weights'])
    return output_file


def generate_hdf5(config, weights_file):
    "Writes an HDF5 file containing weights, model configuration and outputs at each layer"
    model_config = config['config']
    test_image = get_test_image(model_config[0]['config']['batch_input_shape'][1:])

    f = h5py.File('decomposed_model.h5', 'w')
    f['model_config'] = json.dumps(model_config)
    f['test_image'] = test_image
    layer_group = f.create_group('layer_outputs')

    # Create new models incrementally by adding a layer each time and predict on the model
    for i in range(0, len(model_config)):
        model = create_model(config)
        layer_name = "layer_{}_type_{}".format(i, model_config[i]['class_name'])
        model = construct_layers(model_config[0:i+1], model, weights_file)
        output = model.predict(test_image)
        print "Output shape of the final layer {}\n".format(output.shape)
        layer_group[layer_name] = output

    # Write out the original weights to the new hdf5 file
    f = copy_weights(f, weights_file)
    f.close()
    print "HDF5 file written"



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Missing Input: model file and weights are required")
        exit
    else:
        generate_hdf5(load_json(sys.argv[1]), sys.argv[2])
