#!/usr/bin/python

import os
import sys
import json
import caffe
import pdb
import numpy
import h5py


def shape_to_array(shape):
    retval = []
    for x in shape:
        retval.append(x)
    return retval

def create_dataset_from_blob(f,blob):
    return f.create_dataset(None, blob.shape, data=blob.data, dtype='f')


def export_model(model_def, model_weights):
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    shape = net.blobs['data'].shape
    net.blobs['data'].reshape( 1, shape[1], shape[2], shape[3] )
    net.blobs['data'].data[...] = numpy.ones(net.blobs['data'].shape)
    net.forward()
    f = h5py.File('outputs.h5', 'w')
    layer_group = f.create_group('layer_outputs')

    for b in net.blobs:
        layer_group[b] = create_dataset_from_blob(f, net.blobs[b])

    with open (model_def, "r" ) as myfile:
        model_text = myfile.read()

    f['model_prototxt'] = model_text

    layer_weights = f.create_group('model_weights')

    for p in net.params:
        weight_group = layer_weights.create_group(p)
        weight_group[p + '_W'] = create_dataset_from_blob(f, net.params[p][0])
        weight_group[p + '_b'] = create_dataset_from_blob(f, net.params[p][1])

    f.close()

if __name__ == "__main__":
    caffe.set_mode_cpu()
    export_model(sys.argv[1], sys.argv[2])
