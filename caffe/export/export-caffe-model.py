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


def export_model(model_def, model_weights):
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    shape = net.blobs['data'].shape
    net.blobs['data'].reshape( 1, shape[1], shape[2], shape[3] )
    net.blobs['data'].data[...] = numpy.ones(net.blobs['data'].shape)
    net.forward()
    f = h5py.File('outputs.h5', 'w')
    layer_group = f.create_group('layer_outputs')

    for b in net.blobs:
        layer_group[b] = f.create_dataset(b, net.blobs[b].shape, data=net.blobs[b].data, dtype='f')
        
    f.close()
    


if __name__ == "__main__":
    caffe.set_mode_cpu()
    export_model(sys.argv[1], sys.argv[2])

0
