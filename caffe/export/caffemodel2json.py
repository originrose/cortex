#!/usr/bin/python

import os
import sys
import json
import urllib2
import argparse
import tempfile
import subprocess
from google.protobuf.descriptor import FieldDescriptor as FD

# inspired by https://github.com/dpp-name/protobuf-json/blob/master/protobuf_json.py
def pb2json(pb, print_arrays):
	_ftype2js = {
		FD.TYPE_DOUBLE: float,
		FD.TYPE_FLOAT: float,
		FD.TYPE_INT64: long,
		FD.TYPE_UINT64: long,
		FD.TYPE_INT32: int,
		FD.TYPE_FIXED64: float,
		FD.TYPE_FIXED32: float,
		FD.TYPE_BOOL: bool,
		FD.TYPE_STRING: unicode,
		FD.TYPE_BYTES: lambda x: x.encode('string_escape'),
		FD.TYPE_UINT32: int,
		FD.TYPE_ENUM: int,
		FD.TYPE_SFIXED32: float,
		FD.TYPE_SFIXED64: float,
		FD.TYPE_SINT32: int,
		FD.TYPE_SINT64: long,
		FD.TYPE_MESSAGE: lambda x: pb2json(x, print_arrays = print_arrays),
		'unknown' : lambda x: 'Unknown field type: %s' % x
	}
	js = {}
	for field, value in pb.ListFields():
		ftype = _ftype2js[field.type] if field.type in _ftype2js else _ftype2js['unknown']
		if field.label == FD.LABEL_REPEATED:
			js_value = map(ftype, value)
			if not print_arrays and (field.name == 'data' and len(js_value) > 8):
				head_n = 5
				js_value = js_value[:head_n] + ['(%d elements more)' % (len(js_value) - head_n)]
		else:
			js_value = ftype(value)
		js[field.name] = js_value
	return js

parser = argparse.ArgumentParser('Dump model_name.caffemodel to a file JSON format for debugging')
parser.add_argument(metavar = 'caffe.proto', dest = 'caffe_proto', help = 'Path to caffe.proto (typically located at CAFFE_ROOT/src/caffe/proto/caffe.proto)')
parser.add_argument(metavar = 'model.caffemodel', dest = 'model_caffemodel', help = 'Path to model.caffemodel')
parser.add_argument('--data', help = 'Print all arrays in full', action = 'store_true')
parser.add_argument('--codegenDir', help = 'Path to an existing temporary directory to save generated protobuf Python classes', default = tempfile.mkdtemp())
args = parser.parse_args()

local_caffe_proto = os.path.join(args.codegenDir, os.path.basename(args.caffe_proto))
with open(local_caffe_proto, 'w') as f:
	f.write((urllib2.urlopen(args.caffe_proto) if 'http' in args.caffe_proto else open(args.caffe_proto)).read())
	
subprocess.check_call(['protoc', '--proto_path', os.path.dirname(local_caffe_proto), '--python_out', args.codegenDir, local_caffe_proto])
sys.path.insert(0, args.codegenDir)
import caffe_pb2

deserialized = caffe_pb2.NetParameter() if os.path.splitext(args.model_caffemodel)[1] == '.caffemodel' else caffe_pb2.BlobProto()
deserialized.ParseFromString(open(args.model_caffemodel, 'rb').read())

json.dump(pb2json(deserialized, args.data), sys.stdout, indent = 2)
