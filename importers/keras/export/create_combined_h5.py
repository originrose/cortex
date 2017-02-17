"""
This is a utility that takes the older style sidecar keras format convention
of:

arch.json
weights.h5

and combines them into the currently preferred single h5 file.

Invoke as follows:

python export/create_combined_h5.py arch.json weights.h5 combined.h5
"""
from __future__ import print_function # supports both 2/3

import sys
from keras.models import model_from_json


json_file = sys.argv[1]
weights_file = sys.argv[2]
out_file = sys.argv[3]

with open(json_file) as jsonf:
    m = model_from_json(jsonf.read())
    m.load_weights(weights_file)
    m.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    m.save(out_file)
