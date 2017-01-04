from keras.applications import vgg16
import json

output_pre = "models/vgg16"

model = vgg16.VGG16(include_top=True, weights='imagenet')
with open(output_pre + '.json', 'w') as jsonf:
    jsonf.write(model.to_json())
model.save_weights(output_pre + ".h5")
