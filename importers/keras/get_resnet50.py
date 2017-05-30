from keras.applications.resnet50 import ResNet50

output_pre = "models/resnet50"

model = ResNet50(include_top=True, weights='imagenet')
with open(output_pre + '.json', 'w') as jsonf:
    jsonf.write(model.to_json())
model.save_weights(output_pre + ".h5")
