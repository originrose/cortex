from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
import argparse

model = ResNet50(weights='imagenet')

def image_test(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print(preds)
    print('Predicted: ', decode_predictions(preds))


parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="Path to the image to test")
args = parser.parse_args()
image_test(args.img_path)
