import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import argparse
import json

## Arguments
parser = argparse.ArgumentParser(description="the script uses a trained network to predict the class for an input image")
parser.add_argument('image_path',help='the path for the image want to predict',type=str)
parser.add_argument('model',type=str)
parser.add_argument('--top_k',help='Return the top KK most likely classes',type=int,default=5)
parser.add_argument('--category_names',help='Path to a JSON file mapping labels to flower names',type=str)
args = parser.parse_args()

## load the model 
model = tf.keras.models.load_model(args.model,custom_objects={'KerasLayer':hub.KerasLayer})
model.build((None, 224, 224, 3))

## image processing stage
image = Image.open(args.image_path)
image = np.asarray(image)
image = tf.convert_to_tensor(image,tf.float32)
image = tf.image.resize(image,(224,224))
image /= 255
image = image.numpy()
image = np.expand_dims(image,axis=0)

## find probabilities and classes for the image  
predictions = model.predict(image)
probs, classes = tf.nn.top_k(predictions, k=args.top_k)
probs = probs.numpy()
classes = classes.numpy()
classes += 1
print(probs)
print(classes)
## provide class names if asked
if args.category_names != None:
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    names = [] 
    for i in classes[0]:
        name = class_names[str(i)]
        names.append(name)
    print(names)    