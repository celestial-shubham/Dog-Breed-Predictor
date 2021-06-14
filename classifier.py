from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import pickle

def image_classification(img,weights_file):
    # Get all breeds from the txt files
    with open('breeds.txt', 'r') as reader:
        breeds=[]
        for breed in reader:
            try:
                breed=breed.replace('_',' ')
                breed=breed.title()
            except:
                breed=breed.title()
            breeds.append(breed[:-1])

    # Loading the model
    def load_model(model_path):
    #print(f"loading saved model from : {model_path}")
      model = tf.keras.models.load_model(model_path,custom_objects = {"KerasLayer":hub.KerasLayer})
      return model

    model = load_model("/content/drive/MyDrive/Dog breed prediction/Model/20201214-13301607952655-full-image-set-mobilenetv2-Adam.h5")

    # Creating the array to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Turning the image into a numpy array, normalizing and loading the image
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32)/255)
    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)
    return breeds[np.argmax(prediction)]