"""
Extractor class to obtain inception features from images
Nitesh Kumar Chaudhary
niteshku001@e.ntu.edu.sg

"""
import pickle
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import sys

class Extractor_inception():
    def __init__(self, image_shape=(299, 299, 3), weights=None):

        self.weights = weights  # so we can check elsewhere which model

        input_tensor = Input(image_shape)
        # Get model with pretrained weights.
        base_model = InceptionV3(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=True
        )

        # We'll extract features at the final pool layer.
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

    def extract(self, image_path):
        img = image.load_img(image_path)

        return self.extract_image(img)

    def extract_image(self, img):
        x = image.img_to_array(img)
        #print("Ravdess images shape: ", x.shape)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

