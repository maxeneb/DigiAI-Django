from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from mlmodel.serializers import FileSerializer
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

# from keras.models import load_model
# from keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

from sklearn import metrics
import seaborn as sns

# class FileUploadView(APIView):
#     parser_class = (FileUploadParser,)

    # def post(self, request, *args, **kwargs):
    #     file_serializer = FileSerializer(data=request.data)

    #     if file_serializer.is_valid():
    #         file_data = file_serializer.validated_data['file']
    #         # Load your trained model
    #         model = load_model('ai-generated-and-human-made-painting.keras')
    #         model.load_weights('ai-generated-and-human-made-painting.keras')

    #         print(model.summary())

    #         # Preprocess the uploaded image

    #         image_stream = io.BytesIO(file_data.read())
    #         uploaded_image = Image.open(image_stream).convert('RGB')
    #         resized_image = uploaded_image.resize((512, 512))  # adjust the target size to match your model's expected input size
    #         img_array = np.array(resized_image)
    #         img_array = np.expand_dims(img_array, axis=0)

    #         # Make a prediction
    #         prediction = model.predict(img_array)
    #         result = np.argmax(prediction)  
    #         confidence = np.max(prediction)

    #         # Map the result to the corresponding label
    #         labels = ['AI_ART', 'NON_AI_ART']  # adjust this to match your labels
    #         predicted_label = labels[result]

    #         return Response({'result': predicted_label, 'confidence': float(confidence)}, status=status.HTTP_200_OK)
    #     else:
    #         return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def pre_process_image(path, image_shape=512, channels=3, norm_factor=255.):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=channels)
    img = tf.image.resize(img, size=(image_shape, image_shape))
    img = tf.expand_dims(img, axis=0)
    img = img / norm_factor  # Note: Division by norm_factor to normalize
    return img

def test_custom_image(image_path, model, classes):
    random_image = mpimg.imread(image_path)
    predicted_value = model.predict(pre_process_image(image_path))
    predicted_label = classes[np.argmax(predicted_value)]
    confidence = np.max(predicted_value) * 100
    confidence = round(confidence, 2) 

    
    return predicted_label, confidence

def confusion_matrix_plot(y_true, y_pred, class_names, figsize=(10,10)):
    """"Confusion Matrix for true values and predicted values"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize = figsize)
    sns.heatmap(cm, annot=True, cmap="crest", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
class_names = ['AI_art', 'non_AI_art']
class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def __init__(self):
        self.model = load_model('aiornonai.keras')
        self.classes = ['AI_ART', 'NON_AI_ART']

    def post(self, request, *args, **kwargs):
        file_serializer = FileSerializer(data=request.data)

        if file_serializer.is_valid():
            file_data = file_serializer.validated_data['file']
            upload_dir = 'upload_files' 
            file_path = os.path.join(upload_dir, file_data.name)
            with open(file_path, 'wb+') as f:
                for chunk in file_data.chunks():
                    f.write(chunk)

            predicted_label, confidence = test_custom_image(file_path, self.model, self.classes)

            return Response({'result': predicted_label, 'confidence': confidence}, status=status.HTTP_200_OK)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)