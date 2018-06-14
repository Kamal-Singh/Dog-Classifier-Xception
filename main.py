from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
import cv2                
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image   
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from data import dog_names
import matplotlib.image as mpimg
from PIL import Image
from keras.applications.xception import Xception, preprocess_input

ResNet50_model = ResNet50(weights='imagenet')
xception = Xception(weights='imagenet', include_top=False)
def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    img = resnet_preprocess(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def extract_Xception(tensor):
	return xception.predict(preprocess_input(tensor))

def predict_breed(img_path):
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    predicted_vector = Xception_model.predict(bottleneck_feature)
    return predicted_vector

def app(img_path):
    img = Image.open(img_path)
    if dog_detector(img_path):
        plt.imshow(img)
        plt.title("Dog detected!")
        predicted_vector = predict_breed(img_path)  
        max_index=np.argmax(predicted_vector)
        per=predicted_vector[0][max_index]*100 
        plt.xlabel("It's breed is ... {} ({}%)".format(dog_names[max_index],per))
        plt.show()
    elif face_detector(img_path):
        plt.imshow(img)
        plt.title("Hello, human!")
        predicted_vector = predict_breed(img_path)  
        max_index=np.argmax(predicted_vector)
        per=predicted_vector[0][max_index]*100 
        plt.xlabel("You look like a ... {} ({}%)".format(dog_names[max_index],per))
        plt.show()
    else:
        plt.imshow(img)
        plt.title("Error! No human or dog detected!")
        plt.show()

Xception_model = Sequential() 
Xception_model.add(GlobalAveragePooling2D(input_shape=(7,7,2048)))
Xception_model.add(Dropout(0.45))
Xception_model.add(Dense(2660))  
Xception_model.add(Dense(133, activation='softmax'))
# Xception_model.summary()
sgd = optimizers.SGD(momentum=0.35, nesterov=True)
Xception_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
Xception_model.load_weights('./saved_models/weights.best.Xception.hdf5')

app('./images/pug1.jpg')