# Imports
import os
import face_recognition
import cv2
import random
import time
import sys

import matplotlib.pyplot as plt 
import numpy as np

from tkinter import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import keras 
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Dense, MaxPool2D, Flatten 


# Convenience functions
def small_dot(tkinter_canvas, centre_x, centre_y, radius=5, fill="red"):
    """Given the centre point of a dot, this convenience function will draw a small dot with given radius"""
    
    tkinter_canvas.create_oval(centre_x - radius, centre_y - radius,
                               centre_x + radius, centre_y + radius, fill=fill)
    
    return

def random_dot(tkinter_canvas, tk_width, tk_height):
    
    border = 5 # Should be same, or higher than radius of dots
    
    random_width = random.randint(border, tk_width - border)
    random_height = random.randint(border, tk_height - border)
    
    small_dot(tkinter_canvas, random_width, random_height)
    
    return random_width, random_height

def neural_model(dummy_sample, base_channels=8, dense_per_layer=50):
    
    print("About to initialise a neural network with input shape: ", dummy_sample.shape)
    
    l2_reg = keras.regularizers.l2(0.0001)

    visible = Input(shape=(dummy_sample.shape))
    
    c11 = Conv2D(base_channels, 3)(visible)
    c12 = Conv2D(base_channels, 3)(c11)
    p1 = Conv2D(base_channels * 2, 1, strides=2)(c12)
    c21 = Conv2D(base_channels * 2, 3)(p1)
    c22 = Conv2D(base_channels * 2, 3)(c21)
    p2 = Conv2D(base_channels * 4, 1, strides=2)(c22)
    #c31 = Conv2D(8, 3)(p2)
    #c32 = Conv2D(8, 3)(c31)
    #p3 = Conv2D(16, 1, strides=2)(c32)
    
    f1 = Flatten()(p2)
    d1 = Dense(dense_per_layer, activation="relu", kernel_regularizer=l2_reg)(f1)
    d2 = Dense(dense_per_layer, activation="relu", kernel_regularizer=l2_reg)(d1)
    output = Dense(2)(d2)
    
    model = Model(inputs=visible, outputs=output)
    
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer="adam")
    
    return model

def extract_facial_features(frame, display=False):
    
    # Basic code for facial landmark extraction from webcam from:
    # https://elbruno.com/2019/05/29/vscode-lets-do-some-facerecognition-with-20-lines-in-python-3-n/    
    rgb_frame = frame[:, :, ::-1].copy()
    frame_copy = frame.copy()
    bw_frame = np.mean(rgb_frame, axis=2)

    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    
    # Extract region around eyes, before green lines added. Uses face_recognition
    border_height = 10
    border_width = 15
    
    # Creat linear ingredients to bundle with the eye data
    grad_x = np.zeros(frame_copy.shape[:2], dtype=np.float)
    grad_y = np.zeros(frame_copy.shape[:2], dtype=np.float)
    
    for i in range(border_height * 2):
        grad_x[i, :] = i / (border_height * 2)
        
    for j in range(border_width * 2):
        grad_y[:, j] = j / (border_width * 2)
    
    try:
        left_eye = np.mean(np.array(face_landmarks_list[0]["left_eye"]), axis=0, dtype=int)
        left_eye_region = bw_frame[left_eye[1] - border_height: left_eye[1] + border_height,
                                   left_eye[0] - border_width: left_eye[0] + border_width]
        left_eye_x_grad = grad_x[left_eye[1] - border_height: left_eye[1] + border_height,
                                 left_eye[0] - border_width: left_eye[0] + border_width]
        left_eye_y_grad = grad_y[left_eye[1] - border_height: left_eye[1] + border_height,
                                 left_eye[0] - border_width: left_eye[0] + border_width]
        
        left_eye_flattened = left_eye_region.reshape(1,-1)[0]
    
        right_eye = np.mean(np.array(face_landmarks_list[0]["right_eye"]), axis=0, dtype=int)
        right_eye_region = bw_frame[right_eye[1] - border_height: right_eye[1] + border_height,
                                    right_eye[0] - border_width: right_eye[0] + border_width]
        right_eye_x_grad = grad_x[right_eye[1] - border_height: right_eye[1] + border_height,
                                  right_eye[0] - border_width: right_eye[0] + border_width]
        right_eye_y_grad = grad_y[right_eye[1] - border_height: right_eye[1] + border_height,
                                  right_eye[0] - border_width: right_eye[0] + border_width]
        
        right_eye_flattened = right_eye_region.reshape(1,-1)[0]
            
        # Scale features
        scaler = StandardScaler()
        left_eye_region = scaler.fit_transform(left_eye_region)
        right_eye_region = scaler.fit_transform(right_eye_region)
        
        eyes_and_gradients = np.stack((left_eye_region, left_eye_x_grad, left_eye_y_grad,
                                       right_eye_region, right_eye_x_grad, right_eye_y_grad), axis=2)
    except IndexError:
        print("Could not extract eye regions, probably because face not detected")
        return [], [], [], []
        
    for face_landmarks in face_landmarks_list:

        for facial_feature in face_landmarks.keys():
            pts = np.array([face_landmarks[facial_feature]], np.int32) 
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame, [pts], False, (0,255,0))

    if display:
        cv2.imshow('Video', frame)
        
    # print(face_landmarks_list)
    
    # I suspect this code will break if multiple faces
    landmark_array = np.array(np.zeros((0, 2)))
    if face_landmarks_list != []:
        for landmark in face_landmarks_list[0].values():
            landmark_array = np.concatenate((landmark_array, np.array(landmark)))
    else:
        print("No face detected") 
    
    # Concatenate the extracted facial features, with the region around the eyes 
    everything_array = np.concatenate(
        (landmark_array[0], left_eye_flattened, right_eye_flattened))
    landmark_array = landmark_array[0]
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    everything_array = everything_array.reshape(1, -1)
    landmark_array = landmark_array.reshape(1, -1)
    
    # print(landmark_array[0].shape)
    
    return rgb_frame, everything_array, landmark_array, eyes_and_gradients

def predict_gaze(video_capture, webcam_resolution,  
                 tk_width, tk_height, model, model_type, canvas):
    
    ret, frame = video_capture.read()
    (rgb_frame, everything_array, 
     landmark_array, eyes_and_gradients) = extract_facial_features(frame)
    
    try:
        if model_type == "neural net":
            X = np.expand_dims(eyes_and_gradients, 0)
            predicted_gaze = model.predict(X)[0]
        else:
            predicted_gaze = model.predict(everything_array)[0]
    
        print("Predicted gaze is: ", predicted_gaze)
    except ValueError:
        print("Could not predict, probably no face in image")
        predicted_gaze = np.array([0., 0.])
    
    # Scale the prediction to webcam resolution
    predicted_pixel = [predicted_gaze[0] * tk_width, predicted_gaze[1] * tk_height]
    # print(predicted_pixel, predicted_gaze, webcam_resolution)
    
    # Display the prediction as a grey circle
    small_dot(canvas, predicted_pixel[0], predicted_pixel[1], radius=5, fill="grey")
    
    return rgb_frame, everything_array, eyes_and_gradients, predicted_gaze

def capture(counter, canvas, model, model_type, training_X, training_y, tk_width, tk_height, 
            video_capture, rgb_frame, webcam_resolution, 
            landmark_array, eyes_and_gradients, current_target, predicted_gaze, move_smoothly=False, randomise_dot=True):
    """Will capture an image, coordinate pair when the user is looking at the dot"""
    
    path = "data/MZeina_5/"
    train_every = 1
        
    # print("About to learn...")
    if len(landmark_array) != 0:
        current_target = np.array(current_target) / np.array([tk_width, tk_height])
        
        if model_type == "neural net":
            # Neural network can train on each sample at a time, unlike random forest
            training_X = np.expand_dims(eyes_and_gradients, 0)
            training_y = np.expand_dims(current_target, 0)
            # training_X.append(eyes_and_gradients)
        else:
            training_X.append(landmark_array[0])
            training_y.append(current_target)
        
        plt.imsave(path + str(current_target) + ".jpg", rgb_frame)
        
        if counter % train_every == 0:
            model.fit(training_X, training_y)
        
    else:
        print("Face not detected, will not train on this sample")
    
    #canvas.delete("all")
    if move_smoothly:
        speed = 20
        scaled_counter = (counter * speed) % (tk_width * tk_height)
        target_x = (scaled_counter // tk_height * speed) % tk_width
        if (scaled_counter // tk_height)%2 == 0:
            target_y = scaled_counter % tk_height
        else:
            # reverse the direction for alternative lines, so it doesn't skip up to the top
            target_y = tk_height - scaled_counter % tk_height
        print("counter, scaled_counter, are :", counter, scaled_counter)
        print("about to move small circle to", target_x, target_y)
        small_dot(canvas, target_x, target_y)
        current_target = [target_x, target_y]
    elif randomise_dot:
        current_target = random_dot(canvas, tk_width, tk_height)
    # print(random_width, random_height)
    
    return model, current_target

def train_retrospectively(path_to_images, model):
    
    # Build data frame of past images, and the extract features
    # For any non-small neural network, I should replace this technique with a generator
    
    training_X = []
    training_y = []
    counter = 0
    path_to_images = "captures_one/"
    
    # Currently only looks in a single directory
    files = os.listdir(path_to_images)
    
    for file in files:
        print("About to process image number ", counter)
        image = cv2.imread(path_to_images + file)
        rgb_frame, everything_array, landmark_array, eyes_and_gradients = extract_facial_features(image)
        coordinates = [float(coordinate) for coordinate in file[1: -5].split(" ") if len(coordinate) != 0]
        
        training_X.append(eyes_and_gradients)
        training_y.append(coordinates)
        
        counter += 1
                       
    return training_X, training_y



    # Functions that leverage the above to do something useful
def train_and_preview(pretrained_model=None):
    ########## Universal Initialisation ##########
    counter = 0
    captures_per_point = 5
    
    ########## Initialise Video Stream ##########
    video_capture = cv2.VideoCapture(0)
    
    # Extract webcam resolution
    ret, frame = video_capture.read()
    webcam_resolution = frame.shape[:2]
    # print(webcam_resolution) 
    
    ########## Initialise ML Model ##########
    
    # Dummy sample, to help initialising models
    (rgb_frame, dummy_features, 
     landmark_array, eyes_and_gradients) = extract_facial_features(frame)
    
    model_type = "neural net"
    
    if pretrained_model:
        model = pretrained_model
    elif model_type == "random forest":
        # Random forest 
        RF = RandomForestRegressor(n_estimators=500, n_jobs=-1, warm_start=False)
        model = MultiOutputRegressor(RF)
        model.fit(np.zeros_like(dummy_features), np.array([0.5, 0.5]).reshape(1, -1))
    elif model_type == "neural net":
        model = neural_model(eyes_and_gradients)
        model.summary()
        
    # To do:Train on existing pictures
    
    # Initialise
    training_X = []
    training_y = []
    
    ########## Initialise Tkinter ##########
    window = Tk()
    window.attributes("-fullscreen", True)
    
    window.update_idletasks() 
    tk_width = window.winfo_width() 
    tk_height = window.winfo_height()

    canvas = Canvas(window, width = tk_width, height = tk_height)
    canvas.pack()
    
    window.bind("<F11>", lambda event: window.attributes("-fullscreen",
                                        not window.attributes("-fullscreen")))
    window.bind("<Escape>", lambda event: window.attributes("-fullscreen", False))
    # window.bind("c", lambda event: capture(canvas, RFMO, tk_width, tk_height, video_capture, webcam_resolution, landmark_array, current_target, predicted_gaze))
    
    # Variables to store red dot target
    current_target = random_dot(canvas, tk_width, tk_height)
    
    while True:
        
        rgb_frame, landmark_array, eyes_and_gradients, predicted_gaze = predict_gaze(
            video_capture, webcam_resolution, tk_width, tk_height, model, model_type, canvas)
        
        if counter % 4 == 0 and counter != 0:
            canvas.delete("all")
            
            RFMO, current_target = capture(
                counter, canvas, model, model_type, training_X, training_y, tk_width, tk_height, video_capture, 
                rgb_frame, webcam_resolution, landmark_array, eyes_and_gradients, 
                current_target, predicted_gaze, randomise_dot=True)
                
        counter += 1
        
        # Update GUI
        window.update_idletasks()
        window.update()
    return
 

class ScreenshotGenerator(keras.utils.Sequence):
    
    def __init__(self, path_to_images, batch_size=4):
        
        self.path_to_images = path_to_images
        self.batch_size = batch_size
    
        self.files = []# os.listdir(path_to_images)
        self.filenames = []
        
        for root, dirs, files in os.walk(path_to_images):
            for name in files:
                self.files.append(os.path.join(root, name))
                self.filenames.append(name)
    
    def __len__(self):
        
        return len(self.files) // self.batch_size
    
    def __load__(self, index):
        """Returns and processes a single sample, in conjunction with __getitem__"""
        
        # Ensures that if an image is picked without a succesfully detected face, 
        #  it looks for another random one to replace it
        got_good_image = False
        
        while not got_good_image:
        
            file = self.files[index]
            filename = self.filenames[index]
                        
            image = cv2.imread(file)
            
            rgb_frame, everything_array, landmark_array, eyes_and_gradients = extract_facial_features(image)
            coordinates = [float(coordinate) for coordinate in filename[1: -5].split(" ") if len(coordinate) != 0]
            
            X = eyes_and_gradients
            y = coordinates
            
            if len(X) == 0:
                print("This image did not have a recognisable face, will pull a random one in its place")
                index = random.randint(0, self.__len__())
            else:
                got_good_image = True
        
        return X, y
    
    def __getitem__(self, batch):
        
        batch_X = [self.__load__(index)[0] for index in 
                   range((batch * self.batch_size), (batch + 1) * self.batch_size)]
        batch_y = [self.__load__(index)[1] for index in 
                   range((batch * self.batch_size), (batch + 1) * self.batch_size)]
        
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        
        return batch_X, batch_y

    def on_epoch_end(self):
        
        return