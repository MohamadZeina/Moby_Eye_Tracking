# Imports
import os
import face_recognition
import cv2
import random
import time
import sys

import matplotlib.pyplot as plt 
import numpy as np

from scipy.misc import imresize

from tkinter import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import keras 
from keras.models import Model
from keras.layers import (Input, concatenate, Conv2D, Dense, MaxPool2D, 
                          Flatten, Dropout, SpatialDropout2D, GaussianNoise)


# Convenience functions
def small_dot(tkinter_canvas, centre_x, centre_y, radius=10, fill="red"):
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

def neural_model(dummy_sample, base_channels=8, dense_per_layer=50, conv_padding="same",
                 input_noise=0.0, pooling_dropout=0.0, dense_dropout=0.0, spatial_dropout=0.0):
    
    print("About to initialise a neural network with input shape: ", dummy_sample.shape)
    
    l2_reg = keras.regularizers.l2(0.0001)

    visible = Input(shape=(dummy_sample.shape))
    if input_noise: 
        input_augmented = GaussianNoise(input_noise)(visible)
    else:
        input_augmented = visible

    c11 = Conv2D(base_channels, 3, padding=conv_padding)(input_augmented)
    if spatial_dropout: c11 = SpatialDropout2D(spatial_dropout)(c11)
    c12 = Conv2D(base_channels, 3, padding=conv_padding)(c11)
    if spatial_dropout: c12 = SpatialDropout2D(spatial_dropout)(c12)
    p1 = Conv2D(base_channels * 2, 1, strides=2)(c12)
    if pooling_dropout: p1 = Dropout(pooling_dropout)(p1)

    c21 = Conv2D(base_channels * 2, 3, padding=conv_padding)(p1)
    if spatial_dropout: c21 = SpatialDropout2D(spatial_dropout)(c21)
    c22 = Conv2D(base_channels * 2, 3, padding=conv_padding)(c21)
    if spatial_dropout: c22 = SpatialDropout2D(spatial_dropout)(c22)
    p2 = Conv2D(base_channels * 4, 1, strides=2)(c22)
    if pooling_dropout: p2 = Dropout(pooling_dropout)(p2)

    c31 = Conv2D(8, 3, padding=conv_padding)(p2)
    c32 = Conv2D(8, 3, padding=conv_padding)(c31)
    p3 = Conv2D(16, 1, strides=2)(c32)
    if pooling_dropout:
        p3 = Dropout(pooling_dropout)(p3)
    
    f1 = Flatten()(p3)
    d1 = Dense(dense_per_layer, activation="relu", kernel_regularizer=l2_reg)(f1)
    if dense_dropout:
        d1 = Dropout(dense_dropout)(d1)

    d2 = Dense(dense_per_layer, activation="relu", kernel_regularizer=l2_reg)(d1)
    output = Dense(2)(d2)
    
    model = Model(inputs=visible, outputs=output)
    
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer="adam")
    
    return model

def extract_facial_features(frame, downsample=0.5, get_gradients=True, display=False):
    
    # Basic code for facial landmark extraction from webcam from:
    # https://elbruno.com/2019/05/29/vscode-lets-do-some-facerecognition-with-20-lines-in-python-3-n/    
    try:
        rgb_frame = frame[:, :, ::-1].copy()
        rgb_frame_copy = frame[:, :, ::-1].copy()
        if downsample:
            rgb_frame = imresize(rgb_frame, downsample)
    except TypeError:
        print("Problem extracting data from frame.")
        return [], [], [], []

    frame_copy = frame.copy()
    bw_frame = np.mean(rgb_frame_copy, axis=2)

    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    
    # Extract region around eyes, before green lines added. Uses face_recognition
    border_height = 10
    border_width = 15
    
    # Creat linear gradients to bundle with the eye data
    if get_gradients:
        grad_x = np.zeros(frame_copy.shape[:2], dtype=np.float)
        grad_y = np.zeros(frame_copy.shape[:2], dtype=np.float)
    
        for i in range(frame_copy.shape[0]):
            grad_x[i, :] = i / (frame_copy.shape[0])
        
        for j in range(frame_copy.shape[1]):
            grad_y[:, j] = j / (frame_copy.shape[1])
    
    try:
        # Locate the left eye
        left_eye = np.mean(np.array(face_landmarks_list[0]["left_eye"]), axis=0, dtype=int)
        if downsample: left_eye = np.array(left_eye / downsample, int)

        left_eye_region = bw_frame[left_eye[1] - border_height: left_eye[1] + border_height,
                                   left_eye[0] - border_width: left_eye[0] + border_width]

        if get_gradients:
            left_eye_x_grad = grad_x[left_eye[1] - border_height: left_eye[1] + border_height,
                                     left_eye[0] - border_width: left_eye[0] + border_width]
            left_eye_y_grad = grad_y[left_eye[1] - border_height: left_eye[1] + border_height,
                                     left_eye[0] - border_width: left_eye[0] + border_width]
            #print("mean number from left eye x gradient is: ", np.mean(left_eye_x_grad))
            #print("mean number from left eye y gradient is: ", np.mean(left_eye_y_grad))
            #print("left eye centre is at: ", left_eye)
        
        left_eye_flattened = left_eye_region.reshape(1,-1)[0]
    
        # Locate the right eye
        right_eye = np.mean(np.array(face_landmarks_list[0]["right_eye"]), axis=0, dtype=int)
        if downsample: right_eye = np.array(right_eye / downsample, int)

        right_eye_region = bw_frame[right_eye[1] - border_height: right_eye[1] + border_height,
                                    right_eye[0] - border_width: right_eye[0] + border_width]

        if get_gradients:
            right_eye_x_grad = grad_x[right_eye[1] - border_height: right_eye[1] + border_height,
                                      right_eye[0] - border_width: right_eye[0] + border_width]
            right_eye_y_grad = grad_y[right_eye[1] - border_height: right_eye[1] + border_height,
                                      right_eye[0] - border_width: right_eye[0] + border_width]
            #print("mean number from right eye x gradient is: ", np.mean(right_eye_x_grad))
            #print("mean number from right eye y gradient is: ", np.mean(right_eye_y_grad))
            #print("right centre is at: ", right_eye)

        if not get_gradients:
            left_eye_x_grad = np.zeros(left_eye_region.shape[:2])
            left_eye_y_grad = np.zeros(left_eye_region.shape[:2])

            right_eye_x_grad = np.zeros(right_eye_region.shape[:2])
            right_eye_y_grad = np.zeros(right_eye_region.shape[:2])
        
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
    
    return rgb_frame, rgb_frame_copy, everything_array, landmark_array, eyes_and_gradients

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
    
    path = "data/MZeina_6/"
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

def train_and_preview(pretrained_model=None):
    ########## Universal Initialisation ##########
    counter = 0
    captures_per_point = 5
    
    ########## Initialise Video Stream ##########
    #video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(0)#, cv2.CAP_DSHOW)
    #video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
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

class InteractiveTrainer():

    def __init__(self, randomise_dot=True, move_smoothly=False):
        # Arguments to class variables
        self.randomise_dot = randomise_dot
        self.move_smoothly = move_smoothly

        print("Initialised interactive trainer object")

    def predict_gaze(self):
    
        ret, frame = self.video_capture.read()
        (self.rgb_frame_downsampled, self.rgb_frame, self.everything_array, 
        self.landmark_array, self.eyes_and_gradients) = extract_facial_features(frame)
        
        try:
            if self.model_type == "neural net":
                X = np.expand_dims(self.eyes_and_gradients, 0)
                self.predicted_gaze = self.model.predict(X)[0]
            else:
                self.predicted_gaze = self.model.predict(self.everything_array)[0]
        
            print("Predicted gaze is: ", self.predicted_gaze)
        except ValueError:
            print("Could not predict, probably no face in image")
            self.predicted_gaze = np.array([0., 0.])
        
        # Scale the prediction to webcam resolution
        predicted_pixel = [self.predicted_gaze[0] * self.tk_width, self.predicted_gaze[1] * self.tk_height]
        # print(predicted_pixel, predicted_gaze, webcam_resolution)
        
        # Display the prediction as a grey circle
        small_dot(self.canvas, predicted_pixel[0], predicted_pixel[1], radius=5, fill="grey")
    
        return


    def capture(self):
        """Will capture an image + coordinate pair when the user is looking at the dot"""
    
        path = "data/MZeina_6/" # to do: Move to class variable
        train_every = 1
            
        # print("About to learn...")
        if len(self.landmark_array) != 0:
            self.current_target = np.array(self.current_target) / np.array([self.tk_width, self.tk_height])
            
            if self.model_type == "neural net":
                # Neural network can train on each sample at a time, unlike random forest
                self.training_X = np.expand_dims(self.eyes_and_gradients, 0)
                self.training_y = np.expand_dims(self.current_target, 0)
                # training_X.append(eyes_and_gradients)
            else:
                self.training_X.append(self.landmark_array[0])
                self.training_y.append(self.current_target)
            
            plt.imsave(path + str(self.current_target) + ".jpg", self.rgb_frame)
            
            if self.counter % train_every == 0:
                self.model.fit(self.training_X, self.training_y)
            
        else:
            print("Face not detected, will not train on this sample")
    
        #self.canvas.delete("all")
        if self.move_smoothly:
            speed = 20
            scaled_counter = (self.counter * speed) % (self.tk_width * self.tk_height)
            target_x = (scaled_counter // self.tk_height * speed) % self.tk_width
            if (scaled_counter // self.tk_height)%2 == 0:
                target_y = scaled_counter % self.tk_height
            else:
                # reverse the direction for alternative lines, so it doesn't skip up to the top
                target_y = self.tk_height - scaled_counter % self.tk_height
            print("counter, scaled_counter, are :", self.counter, scaled_counter)
            print("about to move small circle to", target_x, target_y)
            small_dot(self.canvas, target_x, target_y)
            self.current_target = [target_x, target_y]
        elif self.randomise_dot:
            self.current_target = random_dot(self.canvas, self.tk_width, self.tk_height)
        # print(random_width, random_height)
        
        return self.model, self.current_target


    def train(self, pretrained_model=None):
        ########## Universal Initialisation ##########
        self.counter = 0
        captures_per_point = 5
        
        ########## Initialise Video Stream ##########
        #video_capture = cv2.VideoCapture(0)
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Extract webcam resolution
        ret, frame = self.video_capture.read()
        self.webcam_resolution = frame.shape[:2]
        # print(webcam_resolution) 
        
        ########## Initialise ML Model ##########
        
        # Dummy sample, to help initialising models
        (self.rgb_frame_downsampled, self.rgb_frame, self.everything_array, 
        self.landmark_array, self.eyes_and_gradients) = extract_facial_features(frame)
    
        self.model_type = "neural net" # to do: can be moved to init method
        
        if pretrained_model:
            self.model = pretrained_model
        elif self.model_type == "random forest":
            # Random forest 
            RF = RandomForestRegressor(n_estimators=500, n_jobs=-1, warm_start=False)
            self.model = MultiOutputRegressor(RF)
            self.model.fit(np.zeros_like(self.dummy_features), np.array([0.5, 0.5]).reshape(1, -1))
        elif self.model_type == "neural net":
            self.model = neural_model(self.eyes_and_gradients)
            self.model.summary()
            
        # To do:Train on existing pictures
        
        # Initialise
        self.training_X = []
        self.training_y = []
        
        ########## Initialise Tkinter ##########
        window = Tk()
        window.attributes("-fullscreen", True)
        
        window.update_idletasks() 
        self.tk_width = window.winfo_width() 
        self.tk_height = window.winfo_height()

        self.canvas = Canvas(window, width = self.tk_width, height = self.tk_height)
        self.canvas.pack()
        
        window.bind("<F11>", lambda event: window.attributes("-fullscreen",
                                            not window.attributes("-fullscreen")))
        window.bind("<Escape>", lambda event: window.attributes("-fullscreen", False))
        # window.bind("c", lambda event: capture(canvas, RFMO, tk_width, tk_height, video_capture, webcam_resolution, landmark_array, current_target, predicted_gaze))
        
        # Variables to store red dot target
        self.current_target = random_dot(self.canvas, self.tk_width, self.tk_height)
        
        while True:
            
            self.predict_gaze()
            
            if self.counter % 6 == 0 and self.counter != 0:
                self.canvas.delete("all")
                
                self.capture()
                    
            self.counter += 1
            
            # Update GUI
            window.update_idletasks()
            window.update()
        return

     
class ScreenshotGenerator(keras.utils.Sequence):
    
    def __init__(self, paths_to_images, batch_size=4, mirror_augment_all=False):
        
        self.paths_to_images = paths_to_images
        self.batch_size = batch_size
        self.mirror_augment_all = mirror_augment_all

        if mirror_augment_all and batch_size % 2 != 0:
            print("When using mirror augmentation, batch size must be an even number")
            assert False

        self.files = []
        self.filenames = []
        
        for path_to_images in paths_to_images:
            for root, dirs, files in os.walk(path_to_images):
                for name in files:
                    if name.endswith(".jpg"):
                        self.files.append(os.path.join(root, name))
                        self.filenames.append(name)
        
        if self.mirror_augment_all:
            self.files = [self.files[i//2] for i in range(len(self.files)*2)]
            self.filenames = [self.filenames[i//2] for i in range(len(self.filenames)*2)]

    def __len__(self):
        
        return len(self.files) // self.batch_size
    
    def __load__(self, index):
        """Returns and processes a single sample, in conjunction with __getitem__"""
        
        time_image_requested = time.time()
        # Ensures that if an image is picked without a succesfully detected face, 
        #  it looks for another random one to replace it
        got_good_image = False

        if index % 2 != 0 and self.mirror_augment_all:
        
            [left_eye_region, 
             left_eye_x_grad, 
             left_eye_y_grad,
             right_eye_region, 
             right_eye_x_grad, 
             right_eye_y_grad] = [self.X[:, :, i] for i in range(6)]

            # Mirror the eye gradients
            left_eye_y_grad, right_eye_y_grad = right_eye_y_grad, left_eye_y_grad
            left_eye_x_grad, right_eye_x_grad = right_eye_x_grad, left_eye_x_grad # Vertical, should not be mirrored

            left_eye_y_grad = -(left_eye_y_grad - 0.5) + 0.5
            right_eye_y_grad = -(right_eye_y_grad - 0.5) + 0.5

            left_eye_y_grad = np.flip(left_eye_y_grad, axis=1)
            right_eye_y_grad = np.flip(right_eye_y_grad, axis=1)

            # Mirror the eyes themselves
            left_eye_region, right_eye_region = right_eye_region, left_eye_region

            left_eye_region = np.flip(left_eye_region, axis=1)
            right_eye_region = np.flip(right_eye_region, axis=1)

            # Mirror the width component of the target
            self.mirrored_y = self.y.copy()
            self.mirrored_y[0] = -(self.y[0] - 0.5) + 0.5

            self.mirrored_X = np.stack((left_eye_region, left_eye_x_grad, left_eye_y_grad,
                               right_eye_region, right_eye_x_grad, right_eye_y_grad), axis=2)

            return self.mirrored_X, self.mirrored_y
        
        while not got_good_image:
        
            file = self.files[index]
            filename = self.filenames[index]
                        
            image = cv2.imread(file)
            
            rgb_frame, everything_array, landmark_array, eyes_and_gradients = extract_facial_features(image, True)
            #print("Extracted features in: ", time.time() - time_image_requested)

            coordinates = [float(coordinate) for coordinate in filename[1: -5].split(" ") if len(coordinate) != 0]
            
            self.X = eyes_and_gradients
            self.y = coordinates
            
            if len(self.X) == 0:
                print("This image did not have a recognisable face, will pull a random one in its place")
                index = random.randint(0, self.__len__())
            else:
                got_good_image = True
        
        return self.X, self.y
    
    def __getitem__(self, batch):

        time_batch_requested = time.time()

        batch = [self.__load__(index) for index in
                 range((batch * self.batch_size), (batch + 1) * self.batch_size)]

        batch_X = [data_point[0] for data_point in batch]
        batch_y = [data_point[1] for data_point in batch]

        
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        
        time_to_get_batch = time.time() - time_batch_requested
        #print("Got batch in: ", time_to_get_batch)

        return batch_X, batch_y

    def on_epoch_end(self):
        
        return