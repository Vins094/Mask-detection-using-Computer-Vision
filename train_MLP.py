
#Import required libraries
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import joblib
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
import time
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score

#mount the drive
from google.colab import drive
drive.mount('/content/drive')

import os

# Google Drive path where coursework material is available
GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'My_Computer_vision/CV_Coursework/CW_Folder_PG_template/CW_Folder_PG'
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
print(os.listdir(GOOGLE_DRIVE_PATH))


#Magic comand to load for automatic loading of modules
# %load_ext autoreload
# %autoreload 2

#import modules located in code directory
import sys
CODE_PATH = os.path.join(GOOGLE_DRIVE_PATH, 'Code')
sys.path.append(CODE_PATH)

#extracted data from zipped file to copy in Colab
zip_path = os.path.join(GOOGLE_DRIVE_PATH, 'CW_Dataset/CV2024_CW_Dataset.zip')

  # Copy it to Colab
!cp '{zip_path}' .
  # Unzip it
!yes|unzip -q CV2024_CW_Dataset.zip

# Delete zipped version from Colab (not from Drive)
!rm CV2024_CW_Dataset.zip

#import required functions from Load_data python script to load data
from Load_data import FileNameList,Fetch_text_data, Fetch_images
#FileNameList is the function to collect all the filnames (images and labels)
#Fetch_text_data is a funtion to import all label names from labels folder into the list
#Fetch_images is a function to import all the images from images folder

test_label_data = Fetch_text_data('test/labels',FileNameList('test/labels','.txt')) #import test labels
train_label_data = Fetch_text_data('train/labels',FileNameList('train/labels','.txt')) # import train labels
test_images_data = Fetch_images('test/images',FileNameList('test/images', '.jpeg')) #import test images
train_images_data = Fetch_images('train/images',FileNameList('train/images', '.jpeg')) #import train images

print(f'Train labels distribution: {Counter(train_label_data)}') #data is imbalance
print(f'Test labels distribution: {Counter(test_label_data)}') #data is imbalance

#import required functions from the preprocessing python script
from preprocessing import ResizeAndHOGTransformation
#ResizeAndHOGTransformation is a function to resize the images and perform HOG transformation of to get the required features of the images which
#later will be used to train the model

train_images,y_label_train, HOG_desctriptors_train, HOG_images_train = ResizeAndHOGTransformation(train_images_data,train_label_data,128,128)

#balacing data using oversampling technique
# x_resampled_imageHOG, y_resampled_label = SMOTE().fit_resample(HOG_desctriptors_train, y_label_train)

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

#define list of hyperparameters to be used in Grid search
param_grid = {
        'hidden_layer_sizes': [(100,), (100, 150), (100, 150, 200)], #hidde layers
        'alpha': [0.0001, 0.001, 0.01], #weight decay
        'learning_rate_init': [0.3, 0.03, 0.003, 0.0003], #learning rate
        'momentum': [0.9, 0.95], #momentum

    }

mlp = MLPClassifier(max_iter=100,verbose=True, random_state=4) #create object of mlp classifier

from sklearn.model_selection import GridSearchCV #import gridsearch

grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1) #create object of grid search

# grid_search.fit(x_resampled_imageHOG, y_resampled_label) #initially considered  balanced data

grid_search.fit(HOG_desctriptors_train, y_label_train) #final grid search on unbalanced data

grid_search.best_estimator_ #find final hyperparameters

grid_search.best_params_

# create final Multi-Layer Perceptron model using hyper parameters obtaiined in Grid search
mlp_model = MLPClassifier(hidden_layer_sizes=(100), max_iter=100, alpha=0.01,
                    solver='adam', verbose=True, random_state=1,
                    learning_rate_init=.0003, momentum=0.9)
start_time = time.time()
mlp_model.fit(HOG_desctriptors_train, y_label_train) # train the model
end_time = time.time()
elapsed_time = end_time- start_time
print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s") # monitor time required to train the model

#HOG transformation on test data to get their HOG descriptors
test_images, y_label_test, HOG_desctriptors_test, HOG_images_test = ResizeAndHOGTransformation(test_images_data,test_label_data,128,128)

start_time = time.time()
y_pred = mlp_model.predict(HOG_desctriptors_test) # prediction on test data
end_time = time.time()
elapsed_time = end_time- start_time
print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

#display images with their true and predicted labels
#code partially adapted from the Lab
images, label, y = shuffle(test_images, y_label_test, y_pred)
fig, axes = plt.subplots(2, 5, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()
dict = {1: 'Mask', 2: 'Incorrect Mask', 0: 'No Mask'}


for i in range(10):
    ax[i].imshow(images[i])
    ax[i].set_title(f'Label: {dict[label[i]]} \n Prediction: {dict[y[i]]}', fontsize =8)
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()

#print the performance matrics model

print(f"""Classification report for classifier {mlp_model}:
      {metrics.classification_report(y_label_test, y_pred)}\n""")

#confusion matrix to compare true and predicted output
metrics.ConfusionMatrixDisplay.from_predictions(y_label_test, y_pred)
plt.show()

#Performance metrics
print(f'Accuracy: {accuracy_score(y_label_test, y_pred)}')
print(f'F1 score: {f1_score(y_label_test, y_pred, average="weighted")}')
print(f'recall: {recall_score(y_label_test, y_pred,average="weighted")}')
print(f'precision:{precision_score(y_label_test, y_pred,average="weighted")}')

#Save best model using joblib
#joblib.dump(mlp_model, 'drive/My Drive/My_Computer_vision/CV_Coursework/CW_Folder_PG_template/CW_Folder_PG/Models/MLP_hog_final')