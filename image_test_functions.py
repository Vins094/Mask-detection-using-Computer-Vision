import os #load required libraries
import random
from skimage import io
from skimage.transform import resize
from skimage.feature import hog
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import sklearn
import matplotlib.pyplot as plt
from torchvision import models, transforms
import torch.nn.functional as F

#set device to be cpu
device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#function to take rake random 4 images

def FileNameList_test(path_sample,PATH_AFTER_,format):
  DIRECTORY_PATH_for_data = os.path.join(path_sample, PATH_AFTER_)
 
  all_files = os.listdir(DIRECTORY_PATH_for_data) #list all files_names from the folder
  file_names_list = [file for file
  in sorted(all_files) if file.endswith(format)] #list all files which ends with jpeg

  random_images_file_name = random.sample(file_names_list, min(5, len(file_names_list)))
  random_labels_file_name =[] #take random 4 filenames
  for image_name in random_images_file_name:
    random_labels_file_name.append(image_name.replace('jpeg', 'txt')) #get list of
    #labels with same filename by replacing extension to .txt

  return random_images_file_name,random_labels_file_name

#function fetch images from random 4 images
def Fetch_images_test(path_sample, DIRECTORY_PATH_Images,images_files_list):
  DIRECTORY_PATH_for_images = os.path.join(path_sample, DIRECTORY_PATH_Images)
  images_data = []
  for image_file in images_files_list:
      image = io.imread(os.path.join(DIRECTORY_PATH_for_images,image_file ))#read image
      images_data.append(image)#list images in image_data

  return images_data #return image_data list


#fetch labels

def Fetch_text_data_test(path_sample,DIRECTORY_PATH_labels,text_file_names_list):
  DIRECTORY_PATH_for_labels = os.path.join(path_sample, DIRECTORY_PATH_labels)
  labels = []
  for txt_file in text_file_names_list:
    file_path = os.path.join(DIRECTORY_PATH_for_labels, txt_file)
    with open(file_path, 'r') as F:
        file_content_txt_file = F.read() #read file 
        labels.append(file_content_txt_file) #add content in the list labels
  return labels

# preprocessing required for SVM and HOG(resize and HOG transformation)

def ResizeAndHOGTransformation(input_images,input_labels, height, width ):
  HOG_desctriptors =[]
  y_label =[]
  HOG_images =[]
  images = []
  for i in range(len(input_images)):
    image = input_images[i]
    resized_img = resize(image, (height, width)) #resize image with given size
    #perform HOG transformation
    HOG_des, HOG_image = hog(resized_img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=True, channel_axis=2)
                   
    if HOG_des is not None: #saving only those features and other data which has HOG features
      HOG_desctriptors.append(HOG_des)
      y_label.append(int(input_labels[i]))
      HOG_images.append(HOG_image)
      images.append(image)

  return images,y_label,HOG_desctriptors,HOG_images
  


#Load CNN mobileNetModel
def load_cnn_mobileNetV2_model(model_path):
  CNN_mobileNet= models.mobilenet_v2(weights=None)#load architecture without weights
  num_classes = 3  # Replace with the desired number of output classes
  num_features = CNN_mobileNet.classifier[1].in_features #get number of neurons in last layer
  CNN_mobileNet.classifier[1] = nn.Linear(num_features, num_classes) #replace number of neurons with 3(no. of classes)
  
  state_dict = torch.load(model_path)
  CNN_mobileNet.load_state_dict(state_dict)
  return CNN_mobileNet



#network architecture for CNN model
class mask_Net(nn.Module):
    def __init__(self):
        super(mask_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3,out_channels= 32,kernel_size= 3, padding =1) #first convolution layer
        self.pool = nn.MaxPool2d(kernel_size=2,stride= 2)# defined maxpooled layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding =1)#second convolution layer

        self.fc1 = nn.Linear(64*32*32, 64)  # input size of 1st fully connected layer
        self.fc2 = nn.Linear(64, 64) #2nd fully connected layer
        self.fc3 = nn.Linear(64, 3) #output layer with 3 (number of classes are three)

    def forward(self, x): #feed forward network defined
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor for using in fully connected network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net_test = mask_Net().to('cpu')#allocate to cpu

#load cnn model
def load_cnn_model(model_path):
  state_dict = torch.load(model_path) #load weights and biases in model architecture 
  net_test.load_state_dict(state_dict)#defined above
  return net_test #return cnn model


  




