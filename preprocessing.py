from skimage.transform import resize #import required libraries
from skimage.feature import hog
#partially adapted from the lab
#function resize the image and perform HOG transformation
def ResizeAndHOGTransformation(input_images,input_labels, height, width ):
  HOG_desctriptors =[] #list where all the HOG feature descriptors of images will be stored
  y_label =[] # list where all the labels of their respective feature descriptor will be stored
  HOG_images =[] #list where all the hog images will be stored
  images = [] #list of all raw images with their respective HOG descriptor
  for i in range(len(input_images)): #itterate images to perform resize and HOG transformation
    image = input_images[i]
    resized_img = resize(image, (height, width))
    HOG_des, HOG_image = hog(resized_img, orientations=8, pixels_per_cell=(16, 16), 
                    cells_per_block=(2, 2), visualize=True, channel_axis=2)

    if HOG_des is not None:
      HOG_desctriptors.append(HOG_des) #append all the values when HOG is not none ( sometime image does not)
      y_label.append(int(input_labels[i])) #provide HOG descriptor ( eg., plain image or blank image)
      HOG_images.append(HOG_image)
      images.append(image)

  return images,y_label,HOG_desctriptors,HOG_images #return the lists descriptors and labels will be
  #required to train the model

  #Note: different values of pixels_per_cell, cells_per_block were experimented before concluding to 
  #above values