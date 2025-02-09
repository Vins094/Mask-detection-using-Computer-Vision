import os #import required libraries
from skimage.io import imread
from skimage import io
#function to fetch names images and label text file
def FileNameList(PATH,format):
  # Get a list of all files in the directory
  all_files = os.listdir(PATH) #it will list all the files in directory
  file_names_list = [file for file in sorted(all_files) if file.endswith(format)] #save the filename
  #with particular format(.txt for label files and .jpeg for images)
  return file_names_list #return the list with images or text filename(depends on which format is passed)

#function to fetch text data (its label in our case)
def Fetch_text_data(PATH,text_file_names_list):
  labels = []
  for txt_file in text_file_names_list:
    file_path = os.path.join(PATH, txt_file) #file path
    with open(file_path, 'r') as F: 
        file_content = F.read() #to read text file which has labels
        labels.append(file_content) #append the labels in 'labels' list
  return labels #return list of labels ( which is out target variable)


#function to fetch images
def Fetch_images(PATH,images_files_list):
  images_data = []
  for image_file in images_files_list:
    image = io.imread(os.path.join(PATH,image_file )) #read the images
    images_data.append(image) #append the image to image_data list

  return images_data