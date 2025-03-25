**Face Mask Detection using Computer Vision**
**Overview**

This project involves training machine learning and deep learning models for face mask detection. The models classify images into three categories:

Class 0: Not wearing a mask

Class 1: Wearing a mask

Class 2: Wearing a mask incorrectly

Additionally, a video was used to test the best-performing model, which detects faces and classifies mask usage in real time.

**Dataset**

The dataset consists of:

Images: 2,852 images, with 2,394 for training and 458 for testing.

Video: A short video showing a person demonstrating different mask-wearing techniques (total of 1,445 frames).
The dataset is imbalanced, with the following distribution in the training set:

Mask (Class 1): 1,940 images

No Mask (Class 0): 376 images

Incorrect Mask (Class 2): 78 images

**Implemented Methods**

**Pre-processing**

**For Traditional ML Models (SVM & MLP):**

Images resized to (128, 128, 3)

Applied Histogram of Oriented Gradients (HOG) for feature extraction

**For Deep Learning Models (CNN & MobileNetV2):**

Converted images to PIL format

Resized to appropriate dimensions:

CNN: (128, 128, 3)

MobileNetV2: (224, 224, 3)

Normalization:

CNN: [-1, 1]

MobileNetV2: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Model Training**

**SVM:** Grid search was used to find optimal hyperparameters (C, kernel type, and gamma values). Class weights were set to "balanced".

**MLP:** Hyperparameter tuning using grid search (hidden layer sizes, alpha values, learning rate, momentum).

**CNN:**

Two convolutional layers (32 and 64 filters) with max pooling

Three fully connected layers

Tuned hyperparameters (kernel, stride, weight decay, learning rate)

**MobileNetV2 (Transfer Learning):**

Used pre-trained MobileNetV2 (ImageNet)

Last layer replaced to classify 3 classes

Fine-tuned with a batch size of 32

**Video Processing**

Used OpenCV to extract frames from video

Applied MTCNN for face detection

Passed detected faces through MobileNetV2 to classify mask usage

Bounding boxes with labels were drawn on faces:

**Green:** Mask

**Purple:** Incorrect Mask

**Red:** No Mask

Animated the processed frames into a video output

**Results**

**Qualitative Results**

SVM: Performed well on "mask" images but struggled with side-angle images of "no mask" and "incorrect mask".

MLP: Incorrect classification of "no mask" and "incorrect mask".

CNN: Good performance for "mask" and "no mask" but struggled with "incorrect mask".

MobileNetV2: Best performance, correctly classifying all classes despite the imbalanced dataset.

**Quantitative Results**
---To be write--

**Video Results**

MobileNetV2 successfully classified all three classes in different orientations.

Some incorrect classifications occurred when the mask was worn on the chin (predicted as "no mask").

MTCNN showed bounding box inconsistencies in some frames.

**Conclusion**

MobileNetV2 was the best-performing model, with the highest accuracy and best generalization.

CNN was the second-best but struggled with "incorrect mask" classifications.

SVM & MLP had lower accuracy but trained significantly faster.

Trade-off exists between accuracy and computational efficiency, with MobileNetV2 being the most accurate but requiring the longest training tim

