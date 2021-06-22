# Dog Breed Classifier- A Capstone Project using CNN and Transfer Learning in PyTorch

This contains the dog breed classifier project of Udacity Machine Learning nanodegree.  The project is implemented using PyTorch.



[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"



## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!


## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	   
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
    
__NOTE:__ if you are using the Udacity workspace, you *DO NOT* need to re-download the datasets in steps 2 and 3 - they can be found in the `/data` folder as noted within the workspace Jupyter notebook.

### Import Datasets

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Make sure you have already installed the necessary Python packages according to the README in the program repository.
5. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```

## Built A Model From Scratch

I have built a three layers convolutional neural network model from scratch to help solve the problem statement.  This is the structure of the model:

Step 1: Convolutional Layer with 3 input dimensions and 32 output dimensions and kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

Relu Activation Function

Pooling layer: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

Relu Activation Function

step 2: Convolutional Layer with 32 input dimensions and 64 output dimensions and kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

Relu Activation Function

Pooling Layer: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

Step 3: Convolutional Layer with 32 input dimensions and 64 output dimensions and kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

Relu Activation Function

Pooling Layer: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

Flatten Layer to convert the pooled feature maps to a single vector

Fully Connected Layer: Linear(in_features=100352, out_features=512, bias=True)

Dropout with a probability of 0.25

Fully Connected Layer: Linear(in_features=512, out_features=133, bias=True)

Accuracy of the model: 10%-13% when trained at 20 - 30 Epochs

## Built Convolutional Neural Network Model using Transfer Learning

The model built from scratch was significantly improved by using transfer learning.  I have used the Resnet101 architecture which is pretrained on the ImageNet dataset.  The architecture produces good output.  The model performed well when compared to the CNN built from scratch.  It obtained an accuracy of 81% when trained between 40 to 50 epochs.



## Libraries
PyTorch, OpenCV, PIL, torchvision, numpy, glob, torch




