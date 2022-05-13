# defects_no_defects_classification

## Introduction 
Defect detection plays a vital role in quality control and assurance process. During the production process, due to inefficiency of humans and machines, 100% quality on manufactued products cannot be guaranteed. This implies that care has to be taken in order to ensure only the best manufactured are shipped. "Defect" can be defined as a deviation from the norm. Surface defect detections can be thought of as detection of scratches, color contamination, holes etc. Manual detection was considered the norm, but this method leads to low efficiency, the speed of detection is relatively low and it's also subject to human emotions.

## Background Study

## Aims and Objectives
Create a convnet (Convolutional Neural Network) that separates defect surface materials from non defects

***Here is an example of the model in action***

![](https://github.com/samie-hash/data-science-repo/blob/main/defects_no_defects_classification/defect_2.png)

The above image is classified as defect

![](https://github.com/samie-hash/data-science-repo/blob/main/defects_no_defects_classification/no_defect.png)

The above image is classified as no_defect

The model fails in the following images, this may be due to the black spot at the image center

![](https://github.com/samie-hash/data-science-repo/blob/main/defects_no_defects_classification/no_defect_missed.png)
![](https://github.com/samie-hash/data-science-repo/blob/main/defects_no_defects_classification/defect_1.png)

## Data Source
The data set was gotten from this [link](https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html) and preprocessed using this utility [notebook](https://github.com/samie-hash/data-science-repo/blob/main/defects_no_defects_classification/helper.ipynb).

## Procedure
The provided data is artificially generated, but similar to real world problems. The orignal data was modified and the majority class was downsampled in order to make the model more robust in detecting defects on surface materials. It consists of multiple data sets, each consisting of 300 images showing the background texture without defects, and of 150 images with one labeled defect each on the background texture. The data set is also weakly labeled and as such, they are not pixel precise. 

To solve this problem, state-of-the-art [elasticnet]('https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1') model is leveraged and a convolutional net is build on top of it.
 
First the libraries that would be used were imported
```
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

Then the class labels were defined
```
class_names = ['defects', 'no_defects']
```


## Technologies and Architecture used
- Python
- Tensorflow
- Convnet
- ElasticNet

## Conclusion
