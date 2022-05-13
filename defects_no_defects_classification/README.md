# defects_no_defects_classification

## Introduction 
Defect detection plays a vital role in quality control and assurance process. During the production process, due to inefficiency of humans and machines, 100% quality on manufactured products cannot be guaranteed. This implies that care has to be taken in order to ensure only the best manufactured products are shipped. 

"Defect" can be defined as a deviation from the norm. Surface defect detections can be thought of as detection of scratches, color contamination, holes etc. Manual detection was considered the norm, but this method leads to low efficiency, the speed of detection is relatively low and it's also subject to human emotions.

## Motivation
Deep learning is a subfield of machine learning that works very well on unstructed data. In machine learning, features are engineered and passed into the architecture to produce a prediction. This task can become very repetitive and laborious. Deep learning takes care of the feature engineering aspect, it tries to learn the features inherent in the data. Surface defect is a common problem that manufacturing industries faced. The need to deliver high quality products are increasing on the daily and due to the vast amount of data being produced every second, we need a way to leverage this data. 


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
```python
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
gdown libary was used to unzip the processed file which is stored on my google drive

```python
import zipfile

!gdown https://drive.google.com/uc?id=1ODEDCLVdkCIdyT7otYkzJbbkebAp975P

# Unzip the downloaded file

zip_ref = zipfile.ZipFile('processed.zip')
zip_ref.extractall()
zip_ref.close()
```

Then the class labels were defined

```python
class_names = ['defects', 'no_defects']
```

The code below sets up the train and test directory and load them in batches

```python
train_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen = ImageDataGenerator(rescale=1/255.0)

train_dir = 'processed/train'
test_dir = 'processed/test'

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                               color_mode='rgb',
                                               shuffle=True,
                                                seed=42)

test_data = test_datagen.flow_from_directory(directory=test_dir,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                             color_mode='rgb',
                                               shuffle=True,
                                                seed=42)
                                               
```

We defined two tensorflow callbacks; the first callback logs the result to tensorboard and the second callback is for early stopping with a patience of 3. This means that the model would stop training when the validation loss does not reduce after 3 consecutive epochs

```python
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)
  print(f'Saving Tensorboard log files to {log_dir}')
  return tensorboard_callback
  
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

```

The model architecture is given below

```python
def create_model(model_url, num_classes=1):
  """
  Takes a TensorFlow Hub URL and creates a Keras Sequential model with it

  Parameters
  ----------
  model_url : str
    A TensorFlow Hub feature extraction URL.
  num_classes: int
    Number of output neurons in the output layer,
    should be equal to number of target classes, default 2

  Returns
  -------
  An uncompiled Keras Sequential model with model_url as feature extractor
  layer and Dense output layer with num_classes output neurons 
  """

  # Download the pretrained model
  feature_extraction_layer = hub.KerasLayer(model_url,
                                           trainable=False,
                                           input_shape=(224, 224, 3))
  
  model = tf.keras.Sequential([
    feature_extraction_layer,
    Dense(num_classes, activation='sigmoid')
  ])

  return model
  
model = create_model('https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1')

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
                
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(train_data, validation_data=test_data,
            epochs=50, steps_per_epoch=len(train_data),
            validation_steps=len(test_data),
            callbacks=[create_tensorboard_callback('tensorflow_hub', 'model_6_tl'), early_stopping])
```

In order to visualize the accuracy and loss curves to understand the performance of the model, a utility function was created

```python
def plot_loss_curves(history):
    """
    Plot loss curves and accuracy curves on separate figure object
    """
    epochs = list(range(1, len(history['loss']) + 1))
    loss = history['loss']
    val_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    
    plt.plot(epochs, loss, label='loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, accuracy, label='accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    
plot_loss_curves(history.history)
```

The function `make_prediction` takes a target folder and randomly selects an image and predicts the class label

```python
def make_prediction(target_class):
    target_folder = 'processed/test/' + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    img = load_img(target_folder + '/' + random_image[0])
    img = img_to_array(img)
    img = img / 255
    img_resize = tf.image.resize(img, size=[224, 224])
    img_expanded = tf.expand_dims(img_resize, axis=0)
    prediction = model_6.predict(img_expanded)
    plt.imshow(img[:,:,0], cmap='gray')
    plt.axis('off')
    plt.title(class_names[int(tf.round(prediction)[0])])
    print(img.shape)
   
make_prediction('no_defects')
```

## Technologies and Architecture used
- Python
- Tensorflow
- Convnet
- ElasticNet

## Conclusion
This projects shows how state-of-art deep learning techniques can be used to detect surface defects. 

## Future
In the future, I hope to perform classification with localization; not only would the model detect defects, it would also draw a bounding box on where the defect is localized

