# Transfer Learning - How to leverage pretrained model to speed up training for machine learning
Hide Inada

## Problem statement
You are a machine learning engineer working for a photo sharing website.
Your manager comes to you and ask you to classify a food image so that they can potentially show a restaurant nearby and raise an ad revenue.

## Concept
It takes significant machine resources and time to train the model especially when there are so many layers.

However, there is a way to short circuit the training process.

This is called transfer learning.

This is based on the assumption that :
Deep neural network is trained to extract features in various layers.
If you go all the way up to the layer one before the output layer, feature extraction is already done.
and classification is mapping the various features with the final classification.
If you train a deep neural network with data that is similar to the data that you want to all you need to do is train the last output layer.

## Dataset preparation
You can use any dataset that you want, but I used Food-101 dataset.  Please see the below page if you want to use this dataset: 
https://github.com/hideyukiinada/transfer-learning/blob/master/food101.md


## Set up
Clone this repo to download:
* retrain.py
* label_img.py

retrain.py and label_img.py were developed by the TensorFlow team and licensed under the terms listed at the top of each file.

If you want to check for the newer version, you can check hub and tensorflow repos:
```
git clone https://github.com/tensorflow/hub
diff hub/examples/image_training/retrain.py <path to this retrain.py>
```

```
git clone https://github.com/tensorflow/tensorflow
diff tensorflow/tensorflow/examples/label_image/label_image.py <path to this label_img.py>
```

food-101.tar.gz

```
tar zxvf food-101.tar.gz
```

This expands the files to:

food-101/images/<category name>/<file name>
 
 For example,
food-101/images/spaghetti_bolognese/3294753.jpg
 

Containing 101,000 images of 101 categories.
Each category contains:
training set: 750 
test set: 250


Dataset
[1] Lukas Bossard, Matthieu Guillaumin, Luc Van Gool. Food-101 – Mining Discriminative Components with Random Forests. https://www.vision.ee.ethz.ch/datasets_extra/food-101/
[2] Lukas Bossard1 Matthieu Guillaumin1 Luc Van Gool1. Food-101 – Mining Discriminative Components with Random Forests
https://www.vision.ee.ethz.ch/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf
