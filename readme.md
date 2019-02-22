# Transfer Learning - How to leverage pretrained model to speed up training for machine learning
Hide Inada

## Problem statement
Imagine you are a machine learning engineer working for a photo sharing website.
One morning, your manager comes to you and asks you to develop a feature that classifies food images into categories so that your company's website can show a restaurant nearby that serves food in that category of the photo on the page, and could raise an ad revenue from restaurants.

The problem is that your manager's manager really wants this feature yesterday and your manager was not able to push back so now he is asking you if there is anyway you can come up with something.
Aside from recommending to your manager that your manager and your manager's manager should take the Management 101 class, is there anything you can do to meet this crazy deadline?

You want to train at least tens of thousands of images which can take days.  Not only that you are not really sure about hyperparameter setting, so tuning that will multiply the time required for training.  On top of that, you are not really sure if your company has the powerful enough hardware to do this training.

Is there a way?

Yes, possibly.  Transfer training may work for you.

## Concept
It takes significant machine resources and time to train the model especially when there are so many layers.

However, there is a way to short circuit the training process.

This is called transfer learning.

This is based on the assumption that a deep neural network is trained to extract features in various layers.
If you go all the way up to the layer one before the output layer, feature extraction is already done.
and classification is mapping the various features with the final classification.
If you train a deep neural network with data that is similar to the data that you want to all you need to do is train the last output layer.

So what you need is:
1. Pretrained model (graph plus weights)
1. Some modification to train the last layer

For the first one, the good news is that Google has various pretrained model available for this under the name TensorFlow Hub, and this is available for anyone.

For the second one, they also made the script available.

So, pretty much what you need to do is just the two steps:

1) Set up the dataset on your file system
2) Run retrain.py

The whole training process can be done in 1 hour!

## Set up the dataset on your file system
The only thing you need to do is create a directory structure:

```
top image directory
--- image category label 1
------ jpeg files in that category
------ jpeg files in that category
------ jpeg files in that category
...

--- image category label 2
------ jpeg files in that category
------ jpeg files in that category
------ jpeg files in that category
...
```

For example,
```
food_images
--- burrito
------ burrito1.jpg
------ burrito2.jpg
...

--- sushi
------ sushi1.jpg
------ sushi2.jpg
...
```


## Training using retrain.py
```
python retrain.py --image_dir=food_images
```

That's it!



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

# What's happening in retrain.py
What retrain.py does is clever.
In addition to just train the last layer, it only calculates the front propagation once and uses the cached value.
This is a huge saving in terms of calculation.

Specifically, when it is first invoked, it uses image as an input and front prop all the way to the bottleneck layer.
Once the value of the bottleneck layer is calculated for the image, it writes the value to the disk.

After that, it uses the cached value as an input for the 1 layer network to optimize the matrix for the last layer.

# Code Walk-through of retrain.py
## License
```
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# NOTICE: This work was derived from tensorflow/examples/image_retraining
# and modified to use TensorFlow Hub modules.
```
