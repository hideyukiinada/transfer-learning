# Transfer Learning - How to leverage pretrained model to speed up training for machine learning
<div style="text-align:right">By Hide Inada</div>

## Problem statement
Imagine you are a machine learning engineer working for a photo sharing website.
One morning, your manager comes to you and asks you to develop a feature that classifies food images into categories so that your company's website can show a restaurant nearby that serves food in that category of the photo on the page, and could raise an ad revenue from restaurants.  Here is the mock up of the page that the manager showed you:

<img src="assets/images/burger.png" width="400px" align="middle">

In this example, your task is to classify the photo as a burger so that a local burger joint's ad is displayed.  Since you are not a front-end engineer, as long as you return the burger class, presentation will be taken care of by another team. 

The problem is that your manager's manager is new to the company. He does not really know much about machine learning, but really wants this feature now to impress his new boss and your manager was not able to push back.  So now he is asking you if there is anyway you can come up with something.
Aside from recommending to your manager that your manager and your manager's manager should take the Management 101 class, is there anything you can do to meet this crazy deadline?

You want to train at least tens of thousands of images which can take days.  Not only that, you are not really sure about hyperparameter settings, so tuning can multiply the time required for training.  On top of that, you don't even know if your company has the powerful enough hardware to do this training fast enough.  You remember that when you asked for NVIDIA DGX-1 and told your manager a price, your manager was so shocked and burst out coffee all over your desk.

Is there a way to do this?

Yes, possibly.  Transfer training may work for you.

<hr>

This article consists of three parts.  In the first part, I will discuss the concept of transfer learning.  In the second part, I will go over the actual steps to train your model and predict using transfer learning.  In the third part, I will walk you through the inside of the script that TensorFlow team provided for transfer learning for you to be able to customize the script if needed.

## 1. Concept
It takes significant machine resources and time to train the model especially when there are so many layers.  For example, Inception-v3 has 42 layers [1].  However, there is a way to short circuit the training process. This is called transfer learning.
Shown below is a conventional network architecture:

<img src="assets/images/conventional_net.png" width="400px" align="middle">

Your input front propagations layer by layer all the way to the output layer.
Once the loss is calculated, gradient of loss is probagated all the way back to the first layer after the input recalculating weights for each layer.

This is done in a loop, and iterations continue until the loss becomes reasonably small. Since there are a lot of calculations involved, the process can take days or even longer.

In transfer learning, instead of training the network scratch, you reuse the network that was already trained, which is most likely by someone else.
In the below diagram, red dotted line shows the part that can be reused.

<img src="assets/images/transfer.png" width="420px" align="middle">

This works based on the assumption that a deep neural network is trained to extract features in various layers.
If you go all the way up to the layer one before the output layer, which is called bottleneck layer, feature extraction is already done and that last layer's responsibility is to map extracted features in the bottleneck layer to a set of classes for your images.

So what exactly does train the last output layer mean?
If the bottleneck layer is a a plain-vanilla neural network layer (aka dense layer or a fully connected layer), then there is a matrix and a set of weights to be added as bias.  So you will be training these two unless you decide to add something extra.  In the diagram above, a green box with "Mat" indicates this matrix.  Bias is not shown in the diagram.

So in summary what you need is:
1. A model with pretrained weights
1. A new layer with a matrix and bias to classify your images

For the first one, the good news is that TensorFlow team has made various pretrained models available for this on their website called TensorFlow Hub.

For the second one, they also made the script called "retrain.py" available to automate the creation and training of this new layer. This script also automatically downloads the pretrained weight to your computer, so pretty much what you need to do is just the two steps:

1) Set up the dataset on your file system
2) Run retrain.py

The whole training process can be done in 1 hour!

# 2. Detailed Steps
In this section, I will go over details of the two high-level steps.

## 1.1. Set up the dataset on your file system
retrain.py expects that image data is stored in a two-level directory structure:

```
top image directory
--- image category label 1
------ jpeg files in that category 1
------ jpeg file in that category 1
------ jpeg file in that category 1
...

--- image category label 2
------ jpeg file in that category 2
------ jpeg file in that category 2
------ jpeg file in that category 2
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

Each image directory name is used as the class label of the images.
retrain.py automatically split validation and test set images from training set, so there is no need for you to separate images if you want to avoid extra work.

You can use any dataset that you want, but I used Food-101 dataset.  Please see the below page if you want to use this dataset: https://github.com/hideyukiinada/transfer-learning/blob/master/food101.md

## 2.2. Training using retrain.py
Once the images were laid out, you can clone this repo to download:
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

Once you have retrain.py on your local disk, run it with the name of the top-level image directory.
For example, if your images are located under food_images, type: 

```
python retrain.py --image_dir=food_images
```

If you clone this repo, you can also use my wrapper called train.bash
```
#!/bin/bash

# Replace the directory name after the --image_dir

export TFHUB_CACHE_DIR=/tmp/food101/module_cache

time python retrain.py --image_dir=../../../dataset/food101/food-101/images

```

Just edit the image_dir path and run it.

This starts the training session.

When the script is completed, verify the output in the following directories:

* /tmp/output_labels.txt
* /tmp/output_graph.pb

output_labels.txt contains the classes of your images which were taken from each directory.
output_graph.pb is the new model file with trained weight in the protobuf format.  You will be using this file for prediction in the next step.

## 2.3. Predict

You can use predict.bash if you cloned this repo. It's a wrapper that calls label_image.py with the set of parameters:
```
#!/bin/bash

python label_image.py \
--graph=/tmp/output_graph.pb \
--labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=$1 2>/dev/null
```

For example, if you want to predict the class for the image, burger.jpg, type:

```
./predict.bash burger.jpg
```

Here is the actual result against the image at the top of this article:

<img src="assets/images/burger_prediction.jpg" width="1000px" align="middle">

# 3. Inside retrain.py
## 3.1. Terminology 
TensorFlow team uses the word "module" to mean a model with pretrained weights.  In this section, I will be following that convention.

## 3.2. Overview of high-level items
Here are the main items that are done in code.
1. Check for command line arguments
1. Clean up TensorBoard log directory and ensure it exists (prepare_file_system())
1. Ensure a directory to store an intermediate graph exists (prepare_file_system())
1. Read the image directory and subdirectories, get the list of files in each subdirectory and split the file into training, validation, test set per the ratio specified in command line arguments (create_image_lists()).
1. Determine any command line argument is specified for data augmentation (should_distort_images())
1. Load module spec of the module that you want to instantiate (hub.load_module_spec())
1. Load the module, and get the last layer of the module (create_module_graph())
1. Add the output layer for classification of our data (add_final_retrain_ops())
1. Add operations to resize the JPEG data to the size that the module expects (add_jpeg_decoding())
1. If any data augmentation option is specified, crop, flip horizontally and/or adjust brightness of the image (add_input_distortions)
1. If data augmentation option was not specified, front propagate the image data through the network all the way up to the bottleneck layer and writes the values to the file system (cache_bottlenecks)
1. Add operations to calculate accuracy by comparing predictions and ground-truth and taking the mean (add_evaluation_step())
1. Consolidate stats that you want to show in TensorBoard and direct the stats log output to the file system by instantiating FileWriter objects
1. Instantiate tf.train.Saver to prepare for saving weights during training
1. Train by repeat the following steps
    1. If data augmentation is specified, read the image file from the file system, apply data augmentation, feed forward to the bottleneck layer (get_random_distorted_bottlenecks())
    2. If not, read the cached bottleneck layer values for each image from the file system (get_random_cached_bottlenecks) 
    3. Feed the bottleneck values and the ground-truth to the graph and optimize by gradienct descent as defined in add_final_retrain_ops
    4. For pre-determined interval, calculation training accuracy and validation accuracy
    5. For pre-determined interval, save graph and weights
1. Once the training is done, save weights
1. Predict against the test set to measure accuracy (run_final_eval())
1. Serialize the graph and save to the file system (save_graph_to_file())
1. If specified in command line, save the labels to the file system
1. If specified in command line, save the model without weights??? using tf.saved_model.simple_save (export_model())

What retrain.py does is clever.
In addition to just train the last layer, it only calculates the front propagation once and uses the cached value.
This is a huge saving in terms of calculation.

Specifically, when it is first invoked, it uses image as an input and front prop all the way to the bottleneck layer.
Once the value of the bottleneck layer is calculated for the image, it writes the value to the disk.

After that, it uses the cached value as an input for the 1 layer network to optimize the matrix for the last layer.

# References
&#91;1&#93; Christian Szegedy et al. Rethinking the Inception Architecture for Computer Vision. https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf, 2016.
&#91;2&#93; TensorFlow team. How to Retrain an Image Classifier for New Categories. https://www.tensorflow.org/hub/tutorials/image_retraining.

