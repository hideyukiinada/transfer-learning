# Transfer Learning - How to leverage pretrained model to speed up training for machine learning
<div style="text-align:right">By Hide Inada</div>

## 0. Problem statement
Imagine you are a machine learning engineer working for a photo sharing website.
One morning, your manager comes to you and asks you to develop a feature that classifies food images into categories so that your company's website can show a restaurant nearby that serves food in that category of the photo on the page, and could raise an ad revenue from restaurants.  Here is the mock up of the page that the manager showed you:

<img src="assets/images/burger.png" width="400px" align="middle">

The problem is that your manager's manager really wants this feature yesterday and your manager was not able to push back so now he is asking you if there is anyway you can come up with something.
Aside from recommending to your manager that your manager and your manager's manager should take the Management 101 class, is there anything you can do to meet this crazy deadline?

You want to train at least tens of thousands of images which can take days.  Not only that, you are not really sure about hyperparameter settings, so tuning can multiply the time required for training.  On top of that, you don't even know if your company has the powerful enough hardware to do this training.

Is there a way?

Yes, possibly.  Transfer training may work for you.

This article consists of three parts.  In the first part, I will first discuss the concept of transfer learning.  In the second part, I will go over the actual steps to train your model and predict using transfer learning.  In the third part, I will walk you through the inside of the script that TensorFlow team provided for transfer learning for you to be able to customize the script if needed.

## 1. Concept
It takes significant machine resources and time to train the model especially when there are so many layers.  For example, Inception-v3 has 42 layers [1].

However, there is a way to short circuit the training process.

This is called transfer learning.

Shown below is a conventional network architecture:

<img src="assets/images/conventional_net.png" width="400px" align="middle">

Your input front propagations layer by layer all the way to the output layer.
Once the loss is calculated, gradient of loss is probagated the other way going all the way to the first layer after the input recalculating weights for each layer.

This loop continues until the loss becomes reasonably small. Since there are a lot of calculating involved, the process can take days or even longer.

In transfer learning, instead of training the network scratch, you reuse the network that was already trained by someone.
In the below diagram, red dotted line shows the part that can be reused.

<img src="assets/images/transfer.png" width="420px" align="middle">



This is based on the assumption that a deep neural network is trained to extract features in various layers.
If you go all the way up to the layer one before the output layer, feature extraction is already done.
and classification is mapping the various features with the final classification.
If you train a deep neural network with data that is similar to the data that you want to all you need to do is train the last output layer.

So what exactly does train the last output layer mean?
If the bottleneck layer is a a plain-vanilla neural network layer (aka dense layer or a fully connected layer), then there is a matrix and a set of weights to be added as bias.  So you will be training these two unless you decide to add something extra.  In the diagram above, a green box with "Mat" indicates this matrix.  Bias is not shown in the diagram.

So in summary you need is:
1. A model with pretrained weights
1. A new layer with a matrix and bias to classify your images

For the first one, the good news is that TensorFlow team has made various pretrained model available for this on their website called TensorFlow Hub.

For the second one, they also made the script available to automate the creation and training of this new layer. This script also automatically downloads the pretrained weight to your computer, so pretty much what you need to do is just the two steps:

1) Set up the dataset on your file system
2) Run retrain.py

The whole training process can be done in 1 hour!

# 2. Detailed Steps

# 1. How to how to train your model and predict using transfer learning
## 1.1. Set up the dataset on your file system
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


## 1.2. Training using retrain.py
```
python retrain.py --image_dir=food_images
```

That's it!

## 1.1 Predict



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

# 3. What's happening in retrain.py
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

## 1. Check for command line arguments
The below code checks for command line arguments:
```

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  parser.add_argument(
      '--tfhub_module',
      type=str,
      default=(
          'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),
      help="""\
      Which TensorFlow Hub module to use. For more options,
      search https://tfhub.dev for image feature vector modules.\
      """)
  parser.add_argument(
      '--saved_model_dir',
      type=str,
      default='',
      help='Where to save the exported graph.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```
## 2. main()

Here are the main items that are done in main().

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
1.1.  If data augmentation is specified, read the image file from the file system, apply data augmentation, feed forward to the bottleneck layer (get_random_distorted_bottlenecks())
1.2.  If not, read the cached bottleneck layer values for each image from the file system (get_random_cached_bottlenecks) 
1.3.  Feed the bottleneck values and the ground-truth to the graph and optimize by gradienct descent as defined in add_final_retrain_ops
1.4.  For pre-determined interval, calculation training accuracy and validation accuracy
1.5.  For pre-determined interval, save graph and weights
1. Once the training is done, save weights
1. Predict against the test set to measure accuracy (run_final_eval())
1. Serialize the graph and save to the file system (save_graph_to_file())
1. If specified in command line, save the labels to the file system
1. If specified in command line, save the model without weights??? using tf.saved_model.simple_save (export_model())

# References
(1) Christian Szegedy et al. Rethinking the Inception Architecture for Computer Vision. https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf, 2016.
