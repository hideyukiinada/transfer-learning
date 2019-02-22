# Preparing Food-101 Dataset for Transfer Learning
Hide Inada

## Overview
In this document, steps to prepare Food-101 Dataset for transfer learning will be discussed.
Key steps are the following:

1.  Download the dataset
2.  Split the dataset to training, validation and test set

## Download the dataset
The dataset is available at:
https://www.vision.ee.ethz.ch/datasets_extra/food-101/

Head over to the page, and download the dataset to your local disk.

## Split the dataset to training set, validation set and test set
After downloading the dataset and expanding the gzipped tar file, you will see that all the images are located under the images directory.
A script that you will be using for transfer learning expects images for each dataset to be stored in separate directories, so you need to split the dataset into training, validation and test set.

meta/test.txt file contains the test set, so you need to move the files in this list to a separate directory.
I wrote a script below to automate this:

https://github.com/hideyukiinada/transfer-learning/blob/master/tools/food-101-split

After running the script, the dataset is split into:

| Type | Number of samples | Directory name |
|---|---|---|
| Training set | 75750 | images |
| Validation set | 10100 | validation_images |
| Test set | 15150 | test_images |
| Total | 101000 | |

# References
&#91;1&#93; Lukas Bossard, Matthieu Guillaumin, Luc Van Gool. Food-101 – Mining Discriminative Components with Random Forests. https://www.vision.ee.ethz.ch/datasets_extra/food-101/

&#91;2&#93; Lukas Bossard, Matthieu Guillaumin, Luc Van Gool. Food-101 – Mining Discriminative Components with Random Forests
https://www.vision.ee.ethz.ch/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf
