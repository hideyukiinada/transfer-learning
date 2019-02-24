# Preparing Food-101 Dataset for Transfer Learning
Hide Inada

The dataset is available at:
https://www.vision.ee.ethz.ch/datasets_extra/food-101/

Head over to the page, and download the dataset to your local disk.
_Please be sure to check if you are allowed to use the dataset for your specific purpose._

The dataset has 101 classes of food images and the total number of images is 101,000.

Once you download the food-101.tar.gz, you can expand by typing:
```
tar zxvf food-101.tar.gz
```

This will expand images to food-101/images directory.
Note that this directory contains all the files for both training and test set.  mega/test.txt contains the list of files that are in the test set.

retrain.py automatically splits the data into training set, validation set and test set while training to calculate stats, so you do not need to split the dataset beforehand unless you want to have a control over how datasets are split for further analysis.

If you want to split the dataset, you can use my script below:
https://github.com/hideyukiinada/transfer-learning/blob/master/tools/food-101-split

# References
&#91;1&#93; Lukas Bossard, Matthieu Guillaumin, Luc Van Gool. Food-101 – Mining Discriminative Components with Random Forests. https://www.vision.ee.ethz.ch/datasets_extra/food-101/

&#91;2&#93; Lukas Bossard, Matthieu Guillaumin, Luc Van Gool. Food-101 – Mining Discriminative Components with Random Forests
https://www.vision.ee.ethz.ch/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf
