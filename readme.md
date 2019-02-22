readme

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



Containing 101,000 images of 101 categories.
Each category contains:
training set: 750 
test set: 250


Dataset
[1] Lukas Bossard, Matthieu Guillaumin, Luc Van Gool. Food-101 – Mining Discriminative Components with Random Forests. https://www.vision.ee.ethz.ch/datasets_extra/food-101/
[2] Lukas Bossard1 Matthieu Guillaumin1 Luc Van Gool1. Food-101 – Mining Discriminative Components with Random Forests
https://www.vision.ee.ethz.ch/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf
