<h1>Gender and Age Detector</h1>

This Keras implementation of a VGG-face model detects gender and age from a face image as well as from a video in 
real time using webcam. In training, the IMDB's wiki_crop dataset is used. The dataset is preprocessed and the model is trained from scratch on 10500 cropped face images and tested on 4500 images. The dataset remains decent because of computational limit. Pre-trained weights for age and gender are utilized to boost the accuracy of the model.

<h3>Dependencies</h3>

- Python 3.6+
- Keras 2.0+
- OpenCV3</li>
- Tensorflow
- Numpy
- h5py
- Scipy

<h3>Usage</h3>

1. Install required packages and libraries in dependencies 

1. Clone this repository:

`git clone https://github.com/Trangle91/Gender_Age_Detector`

2. Download the dataset:

can be found in the dataset folder or [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

3. Run the following command for real-time performance:

`python3 detect_real_time.py`

**Note: Python 2 is not supported.

Pip has to be linked to Python 3+ (pip -V will display info).

Use pip3 instead if pip is linked to Python 2.7.

<h3>Traning</h3>

The model was trained and tested on wiki crop dataset. Depending on the hardware configuration of your system, the execution time will vary. On CPU, training will be slow. If you have an Nvidia GPU, then you can install tensorflow-gpu package. It will make things run a lot faster.

<h3>Demo with images</h3>

<h5>Input</h5>
<img src="https://raw.githubusercontent.com/Trangle91/Gender_Age_Detector/master/Sample inputs/katy.jpg" width="500" height="500">

<h5>Output</h5>
<img src="https://raw.githubusercontent.com/Trangle91/Gender_Age_Detector/master/Sample outputs/output6.png" width="500" height="500">

<h3>Demo in real time</h3>

![](demo/demo.gif)

<h3>Limitations</h3>

Due to short computational power, two major confines were confronted, the size of the dataset and the model 

employed. The dataset, wiki-crop, was small and only cropped faces were comprised. The model could have been

built with more layers and more complex architecture such as ResNet model. A ResNet50 model with 50 complex

layers was attempted to try with wiki-crop, but it did not improve the result because it was not suitable for

this kind of small dataset, so VGG was used. The MAE used as a metrics was not as high as expectation. 

<h3>Further improvement</h3>

If you want better results:

- use a larger dataset such as IMDB's face datasets.
- use pretrained weights for age and gender detections for accuracy boosting.
- use a bigger network such as ResNet, or VGG with more layers or finetune it.

<h3>Lisence</h3>

The IMDB-WIKI dataset used in this project is originally provided under the following conditions.

>Please notice that this dataset is made available for academic research purpose only. All the images are collected from the Internet, and the copyright belongs to the original owners. If any of the images belongs to you and you would like it removed, please kindly inform us, we will remove it from our dataset immediately.
