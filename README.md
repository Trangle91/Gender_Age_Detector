<h1>Gender and Age Detector</h1>
<br>This Keras implementation of a VGG-face model detects gender and age from a face image as well as from a video in </br>
<br>real time using webcam. In training, the IMDB's wiki_crop dataset is used.</br>
<br>The dataset is preprpocessed and the model is trained from scratch on 10500 cropped face images and tested on  </br>
<br> 4500 images. The dataset remains decent because of computational limit. Pre-trained weights for age and gender are </br> 
<br>utilized to boost the accuracy of the model.</br>

###Dependencies
<ul>
<li>Python 3.6+</li>
<li>Keras 2.0+</li>
<li>OpenCV3</li>
<li>Tensorflow</li>
<li>Numpy</li>
<li>h5py</li>
<li>Scipy</li>
</ul>

###Usage

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


###Traning
<br>You can download the dataset on IMDB's website or from here. </br>
<br>Depending on the hardware configuration of your system, the execution time will vary. On CPU, training will be slow. If </br> 
<br>you have an Nvidia GPU, then you can install tensorflow-gpu package. It will make things run a lot faster.</br>

###Demo
![](demo/demo.gif)
