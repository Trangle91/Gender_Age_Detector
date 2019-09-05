<h1>Gender and Age Detector</h1>
<br>This Keras implementation of a VGG-face model detects gender and age from a face image as well as from a video in </br>
<br>real time using webcam. In training, the IMDB's wiki_crop dataset is used.</br>
<br>The dataset is preprpocessed and the model is trained from scratch on 10500 cropped face images and tested on  </br>
<br> 4500 images. The dataset remains decent because of computational limit. Pre-trained weights for age and gender are </br> 
<br>utilized to boost the accuracy of the model.</br>

<h3>Dependencies</h3>
<ul>
<li>Python 3.6+</li>
<li>Keras 2.0+</li>
<li>OpenCV3</li>
<li>Tensorflow</li>
<li>Numpy</li>
<li>h5py</li>
<li>Scipy</li>
</ul>
<br>Install the required packages by executing the following command</br>
<br>$ pip install -r requirements.txt</br>
<br><b>Note: Python 2 is not supported.</b></br>
<br>Pip has to be linked to Python 3+ (pip -V will display info).</br>
<br>Use pip3 instead if pip is linked to Python 2.7.</br>
<h3>Usage</h3>
<h4>webcam</h4>
<br>$ python detect_real_time.py</br>
<br>If python command invokes python2 by default, use the following command</br>
<br>$ python3 detect_real_time.py</br>

<h3>Traning</h3>
<br>You can download the dataset on IMDB's website or from here. </br>
<br>Depending on the hardware configuration of your system, the execution time will vary. On CPU, training will be slow. If you </br> 
<br>have an Nvidia GPU, then you can install tensorflow-gpu package. It will make things run a lot faster.</br>

<h3>Demo</h3>
![My demo](demo/demo.gif)
