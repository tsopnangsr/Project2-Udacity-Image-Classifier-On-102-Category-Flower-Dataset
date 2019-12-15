# Project2-Udacity-Image-Classifier-On-102-Category-Flower-Dataset
Developing an AI application For Image Classification On 102 Category Flower Dataset.


## Introduction
This is an application to predict flowers categories. The model uses transfer learning on a VGG19 pretrained model. With this application, you can input a flower image and get the category name.


## Dataset
The Dataset used [Is HERE](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) With 102 flower categories.
![Flowers](/assets/Flowers.png)


## Installation
* Numpy  
<code>conda install numpy</code>
* Matplotlib  
<code>conda install matplotlib</code>
* Pil  
<code>conda install -c anaconda pil</code>
* Pytorch  
<code>conda install torch</code>

## How to run scripts
__For train.py__  
Argument to be passed:
* data directory : <code>--data_dir</code>
* saved checkpoint directory : <code>--save_dir</code>
* gpu training : <code>--gpu</code>
* architecture : <code>--arch</code>
* learning rate : <code>--learning_rate</code>
* hidden units : <code>--hidden_units</code>
* number of epochs : <code>--epochs</code>

__For predict.py__  
Argument to be passed:
* input image directory : <code>--image</code>
* checkpoint directory : <code>--check_point</code>
* gpu usage : <code>--gpu</code>
* top k classes with probability : <code>--top_k</code>
* category to name : <code>--category_to_name</code>

first run:  
<code>python train.py --data_dir 'flowers' --gpu True --learning_rate 0.001</code>
ou
<code>python train.py --data_dir dir_path --gpu True --learning_rate 0.001</code>

after successful training run:  
<code>python predict.py --image image_name.jpg --checkpoint check_point.pth --gpy True</code>
or
<code>python predict.py --image 'flowers/test/2/image_05100.jpg' --check_point checkpoint.pth --gpu True</code>

# Basic Usage for command line

- Clone the repository use: `git clone https://github.com/tsopnangsr/Project2-Udacity-Image-Classifier-On-102-Category-Flower-Dataset.git`

## DEMO example on Jupyter notebook
*Note (They are also script version to run with Python)*
![Classification](/assets/udacity_ai_result_image_classifier.png)
