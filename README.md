# Face Recognition
## Overview
This is accomplished by Ralph Louis Gopez, Athaliah Palao, Kevin Ramos, and Miguel Soriano for an activity on the Intelligent Systems course. This project provides an easy to install, train, and deploy face recognition pipeline mainly with OpenCV. A face model for Haar Cascade algorithm is used to find the bounding boxes of faces then a Local Binary Patterns Histogram model (that you have to train) is used to specifically identify whose face it is.

## Installation
Install the package requirements `pip install -r requirements.txt`. Ideally, use a virtual environment with `python -m venv venv` command and enter the virtual environment through its activation script.

## Getting Training Data 
Remember, that all commands moving forward assumes you're in the root directory of the project. To get training data use the `batch_capture.py` script which will use an IP Webcam and save images of the faces found in the screen. 

An example usage provided is, simply change the values accordingly:
```
python scripts/batch_capture.py ralph 250 training http://192.168.100.156:8080/shot.jpg
```

This will use the IP Webcam at the specified address (assuming you use the IP Webcam application on Android), and it will save the files inside the `training` directory, with the label being the form of `ralph_1.jpg`, `ralph_2.jpg`, and so on until it reaches the limit of 250.

![Gathering Training Data GIF](/demo/gathering.gif)

## Training the Classifier
Now in the root directory create a file called `labels.txt` inside it you can put values like so:
```txt
1,ralph
2,kevin
3,athaliah
4,miguel
```
We need to do this as the model can only output numerical values, so we essentially define a dictionary through the `labels.txt` file which it will use as a reference. Now images for training such as `ralph_1.jpg` will be matched with the `ralph` and predicted as label `1`.

We then run `train_lpbh.py`. An example usage is:
```
python scripts/train_lpbh.py training labels.txt model.yml
```

Which will use the `training` directory we created earlier with the images we captured, use the `labels.txt` as a dictionary and finally save our model as `model.yml`. Remember, you can change the arguments as needed.

## Using the Classifier
Now to use the classifier in real time, we just have to provide it with the model, the labels, and the IP address of the camera. Again an example usage assuming the previous steps were followed correctly:
```
python scripts/recognize_faces.py model.yml labels.txt http://192.168.100.156:8080/shot.jpg
```

## How it works
Programming object detection for a computer isn't as simple as it sounds for us humans that can do it intuitively. An object detection system requires drawing bounding boxes across the entire screen with varying sizes and tries to identify if something is within those boxes. Only then can it be classified. In real-time scenarios, that is a computationally expensive tasks when you consider the fact that all that has to run within at least 1/24th of a second.

This project uses OpenCV wrapper library on Python3 and a Haar Cascade algorithm with a pre-trained face model which can find the bounding boxes that match a face but cannot identify whose face it is.

Once the bounding boxes are found, each of those bounding boxes are sent to a trained (by you or us, the users) Local Binary Patterns Histogram model to try and identify whose face it is.