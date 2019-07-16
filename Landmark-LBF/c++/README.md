# LBF Model for Facial Landmark Detection in OpenCV
This repository contains code snippets to train and run a demo on webcam for detecting facial landmarks using the local binary features model in the OpenCV Facemark API.
 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

* To run the webcam demo, go to the [build](./build) folder and run the following commands.
```
cmake ..
make
./lbf-webcam-demo ../models/haarcascade_frontalface_alt2.xml ../models/LBF.yaml
```
* To train the model, download the datasets available [here](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) and add the paths to the images and their corresponding landmark points in seperate text files. Refer [this](./train/) folder for the files used for training.
```
cmake ..
make
./lbf-train ../models/haarcascade_frontalface_alt2.xml ../models/my_LBF.yaml ../train/images_train.txt ../train/points_train.txt
```
Now you can find the trained model in the [models](./models/) folder. Note that for all the demo purposes the LBF.yaml model in the models folder was used.

## Webcam Demo

<p align='center'>
  <img src='./output/webcam_landmarks.gif' alt='input'/>
</p>

## Future Work

*  Need to add python bindings to the OpenCV Facemark LBF and then implement this demo in python.

## Author

* Saiteja Talluri - [saiteja-talluri](https://github.com/saiteja-talluri)

## Acknowledgements

* I would like to thank [Laksono Kurnianggoro](https://github.com/kurnianggoro) for his wonderful implementation of LBF in the OpenCV Facemark API.
